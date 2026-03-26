"""Answer Verifier — the critical missing piece.

Instead of just voting on model answers, we:
1. Parse the original problem into a SymPy expression
2. Try to compute the CORRECT answer independently
3. Check each model's answer against the computed value
4. If a model matches SymPy, use that answer with HIGH confidence
5. If none match, report SymPy's answer (or "unverifiable" if SymPy can't solve it)

This is the Aryabhatta principle: compute truth independently, don't just poll opinions.
"""
from __future__ import annotations

import re

import structlog
from sympy import (
    symbols, sqrt, simplify, solve, Eq, N, Rational,
    oo, pi, sin, cos, tan, log, exp, factorial,
    radsimp, nsimplify, sympify,
)
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
    function_exponentiation,
)

logger = structlog.get_logger()

TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
    function_exponentiation,
)


def _try_parse_answer(answer_str: str):
    """Try to parse a model's answer into a SymPy expression."""
    if not answer_str:
        return None

    s = answer_str.strip()
    # Remove $, \boxed, etc
    s = re.sub(r"^\$+|\$+$", "", s)
    m = re.match(r"^\\boxed\{(.+)\}$", s)
    if m:
        s = m.group(1)

    # Clean LaTeX
    s = s.replace("\\sqrt", "sqrt").replace("\\frac", "")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("\\pi", "pi").replace("\\infty", "oo")
    s = s.replace("^", "**").replace("{", "(").replace("}", ")")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\", "")

    try:
        return parse_expr(s, transformations=TRANSFORMS)
    except Exception:
        pass

    # Try direct sympify
    try:
        return sympify(s)
    except Exception:
        pass

    return None


def _try_evaluate_expression(problem_text: str) -> dict | None:
    """Try to parse the problem and compute the answer with SymPy.

    Returns a dict with:
      - 'sympy_answer': the computed answer (SymPy expression)
      - 'numerical_value': float approximation
      - 'method': how we computed it
    Or None if we can't compute it.
    """
    text = problem_text.strip()

    # === Strategy 1: Extract "solve for x: equation" ===
    solve_match = re.search(
        r"(?:solve\s+(?:for\s+)?(\w)\s*[:\-]?\s*)(.*)",
        text, re.IGNORECASE,
    )
    if solve_match:
        var_name = solve_match.group(1)
        equation_str = solve_match.group(2).strip()
        try:
            var = symbols(var_name)
            # Split on = sign
            if "=" in equation_str:
                parts = equation_str.split("=", 1)
                left = parse_expr(parts[0].strip().replace("^", "**"), transformations=TRANSFORMS)
                right = parse_expr(parts[1].strip().replace("^", "**"), transformations=TRANSFORMS)
                solutions = solve(Eq(left, right), var)
                if solutions:
                    if len(solutions) == 1:
                        return {
                            "sympy_answer": solutions[0],
                            "numerical_value": float(N(solutions[0])),
                            "method": "sympy_solve",
                        }
                    else:
                        return {
                            "sympy_answer": solutions,
                            "numerical_value": [float(N(s)) for s in solutions],
                            "method": "sympy_solve_multiple",
                        }
        except Exception as e:
            logger.debug("sympy_solve_failed", error=str(e))

    # === Strategy 2: "What is <expression>?" or "Calculate <expression>" ===
    calc_match = re.search(
        r"(?:what\s+is|calculate|compute|evaluate|find(?:\s+the\s+value\s+of)?)\s+(.+?)(?:\?|$)",
        text, re.IGNORECASE,
    )
    if calc_match:
        expr_str = calc_match.group(1).strip()
        try:
            expr_str = expr_str.replace("^", "**").replace("×", "*").replace("÷", "/")
            expr = parse_expr(expr_str, transformations=TRANSFORMS)
            result = simplify(expr)
            return {
                "sympy_answer": result,
                "numerical_value": float(N(result)),
                "method": "sympy_evaluate",
            }
        except Exception as e:
            logger.debug("sympy_evaluate_failed", error=str(e))

    # === Strategy 3: Integrals — "integral of f(x) dx" ===
    int_match = re.search(
        r"(?:integral|integrate|antiderivative)\s+(?:of\s+)?(.+?)\s*(?:dx|d\s*x|with\s+respect\s+to\s+x)?\s*(?:\?|$)",
        text, re.IGNORECASE,
    )
    if int_match:
        from sympy import integrate as sym_integrate, Symbol
        expr_str = int_match.group(1).strip()
        try:
            expr_str = expr_str.replace("^", "**")
            x = Symbol('x')
            expr = parse_expr(expr_str, local_dict={"x": x, "e": exp(1)}, transformations=TRANSFORMS)
            result = sym_integrate(expr, x)
            return {
                "sympy_answer": result,
                "numerical_value": None,  # Symbolic, not numeric
                "method": "sympy_integrate",
            }
        except Exception as e:
            logger.debug("sympy_integrate_failed", error=str(e))

    # === Strategy 4: Derivatives — "derivative of f(x)" or "d/dx f(x)" ===
    deriv_match = re.search(
        r"(?:derivative|differentiate|d/dx)\s+(?:of\s+)?(.+?)(?:\?|$)",
        text, re.IGNORECASE,
    )
    if deriv_match:
        from sympy import diff as sym_diff, Symbol
        expr_str = deriv_match.group(1).strip()
        try:
            expr_str = expr_str.replace("^", "**").replace("ln(", "log(")
            x = Symbol('x')
            expr = parse_expr(expr_str, local_dict={"x": x, "e": exp(1)}, transformations=TRANSFORMS)
            result = sym_diff(expr, x)
            return {
                "sympy_answer": result,
                "numerical_value": None,
                "method": "sympy_derivative",
            }
        except Exception as e:
            logger.debug("sympy_derivative_failed", error=str(e))

    # === Strategy 5: Limits — "limit of f(x) as x approaches a" ===
    lim_match = re.search(
        r"limit\s+(?:of\s+)?(.+?)\s+as\s+(\w)\s+(?:approaches|->|→|goes\s+to)\s+(.+?)(?:\?|$)",
        text, re.IGNORECASE,
    )
    if lim_match:
        from sympy import limit as sym_limit, Symbol
        expr_str = lim_match.group(1).strip()
        var_name = lim_match.group(2).strip()
        point_str = lim_match.group(3).strip()
        try:
            expr_str = expr_str.replace("^", "**")
            var = Symbol(var_name)
            expr = parse_expr(expr_str, local_dict={var_name: var, "e": exp(1)}, transformations=TRANSFORMS)
            point_str = point_str.replace("infinity", "oo").replace("inf", "oo")
            point = parse_expr(point_str, transformations=TRANSFORMS)
            result = sym_limit(expr, var, point)
            return {
                "sympy_answer": result,
                "numerical_value": float(N(result)) if result.is_number else None,
                "method": "sympy_limit",
            }
        except Exception as e:
            logger.debug("sympy_limit_failed", error=str(e))

    # === Strategy 6: Summation — "sum of 1+2+...+n" ===
    sum_match = re.search(
        r"sum\s+(?:of\s+)?(\d+)\s*\+\s*(\d+)\s*\+\s*(?:\.\.\.|…)\s*\+\s*(\d+)",
        text, re.IGNORECASE,
    )
    if sum_match:
        from sympy import summation, Symbol
        start = int(sum_match.group(1))
        step = int(sum_match.group(2)) - start
        end = int(sum_match.group(3))
        try:
            if step == 1:
                # Arithmetic series: n*(n+1)/2
                result = end * (end + 1) // 2 - (start - 1) * start // 2
            else:
                k = Symbol('k')
                result = summation(start + (k - 1) * step, (k, 1, (end - start) // step + 1))
            return {
                "sympy_answer": result,
                "numerical_value": float(result),
                "method": "sympy_summation",
            }
        except Exception as e:
            logger.debug("sympy_summation_failed", error=str(e))

    # === Strategy 8: Try to parse the ENTIRE text as a math expression ===
    try:
        cleaned = text.replace("^", "**").replace("×", "*").replace("÷", "/")
        cleaned = re.sub(r"\s*[=]\s*\?.*$", "", cleaned)
        cleaned = re.sub(r"\s*find\s+.*$", "", cleaned, flags=re.IGNORECASE)
        expr = parse_expr(cleaned, transformations=TRANSFORMS)
        result = simplify(expr)
        return {
            "sympy_answer": result,
            "numerical_value": float(N(result)),
            "method": "sympy_direct",
        }
    except Exception:
        pass

    # === Strategy 9: Extract expression from "expression = sqrt(a) + sqrt(b)" pattern ===
    eq_match = re.search(r"(.+?)\s*=\s*sqrt\(a\)\s*\+\s*sqrt\(b\)", text, re.IGNORECASE)
    if eq_match:
        lhs = eq_match.group(1).strip()
        try:
            lhs_clean = lhs.replace("^", "**").replace("×", "*")
            lhs_clean = lhs_clean.replace("\\sqrt", "sqrt").replace("\\frac", "Rational")
            # Try to parse and evaluate the LHS
            expr = parse_expr(lhs_clean, transformations=TRANSFORMS)
            result = radsimp(expr)
            return {
                "sympy_answer": result,
                "numerical_value": float(N(result)),
                "method": "sympy_lhs_evaluate",
            }
        except Exception:
            pass

    return None


def verify_answers(
    problem_text: str,
    answer_agreement: dict[str, list[str]],
    tolerance: float = 0.01,
) -> dict:
    """Verify model answers against SymPy-computed ground truth.

    Returns:
        {
            "sympy_computed": True/False (whether SymPy could compute independently)
            "sympy_answer": str (SymPy's answer, if computed)
            "sympy_numerical": float | None
            "method": str
            "verified_answers": {answer: True/False/None}
            "best_answer": str | None (the verified answer, or None)
            "best_answer_models": list[str] (models that had the verified answer)
        }
    """
    result = {
        "sympy_computed": False,
        "sympy_answer": None,
        "sympy_numerical": None,
        "method": None,
        "verified_answers": {},
        "best_answer": None,
        "best_answer_models": [],
    }

    # Try to compute independently
    computation = _try_evaluate_expression(problem_text)

    if computation is None:
        logger.info("sympy_cannot_compute", problem=problem_text[:80])
        # Can't compute — mark all answers as unverified
        for ans in answer_agreement:
            result["verified_answers"][ans] = None
        return result

    result["sympy_computed"] = True
    result["sympy_answer"] = str(computation["sympy_answer"])
    result["method"] = computation["method"]

    result["sympy_numerical"] = computation["numerical_value"]
    if computation["numerical_value"] is None:
        target_values = None  # Symbolic answer (integrals, etc.)
    elif isinstance(computation["numerical_value"], list):
        target_values = computation["numerical_value"]
    else:
        target_values = [computation["numerical_value"]]

    logger.info(
        "sympy_computed_answer",
        answer=str(computation["sympy_answer"]),
        numerical=result["sympy_numerical"],
        method=computation["method"],
    )

    # Check each model's answer against SymPy's answer
    is_symbolic = computation["numerical_value"] is None  # e.g. integrals, derivatives

    for answer_str, models in answer_agreement.items():
        # Clean model answer: strip "+C", "+c", "+ C" (constant of integration)
        clean_ans = re.sub(r"\s*\+\s*[Cc]\s*$", "", answer_str.strip())
        clean_ans = re.sub(r"\s*\+\s*(?:constant|CONSTANT)\s*$", "", clean_ans)

        parsed = _try_parse_answer(clean_ans)
        if parsed is None:
            result["verified_answers"][answer_str] = None
            continue

        try:
            matched = False

            if is_symbolic:
                # Symbolic comparison (integrals, derivatives, etc.)
                sym_answer = computation["sympy_answer"]
                try:
                    diff_expr = simplify(parsed - sym_answer)
                    matched = diff_expr == 0
                except Exception:
                    pass

                # Also try with trigsimp for trig expressions
                if not matched:
                    try:
                        from sympy import trigsimp
                        matched = trigsimp(parsed - sym_answer) == 0
                    except Exception:
                        pass

                # Numerical spot-check at a few x values
                if not matched:
                    try:
                        x = symbols('x')
                        test_vals = [0.5, 1.0, 2.0]
                        sym_answer_expr = computation["sympy_answer"]
                        all_close = True
                        for tv in test_vals:
                            v1 = complex(N(parsed.subs(x, tv)))
                            v2 = complex(N(sym_answer_expr.subs(x, tv)))
                            if abs(v1 - v2) > 0.01:
                                all_close = False
                                break
                        matched = all_close
                    except Exception:
                        pass

            else:
                # Numeric comparison
                try:
                    answer_val = float(N(parsed))
                    if isinstance(target_values, list):
                        matched = any(abs(answer_val - tv) < tolerance for tv in target_values)
                    else:
                        matched = abs(answer_val - target_values) < tolerance
                except (TypeError, ValueError):
                    pass

                # Symbolic fallback
                if not matched and not isinstance(computation["sympy_answer"], list):
                    try:
                        sym_answer = computation["sympy_answer"]
                        if hasattr(sym_answer, '__sub__'):
                            matched = simplify(parsed - sym_answer) == 0
                    except Exception:
                        pass

            result["verified_answers"][answer_str] = matched

            if matched and result["best_answer"] is None:
                result["best_answer"] = answer_str
                result["best_answer_models"] = models
                logger.info("answer_verified", answer=answer_str, models=models)

        except Exception:
            result["verified_answers"][answer_str] = None

    # If no model answer matched but SymPy computed an answer, use SymPy's
    if result["best_answer"] is None and result["sympy_computed"]:
        result["best_answer"] = str(computation["sympy_answer"])
        result["best_answer_models"] = ["SymPy (independent computation)"]
        logger.info("using_sympy_answer", answer=result["best_answer"])

    return result
