from __future__ import annotations

import re

import structlog
from sympy import (
    simplify, sympify, N, oo, pi,
    sin, cos, tan, log, exp, sqrt,
    diff, integrate, limit, Symbol, symbols,
    Matrix, Rational, trigsimp, expand_trig,
    series, factorial, binomial,
)
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
    function_exponentiation,
)

from ..models.verification import StepAlignment

logger = structlog.get_logger()

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
    function_exponentiation,
)


def _clean_expression(expr: str) -> str:
    """Clean a math expression string for SymPy parsing."""
    s = expr.strip()
    # Remove common prefixes
    for prefix in ["=", "→", "=>", "->", ":"]:
        s = s.lstrip(prefix).strip()

    # Replace common notation
    s = s.replace("×", "*").replace("÷", "/").replace("^", "**")
    s = s.replace("π", "pi").replace("∞", "oo")
    s = s.replace("√", "sqrt")
    # Handle common trig/log notation
    s = s.replace("ln(", "log(")
    return s


def _try_parse(expr: str):
    """Try to parse a math expression into a SymPy object. Returns None on failure."""
    try:
        cleaned = _clean_expression(expr)
        return parse_expr(cleaned, transformations=TRANSFORMATIONS)
    except Exception:
        return None


def _try_parse_matrix(expr: str):
    """Try to parse a matrix expression like [[1,2],[3,4]]."""
    try:
        cleaned = _clean_expression(expr)
        if "[[" in cleaned:
            return Matrix(eval(cleaned))
    except Exception:
        pass
    return None


def _try_numeric_compare(a: str, b: str, tolerance: float = 1e-6) -> bool | None:
    """Try to compare two values numerically. Returns None if not possible."""
    try:
        va = float(_clean_expression(a))
        vb = float(_clean_expression(b))
        return abs(va - vb) < tolerance
    except (ValueError, TypeError):
        return None


def verify_step_symbolically(step: StepAlignment) -> bool | None:
    """Verify a single step using SymPy.

    Supports: arithmetic, algebra, calculus, trigonometry, linear algebra, and more.

    Returns:
        True  — step is symbolically verified as correct
        False — step is symbolically verified as incorrect
        None  — cannot determine (expression too complex, unparseable, etc.)
    """
    present_steps = {
        name: s for name, s in step.model_steps.items() if s is not None
    }
    if len(present_steps) < 2:
        return None

    # Collect all results for this step
    results = []
    for name, s in present_steps.items():
        if s.result and s.result.upper() != "N/A":
            results.append((name, s.result))

    if len(results) < 2:
        return None

    # Strategy 1: Numeric comparison
    numeric_values: list[tuple[str, float]] = []
    for name, result in results:
        try:
            val = float(_clean_expression(result))
            numeric_values.append((name, val))
        except (ValueError, TypeError):
            pass

    if len(numeric_values) >= 2:
        reference = numeric_values[0][1]
        all_agree = all(abs(v - reference) < 1e-6 for _, v in numeric_values)
        if all_agree:
            return True
        return False

    # Strategy 2: Matrix comparison
    matrix_exprs: list[tuple[str, Matrix]] = []
    for name, result in results:
        mat = _try_parse_matrix(result)
        if mat is not None:
            matrix_exprs.append((name, mat))

    if len(matrix_exprs) >= 2:
        try:
            ref = matrix_exprs[0][1]
            all_equal = all(ref.equals(m) for _, m in matrix_exprs[1:])
            return all_equal
        except Exception:
            pass

    # Strategy 3: Symbolic comparison (algebra, calculus, trig)
    sympy_exprs: list[tuple[str, object]] = []
    for name, result in results:
        expr = _try_parse(result)
        if expr is not None:
            sympy_exprs.append((name, expr))

    if len(sympy_exprs) >= 2:
        try:
            ref_expr = sympy_exprs[0][1]
            # Try regular simplify first
            all_equal = all(
                simplify(expr - ref_expr) == 0
                for _, expr in sympy_exprs[1:]
            )
            if all_equal:
                return True

            # Try trigsimp for trig expressions
            all_equal_trig = all(
                trigsimp(expr - ref_expr) == 0
                for _, expr in sympy_exprs[1:]
            )
            if all_equal_trig:
                return True

            # Try numerical evaluation as fallback
            x = Symbol('x')
            test_vals = [0.5, 1.0, 2.0, -1.0]
            for val in test_vals:
                try:
                    ref_num = complex(ref_expr.subs(x, val).evalf())
                    for _, expr in sympy_exprs[1:]:
                        other_num = complex(expr.subs(x, val).evalf())
                        if abs(ref_num - other_num) > 1e-6:
                            return False
                except Exception:
                    continue

            return None  # Couldn't definitively prove or disprove
        except Exception:
            pass

    # Strategy 4: Check if math expression leads to result
    for name, s in present_steps.items():
        if s and s.mathematical_expression and s.result:
            math_expr = s.mathematical_expression
            result_expr = s.result

            # Handle equations like "3x + 6 = 15"
            if "=" in math_expr and "=" not in result_expr:
                pass  # Complex — skip

    return None  # Cannot determine


def verify_all_steps(
    aligned_steps: list[StepAlignment],
) -> tuple[list[StepAlignment], list[bool | None]]:
    """Run symbolic verification on all aligned steps."""
    results: list[bool | None] = []

    for step in aligned_steps:
        try:
            result = verify_step_symbolically(step)
            step.symbolic_verified = result
            results.append(result)

            if result is not None:
                logger.info(
                    "symbolic_verification",
                    step=step.canonical_step_number,
                    verified=result,
                )
        except Exception as e:
            logger.warning(
                "symbolic_verification_error",
                step=step.canonical_step_number,
                error=str(e),
            )
            step.symbolic_verified = None
            results.append(None)

    return aligned_steps, results
