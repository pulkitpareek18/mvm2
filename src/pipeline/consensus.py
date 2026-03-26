"""Consensus engine with SymPy-powered mathematical equivalence checking.

The key insight: models often give the SAME correct answer in different formats.
e.g., "e^x*sin(x) + e^x*cos(x)" vs "exp(x)(sin(x)+cos(x))" vs "(sin x + cos x)e^x"
Pure string matching treats these as different answers, tanking confidence.

This module uses SymPy to check if two answers are mathematically equivalent
before falling back to string comparison.
"""
from __future__ import annotations

import re
from collections import Counter

import structlog

from ..models.solution import ModelSolution
from ..models.verification import StepAlignment

logger = structlog.get_logger()


# ── SymPy equivalence cache ──────────────────────────────────
_equiv_cache: dict[tuple[str, str], bool] = {}


def _try_sympy_parse(expr_str: str):
    """Try to parse a math expression into a SymPy object. Returns None on failure."""
    try:
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations,
            implicit_multiplication_application, convert_xor,
            function_exponentiation,
        )
        transformations = standard_transformations + (
            implicit_multiplication_application,
            convert_xor,
            function_exponentiation,
        )

        s = expr_str.strip()

        # Convert Unicode superscripts/subscripts to ASCII
        unicode_sup = {"⁰":"0","¹":"1","²":"2","³":"3","⁴":"4","⁵":"5","⁶":"6","⁷":"7","⁸":"8","⁹":"9","ⁿ":"n"}
        for uc, digit in unicode_sup.items():
            s = s.replace(uc, f"**{digit}")
        unicode_sub = {"₀":"0","₁":"1","₂":"2","₃":"3","₄":"4","₅":"5","₆":"6","₇":"7","₈":"8","₉":"9","ₙ":"n"}
        for uc, digit in unicode_sub.items():
            s = s.replace(uc, f"_{digit}")

        # Strip $ wrappers
        if s.startswith("$$") and s.endswith("$$"):
            s = s[2:-2].strip()
        elif s.startswith("$") and s.endswith("$"):
            s = s[1:-1].strip()

        # Strip \boxed{}
        m = re.match(r"^\\boxed\{(.+)\}$", s)
        if m:
            s = m.group(1)

        # Convert LaTeX-isms to SymPy-parseable form
        s = s.replace("\\frac", "frac").replace("\\sqrt", "sqrt")
        s = s.replace("\\cdot", "*").replace("\\times", "*")
        s = s.replace("\\pi", "pi").replace("\\infty", "oo")
        s = s.replace("\\sin", "sin").replace("\\cos", "cos").replace("\\tan", "tan")
        s = s.replace("\\log", "log").replace("\\ln", "log").replace("\\exp", "exp")
        s = s.replace("\\left", "").replace("\\right", "")
        s = s.replace("^", "**").replace("{", "(").replace("}", ")")

        # Handle e^x → exp(x): replace e**<term> with exp(<term>)
        # e**x → exp(x), e**(2*x) → exp(2*x), e**(x+1) → exp(x+1)
        # But NOT: ae**x (variable 'ae')
        s = re.sub(r'\be\*\*\(([^)]+)\)', r'exp(\1)', s)  # e**(expr) → exp(expr)
        s = re.sub(r'\be\*\*([a-zA-Z0-9_]+)', r'exp(\1)', s)  # e**x → exp(x)

        # Handle frac(a)(b) -> (a)/(b)
        s = re.sub(r"frac\(([^)]+)\)\(([^)]+)\)", r"(\1)/(\2)", s)

        # Common notations
        s = s.replace("×", "*").replace("÷", "/")

        return parse_expr(s, transformations=transformations)
    except Exception:
        return None


def _sympy_equivalent(a: str, b: str) -> bool | None:
    """Check if two expressions are mathematically equivalent using SymPy.

    Returns:
        True  — expressions are equivalent
        False — expressions are provably different
        None  — cannot determine (parse failure, timeout, etc.)
    """
    cache_key = (a, b)
    if cache_key in _equiv_cache:
        return _equiv_cache[cache_key]

    expr_a = _try_sympy_parse(a)
    expr_b = _try_sympy_parse(b)

    if expr_a is None or expr_b is None:
        return None

    try:
        from sympy import simplify, N, Symbol, trigsimp

        # Strategy 1: Direct simplify(a - b) == 0
        diff = simplify(expr_a - expr_b)
        if diff == 0:
            _equiv_cache[cache_key] = True
            return True

        # Strategy 2: Trigsimp for trig expressions
        diff_trig = trigsimp(expr_a - expr_b)
        if diff_trig == 0:
            _equiv_cache[cache_key] = True
            return True

        # Strategy 3: Numerical evaluation at test points
        x = Symbol('x')
        free_syms = (expr_a.free_symbols | expr_b.free_symbols)
        if len(free_syms) <= 1:
            test_vals = [0.5, 1.0, 1.5, 2.0, -0.5]
            sym = list(free_syms)[0] if free_syms else x
            all_match = True
            tested = 0
            for val in test_vals:
                try:
                    va = complex(expr_a.subs(sym, val).evalf())
                    vb = complex(expr_b.subs(sym, val).evalf())
                    if abs(va - vb) > 1e-6:
                        all_match = False
                        break
                    tested += 1
                except Exception:
                    continue
            if tested >= 3 and all_match:
                _equiv_cache[cache_key] = True
                return True
            if not all_match:
                _equiv_cache[cache_key] = False
                return False

        # Strategy 4: For pure numbers, compare numerically
        if not free_syms:
            try:
                va = complex(N(expr_a))
                vb = complex(N(expr_b))
                result = abs(va - vb) < 1e-6
                _equiv_cache[cache_key] = result
                return result
            except Exception:
                pass

        return None
    except Exception:
        return None


def _normalize_answer(answer: str) -> str:
    """Normalize a math answer for string comparison.

    Strips whitespace, removes trailing periods, normalizes fractions and decimals.
    """
    s = answer.strip().rstrip(".").strip()
    # Remove spaces around operators
    s = re.sub(r"\s*([=+\-*/^])\s*", r"\1", s)
    # Normalize common patterns
    s = s.replace("×", "*").replace("÷", "/")
    # Try to evaluate simple fractions to decimals for comparison
    try:
        if "/" in s and not any(c.isalpha() for c in s):
            val = eval(s)
            if isinstance(val, (int, float)):
                if isinstance(val, int) or val == int(val):
                    return str(int(val))
                return f"{val:.10g}"
    except Exception:
        pass
    return s.lower()


def _results_match(a: str | None, b: str | None) -> bool:
    """Check if two results are equivalent.

    Uses a 3-tier comparison:
    1. Exact string match after normalization (fast)
    2. SymPy symbolic equivalence (catches different-but-equal formats)
    3. Numerical evaluation (fallback for complex expressions)
    """
    if a is None or b is None:
        return a is None and b is None

    # Tier 1: String match
    na, nb = _normalize_answer(a), _normalize_answer(b)
    if na == nb:
        return True

    # Tier 2: SymPy equivalence
    sympy_result = _sympy_equivalent(a, b)
    if sympy_result is True:
        logger.debug("sympy_equiv_match", a=a[:50], b=b[:50])
        return True
    if sympy_result is False:
        return False

    # Tier 3: If both can be evaluated to numbers, compare numerically
    try:
        va = float(na)
        vb = float(nb)
        return abs(va - vb) < 1e-6
    except (ValueError, TypeError):
        pass

    return False


def compute_step_consensus(aligned_steps: list[StepAlignment], threshold: float) -> list[StepAlignment]:
    """Compute agreement ratio for each aligned step and flag disagreements."""
    for step in aligned_steps:
        present_steps = {
            name: s for name, s in step.model_steps.items() if s is not None
        }
        if not present_steps:
            step.agreement_ratio = 0.0
            step.flagged = True
            continue

        # Group by result using equivalence matching
        results = [(name, s.result or "") for name, s in present_steps.items()]
        groups: dict[str, list[str]] = {}
        for name, result in results:
            matched = False
            for key in groups:
                if _results_match(result, key):
                    groups[key].append(name)
                    matched = True
                    break
            if not matched:
                groups[result] = [name]

        # Largest agreement group
        largest_group = max(groups.values(), key=len)
        step.agreement_ratio = len(largest_group) / len(present_steps)
        step.flagged = step.agreement_ratio < threshold

        if step.flagged:
            logger.warning(
                "step_disagreement",
                step=step.canonical_step_number,
                agreement=step.agreement_ratio,
                groups={k[:40]: v for k, v in groups.items()},
            )

    return aligned_steps


def compute_answer_agreement(solutions: list[ModelSolution]) -> dict[str, list[str]]:
    """Group models by their final answer using SymPy equivalence."""
    answer_groups: dict[str, list[str]] = {}

    for sol in solutions:
        if sol.error or not sol.final_answer:
            continue

        matched = False
        for key in answer_groups:
            if _results_match(sol.final_answer, key):
                answer_groups[key].append(sol.model_name)
                matched = True
                break
        if not matched:
            answer_groups[sol.final_answer] = [sol.model_name]

    return answer_groups


def get_consensus_answer(answer_agreement: dict[str, list[str]]) -> tuple[str, float]:
    """Return the majority answer and confidence score."""
    if not answer_agreement:
        return "", 0.0

    total_models = sum(len(models) for models in answer_agreement.values())
    best_answer = max(answer_agreement, key=lambda k: len(answer_agreement[k]))
    confidence = len(answer_agreement[best_answer]) / total_models

    return best_answer, confidence
