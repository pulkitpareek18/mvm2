from __future__ import annotations

import signal
import structlog

logger = structlog.get_logger()

# Allowed modules for sandboxed execution
ALLOWED_MODULES = {"sympy", "math", "fractions"}

# Maximum execution time in seconds
EXEC_TIMEOUT = 5


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out")


def execute_verification_code(code: str) -> bool | None:
    """Execute generated verification code in a restricted environment.

    Returns:
        True  — code executed and `result` variable is truthy
        False — code executed and `result` variable is falsy
        None  — execution failed (timeout, error, etc.)
    """
    if not code or not code.strip():
        return None

    # Basic safety checks
    dangerous_patterns = [
        "import os",
        "import sys",
        "import subprocess",
        "__import__",
        "open(",
        "exec(",
        "eval(",
        "compile(",
        "globals(",
        "locals(",
        "getattr(",
        "setattr(",
        "delattr(",
        "__builtins__",
    ]

    code_lower = code.lower()
    for pattern in dangerous_patterns:
        if pattern.lower() in code_lower:
            logger.warning("dangerous_code_blocked", pattern=pattern)
            return None

    # Create restricted globals
    import sympy
    import math
    import fractions

    restricted_globals = {
        "__builtins__": {
            "range": range,
            "len": len,
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "abs": abs,
            "round": round,
            "sum": sum,
            "min": min,
            "max": max,
            "True": True,
            "False": False,
            "None": None,
            "print": lambda *args, **kwargs: None,  # no-op print
        },
        "sympy": sympy,
        "math": math,
        "fractions": fractions,
    }

    # Add commonly used sympy functions to top-level scope
    for name in [
        "symbols",
        "Symbol",
        "simplify",
        "expand",
        "factor",
        "solve",
        "Eq",
        "sqrt",
        "Rational",
        "pi",
        "oo",
        "sin",
        "cos",
        "tan",
        "log",
        "exp",
        "diff",
        "integrate",
        "limit",
        "parse_expr",
        "N",
    ]:
        if hasattr(sympy, name):
            restricted_globals[name] = getattr(sympy, name)

    from sympy.parsing.sympy_parser import parse_expr

    restricted_globals["parse_expr"] = parse_expr

    restricted_locals: dict = {}

    # Set timeout
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(EXEC_TIMEOUT)

    try:
        exec(code, restricted_globals, restricted_locals)
        result = restricted_locals.get("result")

        if result is None:
            return None

        return bool(result)

    except TimeoutError:
        logger.warning("code_execution_timeout", code_preview=code[:100])
        return None
    except Exception as e:
        logger.warning("code_execution_error", error=str(e), code_preview=code[:100])
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
