"""Generate professional PDF reports via LaTeX compilation.

Writes a .tex file, compiles with pdflatex, returns the PDF bytes.
Produces publication-quality math rendering via native LaTeX.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from datetime import datetime

import structlog

logger = structlog.get_logger()


# Unicode math symbols → LaTeX command mapping
UNICODE_TO_LATEX = {
    # Greek letters
    "α": "\\alpha", "β": "\\beta", "γ": "\\gamma", "δ": "\\delta",
    "ε": "\\epsilon", "ζ": "\\zeta", "η": "\\eta", "θ": "\\theta",
    "λ": "\\lambda", "μ": "\\mu", "ν": "\\nu", "ξ": "\\xi",
    "π": "\\pi", "ρ": "\\rho", "σ": "\\sigma", "τ": "\\tau",
    "φ": "\\phi", "χ": "\\chi", "ψ": "\\psi", "ω": "\\omega",
    "Γ": "\\Gamma", "Δ": "\\Delta", "Θ": "\\Theta", "Λ": "\\Lambda",
    "Σ": "\\Sigma", "Π": "\\Pi", "Φ": "\\Phi", "Ψ": "\\Psi", "Ω": "\\Omega",
    # Operators and symbols
    "∫": "\\int", "∑": "\\sum", "∏": "\\prod", "√": "\\sqrt",
    "∞": "\\infty", "∂": "\\partial", "∇": "\\nabla",
    "×": "\\times", "÷": "\\div", "±": "\\pm", "∓": "\\mp",
    "·": "\\cdot", "°": "^{\\circ}", "′": "'",
    # Relations
    "≤": "\\leq", "≥": "\\geq", "≠": "\\neq", "≈": "\\approx",
    "≡": "\\equiv", "∝": "\\propto", "≪": "\\ll", "≫": "\\gg",
    # Set theory
    "∈": "\\in", "∉": "\\notin", "⊂": "\\subset", "⊃": "\\supset",
    "∪": "\\cup", "∩": "\\cap", "∅": "\\emptyset",
    # Arrows
    "→": "\\to", "←": "\\leftarrow", "⇒": "\\Rightarrow",
    "⇐": "\\Leftarrow", "↔": "\\leftrightarrow",
    # Blackboard bold
    "ℝ": "\\mathbb{R}", "ℤ": "\\mathbb{Z}", "ℕ": "\\mathbb{N}", "ℂ": "\\mathbb{C}",
    # Dots
    "…": "\\ldots", "⋯": "\\cdots",
    # Unicode superscripts → LaTeX superscripts
    "⁰": "^{0}", "¹": "^{1}", "²": "^{2}", "³": "^{3}",
    "⁴": "^{4}", "⁵": "^{5}", "⁶": "^{6}", "⁷": "^{7}",
    "⁸": "^{8}", "⁹": "^{9}", "⁺": "^{+}", "⁻": "^{-}",
    "ⁿ": "^{n}", "ⁱ": "^{i}",
    # Unicode subscripts → LaTeX subscripts
    "₀": "_{0}", "₁": "_{1}", "₂": "_{2}", "₃": "_{3}",
    "₄": "_{4}", "₅": "_{5}", "₆": "_{6}", "₇": "_{7}",
    "₈": "_{8}", "₉": "_{9}", "ₙ": "_{n}", "ₓ": "_{x}",
}


def _replace_unicode_math(text: str) -> str:
    """Replace Unicode math symbols with LaTeX commands.
    Also strips any remaining non-ASCII that pdflatex can't handle."""
    for unicode_char, latex_cmd in UNICODE_TO_LATEX.items():
        # Add trailing space after commands that are letters (prevents \intx)
        if latex_cmd.startswith("\\") and latex_cmd[-1].isalpha():
            text = text.replace(unicode_char, latex_cmd + " ")
        else:
            text = text.replace(unicode_char, latex_cmd)
    # Strip any remaining non-ASCII characters that would crash pdflatex
    # (keep basic ASCII + common accented chars that texlive handles)
    cleaned = []
    for ch in text:
        if ord(ch) < 128:
            cleaned.append(ch)
        elif ch in UNICODE_TO_LATEX:
            cleaned.append(UNICODE_TO_LATEX[ch])  # shouldn't happen after above
        else:
            cleaned.append("?")  # replace unknown Unicode with ?
    return "".join(cleaned)


def _escape_tex(text: str) -> str:
    """Escape special LaTeX characters in plain text.
    Also converts Unicode math symbols to LaTeX commands."""
    if not text:
        return ""
    # First convert Unicode math symbols
    text = _replace_unicode_math(text)
    # Order matters — & must be escaped before others that might produce &
    replacements = [
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _balance_braces(s: str) -> str:
    """Ensure braces are balanced in a LaTeX string."""
    depth = 0
    for ch in s:
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
    if depth > 0:
        s += '}' * depth
    elif depth < 0:
        s = '{' * (-depth) + s
    return s


def _math_to_latex(text: str) -> str:
    """Convert any math expression to safe LaTeX for PDF rendering.

    Strategy: strip ALL dollar signs, clean up, balance braces,
    then wrap in $...$ if it looks like math. Never pass through
    raw model output with $ signs — always re-wrap cleanly.
    """
    if not text or text.upper() in ("N/A", "NONE", "--", ""):
        return "N/A"

    s = text.strip()

    # Convert Unicode math symbols to LaTeX commands FIRST
    s = _replace_unicode_math(s)

    # Strip ALL dollar signs — we'll re-wrap ourselves
    s = s.replace("$$", "").replace("$", "")

    # Unescape JSON escapes
    s = s.replace("\\\\", "\\")

    # Strip \boxed wrapper
    m = re.match(r"^\\boxed\{(.+)\}$", s, re.DOTALL)
    if m:
        s = m.group(1)

    s = s.strip()
    if not s:
        return "N/A"

    # Check if this looks like math
    latex_cmds = ["\\frac", "\\sqrt", "\\int", "\\sum", "\\lim", "\\log",
                  "\\sin", "\\cos", "\\tan", "\\pi", "\\gamma", "\\Gamma",
                  "\\alpha", "\\beta", "\\infty", "\\cdot", "\\times",
                  "\\left", "\\right", "^{", "_{"]
    math_chars = ["=", "+", "/", "^", "_", "<", ">"]
    is_math = (
        any(cmd in s for cmd in latex_cmds)
        or (any(c in s for c in math_chars) and any(c.isdigit() or c.isalpha() for c in s))
        or re.match(r"^-?\d+\.?\d*$", s)
    )

    if is_math:
        s = s.replace("*", "\\cdot ")
        s = _balance_braces(s)
        return f"${s}$"

    # Plain text — escape special characters
    return _escape_tex(s)


def _short(name: str) -> str:
    n = name.split("/")[-1]
    for suf in ["-instruct", "-versatile", "-it"]:
        n = n.replace(suf, "")
    return n[:25]


def _conf_color(confidence: float) -> str:
    """Return a color name defined in our LaTeX preamble."""
    if confidence >= 0.8:
        return "success"
    elif confidence >= 0.5:
        return "warning"
    return "danger"


def _conf_label(confidence: float) -> str:
    if confidence >= 0.8:
        return "HIGH"
    elif confidence >= 0.5:
        return "MEDIUM"
    return "LOW"


def generate_pdf_report(result: dict) -> bytes:
    """Generate a LaTeX report and compile to PDF."""

    confidence = result.get("confidence", 0)
    answer = result.get("final_answer", "N/A")
    debate_rounds = result.get("debate_rounds", 0)
    sympy_override = result.get("symbolic_override", False)
    aligned = result.get("aligned_steps", [])
    agreement = result.get("answer_agreement", {})
    audit = result.get("audit_trail", [])
    raw_solutions = result.get("model_solutions", {})
    problem_text = result.get("problem_text", "N/A")

    agreed_count = sum(1 for s in aligned if not s.get("flagged", False))
    total_models = sum(len(m) for m in agreement.values())
    color = _conf_color(confidence)
    label = _conf_label(confidence)
    now = datetime.now().strftime("%B %d, %Y \\enspace %H:%M")

    # Build LaTeX document
    tex = r"""
\documentclass[11pt,a4paper]{article}

% ── Packages ──────────────────────────────────────
\usepackage[margin=2.2cm, top=2.5cm, bottom=2.5cm]{geometry}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage[dvipsnames,table]{xcolor}
\usepackage{booktabs}
\usepackage{fancyhdr}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{tabularx}
\usepackage{colortbl}
\usepackage{graphicx}
\usepackage{tikz}
\usepackage{tcolorbox}
\tcbuselibrary{skins}

% ── Colors ────────────────────────────────────────
\definecolor{brand}{HTML}{3B5BDB}
\definecolor{darkbg}{HTML}{1E2128}
\definecolor{lightbg}{HTML}{F5F6F8}
\definecolor{muted}{HTML}{8C8C8C}
\definecolor{success}{HTML}{16A34A}
\definecolor{warning}{HTML}{D97706}
\definecolor{danger}{HTML}{DC2626}
\definecolor{rowalt}{HTML}{F8F9FA}

% ── Header/Footer ────────────────────────────────
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0.4pt}
\fancyhead[L]{\small\color{muted} MVM\textsuperscript{2} Verification Report}
\fancyhead[R]{\small\color{muted} \thepage}
\fancyfoot[C]{\footnotesize\color{muted} Generated by MVM\textsuperscript{2} --- Multi-LLM Math Verification System}

% ── Hyperlinks ───────────────────────────────────
\hypersetup{colorlinks=true, linkcolor=brand, urlcolor=brand}

% ── Custom commands ──────────────────────────────
\newcommand{\sectionrule}{\noindent\textcolor{brand}{\rule{4cm}{1.5pt}}\par\vspace{6pt}}
\newcommand{\badge}[2]{\tikz[baseline=(X.base)]{\node[fill=#1, text=white, rounded corners=2pt, inner sep=3pt, font=\bfseries\scriptsize] (X) {#2};}}

\begin{document}

% ══════════════════════════════════════════════════
% TITLE
% ══════════════════════════════════════════════════
\thispagestyle{empty}

\begin{center}
\vspace*{1cm}

{\Huge\bfseries Verification Report}

\vspace{8pt}
{\large\color{muted} MVM\textsuperscript{2} Multi-Model Math Verification System}

\vspace{4pt}
{\color{muted} """ + now + r"""}

\vspace{8pt}
\textcolor{brand}{\rule{\textwidth}{2pt}}
\end{center}

\vspace{1cm}

% ══════════════════════════════════════════════════
% 1. PROBLEM STATEMENT
% ══════════════════════════════════════════════════
\section{Problem Statement}
\sectionrule

\begin{tcolorbox}[colback=lightbg, colframe=lightbg, boxrule=0pt, arc=4pt, left=12pt, right=12pt, top=10pt, bottom=10pt]
\large
""" + _math_to_latex(problem_text) + r"""
\end{tcolorbox}

\vspace{0.5cm}

% ══════════════════════════════════════════════════
% 2. EXECUTIVE SUMMARY
% ══════════════════════════════════════════════════
\section{Executive Summary}
\sectionrule

\begin{center}
\renewcommand{\arraystretch}{1.5}
\begin{tabular}{ccccc}
\toprule
\textbf{Confidence} & \textbf{Level} & \textbf{Models} & \textbf{Steps Agreed} & \textbf{Debates} \\
\midrule
{\color{""" + color + r"""}\textbf{""" + f"{confidence*100:.0f}\\%" + r"""}} & {\color{""" + color + r"""}\textbf{""" + label + r"""}} & """ + str(total_models) + r""" & """ + str(agreed_count) + "/" + str(len(aligned)) + r""" & """ + str(debate_rounds) + r""" \\
\bottomrule
\end{tabular}
\end{center}
"""

    # SymPy override notice
    if sympy_override:
        tex += r"""
\begin{tcolorbox}[colback=success!8, colframe=success, boxrule=0.8pt, arc=3pt, left=10pt, right=10pt, top=6pt, bottom=6pt]
\textcolor{success}{\textbf{SYMPY INDEPENDENTLY VERIFIED}} --- Answer confirmed by symbolic computation engine.
\end{tcolorbox}
"""

    # ══════════════════════════════════════════════════
    # 3. FINAL ANSWER
    # ══════════════════════════════════════════════════
    tex += r"""
\vspace{0.5cm}
\section{Final Verified Answer}
\sectionrule

\begin{tcolorbox}[colback=lightbg, colframe=lightbg!80!black, boxrule=0.5pt, arc=4pt, left=14pt, right=14pt, top=14pt, bottom=14pt]
\begin{center}
{\LARGE """ + _math_to_latex(answer) + r"""}
\end{center}
\end{tcolorbox}

\begin{center}
\fcolorbox{""" + color + r"""}{""" + color + r"""!10}{\textcolor{""" + color + r"""}{\textbf{""" + label + r""" CONFIDENCE --- """ + f"{confidence*100:.0f}\\%" + r"""}}}"""

    if sympy_override:
        tex += r"""\quad \fcolorbox{brand}{brand!10}{\textcolor{brand}{\textbf{SYMPY VERIFIED}}}"""

    tex += r"""
\end{center}

\vspace{0.5cm}
"""

    # ══════════════════════════════════════════════════
    # 4. MODEL AGREEMENT
    # ══════════════════════════════════════════════════
    if agreement:
        tex += r"""
\section{Model Agreement}
\sectionrule

""" + str(total_models) + r""" models solved the problem independently. """ + str(len(agreement)) + r""" distinct answer(s) were produced.

\vspace{6pt}
\begin{itemize}[leftmargin=1.5em, itemsep=8pt]
"""
        for ans_text, model_list in agreement.items():
            count = len(model_list)
            names = ", ".join(_short(m) for m in model_list)
            tex += r"\item \textbf{" + str(count) + " model" + ("s" if count > 1 else "") + r":} " + _escape_tex(names) + r" \\ " + "\n"
            tex += r"Answer: " + _math_to_latex(ans_text) + "\n"

        tex += r"""
\end{itemize}

\vspace{0.3cm}
"""

    # ══════════════════════════════════════════════════
    # 5. STEP-BY-STEP VERIFICATION
    # ══════════════════════════════════════════════════
    if aligned:
        tex += r"""
\newpage
\section{Step-by-Step Verification}
\sectionrule

""" + str(len(aligned)) + r""" canonical steps identified. """ + str(agreed_count) + r""" agreed, """ + str(len(aligned) - agreed_count) + r""" disputed.

\vspace{8pt}
"""
        for step in aligned:
            snum = step["canonical_step_number"]
            ratio = step.get("agreement_ratio", 0)
            flagged = step.get("flagged", False)
            symbolic = step.get("symbolic_verified")
            desc = step.get("description", "")[:70]

            if flagged:
                tag, tag_color = "DISPUTED", "danger"
            elif symbolic is True:
                tag, tag_color = "VERIFIED", "success"
            elif ratio >= 0.8:
                tag, tag_color = "AGREED", "success"
            else:
                tag, tag_color = "PARTIAL", "warning"

            tex += r"""
\noindent\fcolorbox{""" + tag_color + r"""}{""" + tag_color + r"""!15}{\textcolor{""" + tag_color + r"""}{\scriptsize\bfseries """ + tag + r"""}}\enspace\textbf{Step """ + str(snum) + r""":} """ + _escape_tex(desc) + r"""
\hfill {\small\color{muted} Agreement: """ + f"{ratio:.0%}" + r"""}

\vspace{2pt}
"""

            model_steps = step.get("model_steps", {})
            if model_steps:
                tex += r"""
\begin{small}
\renewcommand{\arraystretch}{1.3}
\begin{tabularx}{\textwidth}{l X X}
\toprule
\textbf{Model} & \textbf{Expression} & \textbf{Result} \\
\midrule
"""
                row_idx = 0
                for mname, mstep in model_steps.items():
                    short = _escape_tex(_short(mname))
                    if row_idx % 2 == 1:
                        tex += r"\rowcolor{rowalt}" + "\n"

                    if mstep:
                        math_val = _math_to_latex(mstep.get("mathematical_expression", ""))
                        result_val = _math_to_latex(mstep.get("result", ""))
                    else:
                        math_val = r"\textcolor{muted}{---}"
                        result_val = r"\textcolor{muted}{---}"

                    tex += short + r" & " + math_val + r" & " + result_val + r" \\" + "\n"
                    row_idx += 1

                tex += r"""
\bottomrule
\end{tabularx}
\end{small}
"""
                if symbolic is True:
                    tex += r"\textcolor{success}{\small\itshape Symbolically verified by SymPy}" + "\n"
                elif symbolic is False:
                    tex += r"\textcolor{danger}{\small\itshape Symbolic verification FAILED}" + "\n"

            tex += r"\vspace{10pt}" + "\n"

    # ══════════════════════════════════════════════════
    # 6. MODEL SUMMARY
    # ══════════════════════════════════════════════════
    if raw_solutions:
        tex += r"""
\section{Model Solution Summary}
\sectionrule

\begin{center}
\renewcommand{\arraystretch}{1.3}
\begin{tabular}{llp{8cm}}
\toprule
\textbf{Model} & \textbf{Steps} & \textbf{Final Answer} \\
\midrule
"""
        for mname, steps in raw_solutions.items():
            short = _escape_tex(_short(mname))
            model_ans = ""
            for ans_text, model_list in agreement.items():
                if mname in model_list:
                    model_ans = _math_to_latex(ans_text)
                    break
            tex += short + r" & " + str(len(steps)) + r" & " + (model_ans or r"\textcolor{muted}{---}") + r" \\" + "\n"

        tex += r"""
\bottomrule
\end{tabular}
\end{center}

\vspace{0.5cm}
"""

    # ══════════════════════════════════════════════════
    # 7. AUDIT TRAIL
    # ══════════════════════════════════════════════════
    if audit:
        tex += r"""
\section{Verification Audit Trail}
\sectionrule

{\small Complete log of pipeline decisions:}

\vspace{4pt}
\begin{tcolorbox}[colback=lightbg, colframe=lightbg!70!black, boxrule=0.3pt, arc=2pt, left=8pt, right=8pt, top=6pt, bottom=6pt]
\ttfamily\scriptsize
"""
        for entry in audit[:40]:
            clean = _escape_tex(entry[:120])
            if "VERIFIED" in entry or "confirmed" in entry:
                tex += r"\textcolor{success}{" + clean + r"}\par" + "\n"
            elif "OVERRIDE" in entry or "WRONG" in entry or "FAILED" in entry:
                tex += r"\textcolor{danger}{" + clean + r"}\par" + "\n"
            elif "SymPy" in entry:
                tex += r"\textcolor{brand}{" + clean + r"}\par" + "\n"
            else:
                tex += clean + r"\par" + "\n"

        tex += r"""
\end{tcolorbox}
"""

    # ══════════════════════════════════════════════════
    # 8. METHODOLOGY
    # ══════════════════════════════════════════════════
    tex += r"""
\vspace{1cm}
\noindent\textcolor{muted}{\rule{\textwidth}{0.4pt}}
\vspace{4pt}

\noindent\textbf{Methodology}

\vspace{4pt}
\noindent{\small\color{muted}
This report was generated by MVM\textsuperscript{2}, a multi-model math verification system.
The problem was solved independently by """ + str(total_models) + r""" large language models across multiple providers
(Groq, NVIDIA NIM). Solutions were parsed into structured steps, aligned across models,
and verified through three mechanisms: (1)~step-level consensus voting,
(2)~symbolic verification via SymPy, and (3)~multi-agent debate for disputed steps.
When possible, SymPy independently computes the answer and cross-checks all model outputs,
providing a ground-truth verification layer.
}

\end{document}
"""

    # ── 3-tier compile chain: NEVER crashes ──
    # Tier 1: Full LaTeX with math
    pdf = _try_compile_latex(tex)
    if pdf:
        return pdf

    # Tier 2: Strip all math, use plain text
    logger.warning("pdflatex_tier1_failed_stripping_math")
    tex_plain = _strip_math_for_fallback(tex)
    pdf = _try_compile_latex(tex_plain)
    if pdf:
        return pdf

    # Tier 3: Minimal safe LaTeX (absolutely cannot fail)
    logger.warning("pdflatex_tier2_failed_using_minimal")
    return _generate_minimal_pdf(result)


def _strip_math_for_fallback(tex: str) -> str:
    """Strip all inline math $...$ and replace with plain text."""
    import re as _re

    def _replace_math(match):
        inner = match.group(1)
        inner = inner.replace("\\frac", "").replace("\\sqrt", "sqrt")
        inner = inner.replace("\\cdot", "*").replace("\\times", "x")
        inner = inner.replace("\\left", "").replace("\\right", "")
        inner = inner.replace("\\", "").replace("{", "(").replace("}", ")")
        inner = inner.replace("^", "**")
        return inner

    result = _re.sub(r'\$([^$]+)\$', _replace_math, tex)
    # Also remove any tcolorbox breakable (common failure point)
    result = result.replace("breakable", "")
    return result


def _try_compile_latex(tex_source: str) -> bytes | None:
    """Try to compile LaTeX. Returns PDF bytes or None on failure."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = os.path.join(tmpdir, "report.tex")
            pdf_path = os.path.join(tmpdir, "report.pdf")

            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(tex_source)

            # Run pdflatex twice (for cross-references)
            for run in range(2):
                proc = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
                    cwd=tmpdir,
                    capture_output=True,
                    timeout=30,
                )

            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    return f.read()

            stdout = proc.stdout.decode("utf-8", errors="replace")
            logger.error("pdflatex_failed", log_tail=stdout[-500:])
            return None
    except Exception as e:
        logger.error("pdflatex_exception", error=str(e))
        return None


def _generate_minimal_pdf(result: dict) -> bytes:
    """Tier 3 fallback: generate a simple but guaranteed PDF using reportlab.
    This CANNOT fail — no pdflatex, no external deps, pure Python."""
    import io as _io
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER

    buf = _io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=20*mm, bottomMargin=20*mm,
                            leftMargin=20*mm, rightMargin=20*mm)
    styles = getSampleStyleSheet()
    title_s = ParagraphStyle("T", parent=styles["Title"], fontSize=18, spaceAfter=8)
    sub_s = ParagraphStyle("S", parent=styles["Normal"], fontSize=10, textColor=colors.grey)
    h2_s = ParagraphStyle("H", parent=styles["Heading2"], fontSize=13, spaceBefore=12, spaceAfter=6)
    body_s = ParagraphStyle("B", parent=styles["Normal"], fontSize=10, leading=14)
    mono_s = ParagraphStyle("M", parent=styles["Code"], fontSize=8, leading=10)

    def _safe(text):
        """Make text safe for reportlab — ASCII only."""
        if not text:
            return "N/A"
        s = str(text)
        # Strip LaTeX
        s = re.sub(r'\$([^$]*)\$', r'\1', s)
        s = re.sub(r'\\[a-zA-Z]+', '', s)
        s = s.replace('{', '(').replace('}', ')').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        # ASCII only
        return ''.join(c if ord(c) < 128 else '?' for c in s)

    elements = []

    confidence = result.get("confidence", 0)
    answer = result.get("final_answer", "N/A")
    debate_rounds = result.get("debate_rounds", 0)
    aligned = result.get("aligned_steps", [])
    agreement = result.get("answer_agreement", {})
    audit = result.get("audit_trail", [])
    problem_text = result.get("problem_text", "N/A")
    agreed = sum(1 for s in aligned if not s.get("flagged", False))

    # Title
    elements.append(Paragraph("MVM2 Verification Report", title_s))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", sub_s))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    elements.append(Spacer(1, 12))

    # Problem
    elements.append(Paragraph("Problem", h2_s))
    elements.append(Paragraph(_safe(problem_text)[:500], body_s))
    elements.append(Spacer(1, 8))

    # Summary
    elements.append(Paragraph("Summary", h2_s))
    conf_label = "HIGH" if confidence >= 0.8 else "MEDIUM" if confidence >= 0.5 else "LOW"
    data = [
        ["Confidence", "Steps", "Debates", "Level"],
        [f"{confidence:.0%}", f"{agreed}/{len(aligned)}", str(debate_rounds), conf_label],
    ]
    t = Table(data, colWidths=[100, 100, 100, 100])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.9, 0.9, 0.9)),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 12))

    # Answer
    elements.append(Paragraph("Final Answer", h2_s))
    elements.append(Paragraph(f"<b>{_safe(answer)}</b>", ParagraphStyle(
        "A", parent=body_s, fontSize=14, backColor=colors.Color(0.95, 0.95, 0.95),
        borderPadding=10, borderWidth=1, borderColor=colors.grey,
    )))
    elements.append(Spacer(1, 8))

    # Agreement
    if agreement:
        elements.append(Paragraph("Model Agreement", h2_s))
        for ans_text, models in agreement.items():
            names = ", ".join(m.split("/")[-1][:20] for m in models)
            elements.append(Paragraph(f"<b>{len(models)} model(s):</b> {_safe(names)}", body_s))
            elements.append(Paragraph(f"Answer: {_safe(ans_text)[:100]}", mono_s))
            elements.append(Spacer(1, 4))

    # Steps
    if aligned:
        elements.append(Paragraph("Steps", h2_s))
        for step in aligned[:15]:  # Cap at 15 steps
            snum = step["canonical_step_number"]
            ratio = step.get("agreement_ratio", 0)
            flagged = step.get("flagged", False)
            desc = _safe(step.get("description", ""))[:60]
            tag = "DISPUTED" if flagged else "AGREED" if ratio >= 0.8 else "PARTIAL"
            elements.append(Paragraph(f"<b>Step {snum}:</b> {desc} ({ratio:.0%}) [{tag}]", body_s))

    # Audit
    if audit:
        elements.append(Spacer(1, 8))
        elements.append(Paragraph("Audit Trail", h2_s))
        for entry in audit[:25]:
            elements.append(Paragraph(_safe(entry)[:120], mono_s))

    # Footer
    elements.append(Spacer(1, 16))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    elements.append(Paragraph("Generated by MVM2 (fallback mode)", sub_s))

    doc.build(elements)
    return buf.getvalue()
