"""MVM2 -- Multi-LLM Math Verification System"""

import os
import json
import re
import time
import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000/api")

st.set_page_config(page_title="MVM2 Math Verifier", page_icon="M", layout="wide")

# ── CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    .block-container { max-width: 1200px; }

    .model-grid { display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0; }
    .m-card {
        flex: 1; min-width: 130px; padding: 10px 14px;
        border-radius: 8px; border: 1.5px solid #333;
        font-size: 13px; font-family: 'Inter', sans-serif;
    }
    .m-waiting { border-color: #444; background: #111; color: #666; }
    .m-running { border-color: #d97706; background: #1c1a0a; color: #fbbf24; }
    .m-ok { border-color: #16a34a; background: #0a1c0e; color: #4ade80; }
    .m-error { border-color: #dc2626; background: #1c0a0a; color: #f87171; }
    .m-name { font-weight: 600; font-size: 12px; }
    .m-meta { font-size: 11px; opacity: 0.7; margin-top: 2px; }

    .pipeline-bar { display: flex; gap: 4px; margin: 8px 0 16px 0; flex-wrap: wrap; }
    .p-stage { padding: 5px 14px; border-radius: 16px; font-size: 12px; font-weight: 600; }
    .p-active { background: #d97706; color: #000; }
    .p-done { background: #16a34a; color: #fff; }
    .p-pending { background: #222; color: #555; }

    .activity-bar {
        display: flex; align-items: center; gap: 10px;
        padding: 10px 16px; border-radius: 8px;
        background: #111; border: 1px solid #222;
        font-size: 13px; color: #ccc; margin: 8px 0;
    }
    .spinner {
        width: 16px; height: 16px; border: 2px solid #333;
        border-top-color: #d97706; border-radius: 50%;
        animation: spin 0.8s linear infinite; flex-shrink: 0;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    .live-log {
        font-family: 'SF Mono', 'Fira Code', monospace;
        font-size: 11.5px; line-height: 1.6;
        background: #0a0a0a; border: 1px solid #1a1a1a;
        border-radius: 8px; padding: 14px;
        max-height: 260px; overflow-y: auto; color: #999;
    }
    .ll { margin: 1px 0; }
    .ll-solve { color: #60a5fa; }
    .ll-consensus { color: #fb923c; }
    .ll-symbolic { color: #a78bfa; }
    .ll-resolve { color: #f87171; }
    .ll-done { color: #4ade80; }

    .stat-row { display: flex; gap: 12px; margin: 12px 0; }
    .stat-box {
        flex: 1; padding: 12px 16px; border-radius: 8px;
        background: #111; border: 1px solid #1a1a1a; text-align: center;
    }
    .stat-val { font-size: 22px; font-weight: 700; color: #fff; }
    .stat-label { font-size: 11px; color: #666; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.5px; }

    .conf-high { color: #4ade80; }
    .conf-med { color: #fbbf24; }
    .conf-low { color: #f87171; }
    .badge { display: inline-block; padding: 3px 10px; border-radius: 4px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; }
    .badge-high { background: #16a34a22; color: #4ade80; border: 1px solid #16a34a44; }
    .badge-med { background: #d9770622; color: #fbbf24; border: 1px solid #d9770644; }
    .badge-low { background: #dc262622; color: #f87171; border: 1px solid #dc262644; }

    .answer-box {
        padding: 16px 20px; border-radius: 8px;
        background: #111; border: 1px solid #222; margin: 12px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── LaTeX helpers ───────────────────────────────────────────
def clean_latex(text: str) -> str:
    """Clean LaTeX string for st.latex() rendering.
    Strips surrounding $ signs and common artifacts."""
    if not text or text.upper() == "N/A" or text == "None":
        return ""
    s = text.strip()
    # Remove wrapping $ or $$
    if s.startswith("$$") and s.endswith("$$"):
        s = s[2:-2].strip()
    elif s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    # Remove \boxed wrapper
    m = re.match(r"^\\boxed\{(.+)\}$", s)
    if m:
        s = m.group(1)
    return s


def has_latex(text: str) -> bool:
    """Check if text contains LaTeX commands."""
    if not text:
        return False
    indicators = ["\\frac", "\\sqrt", "\\int", "\\sum", "\\lim", "\\log",
                  "\\gamma", "\\Gamma", "\\pi", "\\boxed", "^{", "_{",
                  "\\cdot", "\\times", "\\div", "\\infty", "\\alpha", "\\beta"]
    return any(i in text for i in indicators)


def render_math(text: str):
    """Render math expression: use st.latex if it contains LaTeX, otherwise st.code."""
    if not text or text.upper() in ("N/A", "NONE", ""):
        st.text("N/A")
        return
    cleaned = clean_latex(text)
    if has_latex(cleaned) or has_latex(text):
        try:
            st.latex(cleaned)
        except Exception:
            st.code(text, language=None)
    else:
        st.code(text, language=None)


def short_model(name: str) -> str:
    parts = name.split("/")
    n = parts[-1]
    for suf in ["-instruct", "-versatile", "-it"]:
        n = n.replace(suf, "")
    return n[:22]


# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### System")
    if st.button("Check Health", use_container_width=True):
        try:
            resp = httpx.get(f"{API_URL}/health", timeout=30)
            data = resp.json()
            for model, ok in data.get("providers", {}).items():
                color = "#4ade80" if ok else "#f87171"
                label = "online" if ok else "offline"
                st.markdown(f'<span style="color:{color}">&#9679;</span> {short_model(model)} -- {label}', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"API unreachable: {e}")

    st.divider()
    st.markdown("### Pipeline")
    st.markdown("""
1. **Parallel Solve** -- 5 LLMs simultaneously
2. **Parse** -- Extract structured steps
3. **Consensus** -- Majority vote per step
4. **Symbolic** -- SymPy verification
5. **Debate** -- Re-query + multi-agent debate
6. **Report** -- Verified answer
    """)

# ── Header ──────────────────────────────────────────────────
st.markdown("## MVM2 -- Math Verification System")
st.caption("Multi-model consensus verification with symbolic checking")

# ── Input ───────────────────────────────────────────────────
col_in, col_out = st.columns([1, 2])

with col_in:
    st.markdown("#### Problem Input")
    problem_text = st.text_area(
        "Enter a math problem:",
        height=140,
        placeholder="Solve for x: 3(x + 2) = 15\nFind d/dx of x^3 sin(x)\nIntegral of 1/(1+x^2) dx",
        label_visibility="collapsed",
    )
    uploaded_file = st.file_uploader("Or upload an image:", type=["png", "jpg", "jpeg", "webp"])
    solve_btn = st.button("Verify", type="primary", use_container_width=True)


# ── Render helpers ──────────────────────────────────────────
def render_models(ph, states):
    cards = ""
    for model, s in states.items():
        cls = f"m-{s['status']}"
        lat = f"{s['latency_ms']:.0f}ms" if s.get("latency_ms") else "..."
        sym = {"waiting": "&#9675;", "running": "&#9696;", "ok": "&#10003;", "error": "&#10007;"}.get(s["status"], "")
        cards += f'<div class="m-card {cls}"><div class="m-name">{sym} {short_model(model)}</div><div class="m-meta">{lat}</div></div>'
    ph.markdown(f'<div class="model-grid">{cards}</div>', unsafe_allow_html=True)


def render_stages(ph, stages):
    html = ""
    for name, status in stages.items():
        html += f'<span class="p-stage p-{status}">{name}</span>'
    ph.markdown(f'<div class="pipeline-bar">{html}</div>', unsafe_allow_html=True)


def render_activity(ph, text, spinning=True):
    icon = '<div class="spinner"></div>' if spinning else '<span style="color:#4ade80">&#10003;</span>'
    ph.markdown(f'<div class="activity-bar">{icon}<span>{text}</span></div>', unsafe_allow_html=True)


# ── Main solve ──────────────────────────────────────────────
if solve_btn and (problem_text or uploaded_file):
    with col_out:
        st.markdown("#### Verification")
        stage_ph = st.empty()
        activity_ph = st.empty()
        models_ph = st.empty()
        log_ph = st.empty()
        report_ct = st.container()

        stages = {"Solve": "active", "Parse": "pending", "Consensus": "pending",
                  "Symbolic": "pending", "Debate": "pending", "Report": "pending"}
        model_states = {}
        logs = []
        t0 = time.time()

        def log(text, cls=""):
            e = time.time() - t0
            logs.append(f'<div class="ll {cls}">[{e:5.1f}s] {text}</div>')

        def flush():
            log_ph.markdown(f'<div class="live-log">{"".join(logs[-25:])}</div>', unsafe_allow_html=True)

        payload = {"text": problem_text or ""}
        if uploaded_file:
            import base64
            payload["image_base64"] = base64.b64encode(uploaded_file.read()).decode()
            if not problem_text:
                payload["text"] = ""

        render_stages(stage_ph, stages)
        render_activity(activity_ph, "Initializing pipeline...")
        final_result = None

        try:
            with httpx.stream("POST", f"{API_URL}/solve/stream", json=payload, timeout=600.0) as resp:
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    ev = json.loads(line[6:])
                    evt = ev.get("event", "")
                    data = ev.get("data", {})

                    if evt == "pipeline_start":
                        ms = data.get("models", [])
                        model_states = {m: {"status": "waiting"} for m in ms}
                        render_models(models_ph, model_states)
                        render_activity(activity_ph, f"Sending problem to {len(ms)} models...")
                        log(f"Pipeline started with {len(ms)} models", "ll-solve")

                    elif evt == "model_start":
                        m = data.get("model", "")
                        if m in model_states:
                            model_states[m]["status"] = "running"
                        render_models(models_ph, model_states)

                    elif evt == "model_done":
                        m = data.get("model", "")
                        s = data.get("status", "ok")
                        lat = data.get("latency_ms", 0)
                        done = data.get("completed", 0)
                        total = data.get("total", 0)
                        if m in model_states:
                            model_states[m]["status"] = s
                            if lat:
                                model_states[m]["latency_ms"] = lat
                        render_models(models_ph, model_states)
                        render_activity(activity_ph, f"Solving: {done}/{total} models responded")
                        mark = "+" if s == "ok" else "x"
                        log(f"[{mark}] {short_model(m)} -- {lat:.0f}ms ({done}/{total})", "ll-solve")

                    elif evt == "parse_done":
                        stages["Solve"] = "done"; stages["Parse"] = "active"
                        render_stages(stage_ph, stages)
                        p_ok = data.get('parsed_count', 0)
                        p_fail = data.get('failed_count', 0)
                        if p_fail > 0:
                            render_activity(activity_ph, f"Parsed {p_ok}, re-parsing {p_fail} failed...")
                        else:
                            render_activity(activity_ph, f"Parsed {p_ok} solutions")
                        log(f"Parsed {p_ok} solutions, {p_fail} failed", "ll-consensus")

                    elif evt == "reparse_start":
                        render_activity(activity_ph, f"Re-parsing {data.get('model','?')} output...")
                        log(f"Re-parsing {short_model(data.get('model',''))}", "ll-consensus")

                    elif evt == "reparse_done":
                        s = data.get("status", "")
                        m = short_model(data.get("model", ""))
                        steps = data.get("steps", 0)
                        if s == "ok":
                            log(f"Re-parse OK: {m} -> {steps} steps recovered", "ll-done")
                        else:
                            log(f"Re-parse failed: {m}", "ll-resolve")

                    elif evt == "parse_complete":
                        stages["Parse"] = "done"; stages["Consensus"] = "active"
                        render_stages(stage_ph, stages)
                        render_activity(activity_ph, "Computing consensus...")
                        log(f"Total parsed: {data.get('parsed_count',0)}/{data.get('total',0)}", "ll-consensus")

                    elif evt == "answer_agreement":
                        ag = data.get("agreement", {})
                        ans = data.get("consensus_answer", "?")
                        c = data.get("confidence", 0)
                        log(f"Consensus: {ans[:50]} ({c:.0%})", "ll-consensus")

                    elif evt == "steps_aligned":
                        log(f"Aligned {data.get('step_count',0)} steps", "ll-consensus")

                    elif evt == "step_consensus":
                        fl = data.get("flagged_count", 0)
                        ts = data.get("total_steps", 0)
                        stages["Consensus"] = "done"
                        stages["Symbolic"] = "active" if fl > 0 else "done"
                        if fl == 0:
                            stages["Debate"] = "done"
                        render_stages(stage_ph, stages)
                        render_activity(activity_ph, f"{ts - fl}/{ts} steps agreed" if fl else "All steps agreed")
                        log(f"Consensus: {fl}/{ts} flagged", "ll-consensus")

                    elif evt == "symbolic_start":
                        stages["Symbolic"] = "active"
                        render_stages(stage_ph, stages)
                        render_activity(activity_ph, "Running SymPy verification...")
                        log("Symbolic verification started", "ll-symbolic")

                    elif evt == "step_verified":
                        sn = data.get("step_num", "?")
                        r = data.get("result")
                        label = {True: "verified", False: "failed", None: "inconclusive"}.get(r, "inconclusive")
                        log(f"  Step {sn}: {label}", "ll-symbolic")

                    elif evt == "symbolic_done":
                        stages["Symbolic"] = "done"
                        render_stages(stage_ph, stages)
                        log(f"Symbolic: {data.get('verified',0)}v {data.get('failed',0)}f {data.get('inconclusive',0)}i", "ll-symbolic")

                    elif evt == "resolution_start":
                        stages["Debate"] = "active"
                        render_stages(stage_ph, stages)
                        render_activity(activity_ph, f"Resolving {data.get('flagged_count',0)} disagreements...")
                        log(f"Resolution: {data.get('flagged_count',0)} steps", "ll-resolve")

                    elif evt == "requery_sent":
                        render_activity(activity_ph, f"Re-querying {short_model(data.get('model',''))} on step {data.get('step_num','?')}...")
                        log(f"  Re-query: {short_model(data.get('model',''))} step {data.get('step_num','?')}", "ll-resolve")

                    elif evt == "debate_round":
                        render_activity(activity_ph, f"Debate round {data.get('round','?')}, step {data.get('step_num','?')}...")
                        log(f"  Debate R{data.get('round','?')} step {data.get('step_num','?')}", "ll-resolve")

                    elif evt == "resolution_done":
                        stages["Debate"] = "done"
                        render_stages(stage_ph, stages)
                        log(f"Done: {data.get('remaining_flagged',0)} unresolved, {data.get('debate_rounds',0)} rounds", "ll-resolve")

                    elif evt == "verify_start":
                        render_activity(activity_ph, "SymPy independent verification...")
                        log("Independent answer verification (SymPy)", "ll-symbolic")

                    elif evt == "verify_done":
                        computed = data.get("sympy_computed", False)
                        verified = data.get("verified_answer")
                        override = data.get("override", False)
                        method = data.get("method", "")
                        if computed and verified:
                            if override:
                                log(f"SymPy OVERRIDE: answer = {str(verified)[:50]} (method: {method})", "ll-done")
                            else:
                                log(f"SymPy VERIFIED: {str(verified)[:50]}", "ll-done")
                        elif computed:
                            log("SymPy computed but no model matched", "ll-resolve")
                        else:
                            log("SymPy cannot compute this problem type", "ll-symbolic")

                    elif evt == "result":
                        stages["Report"] = "done"
                        render_stages(stage_ph, stages)
                        render_activity(activity_ph, "Verification complete", spinning=False)
                        log("Complete", "ll-done")

                    elif evt == "final_result":
                        final_result = data

                    elif evt in ("pipeline_error", "error"):
                        render_activity(activity_ph, f"Error: {data.get('error', data.get('message',''))}", spinning=False)

                    flush()

        except httpx.TimeoutException:
            st.error("Request timed out.")
        except httpx.ConnectError:
            st.error("Cannot connect to API.")
        except Exception as e:
            st.error(f"Error: {e}")

        # ── REPORT ──────────────────────────────────────────
        if final_result:
            elapsed = time.time() - t0
            confidence = final_result.get("confidence", 0)
            answer = final_result.get("final_answer", "N/A")
            debate_rounds = final_result.get("debate_rounds", 0)
            agreement = final_result.get("answer_agreement", {})
            aligned = final_result.get("aligned_steps", [])
            audit = final_result.get("audit_trail", [])
            raw_solutions = final_result.get("model_solutions", {})
            sympy_override = final_result.get("symbolic_override", False)

            if confidence >= 0.8:
                conf_cls, badge_cls, conf_text = "conf-high", "badge-high", "HIGH"
            elif confidence >= 0.5:
                conf_cls, badge_cls, conf_text = "conf-med", "badge-med", "MEDIUM"
            else:
                conf_cls, badge_cls, conf_text = "conf-low", "badge-low", "LOW"

            with report_ct:
                st.markdown("---")
                st.markdown("#### Verification Report")

                # Stats
                ok_count = sum(1 for v in model_states.values() if v.get("status") == "ok")
                agreed = sum(1 for s in aligned if not s.get("flagged", False))
                st.markdown(f"""
                <div class="stat-row">
                    <div class="stat-box"><div class="stat-val {conf_cls}">{confidence:.0%}</div><div class="stat-label">Confidence</div></div>
                    <div class="stat-box"><div class="stat-val">{ok_count}/5</div><div class="stat-label">Models OK</div></div>
                    <div class="stat-box"><div class="stat-val">{agreed}/{len(aligned)}</div><div class="stat-label">Steps Agreed</div></div>
                    <div class="stat-box"><div class="stat-val">{debate_rounds}</div><div class="stat-label">Debates</div></div>
                    <div class="stat-box"><div class="stat-val">{elapsed:.1f}s</div><div class="stat-label">Time</div></div>
                </div>
                """, unsafe_allow_html=True)

                # Answer
                verify_badge = ""
                if sympy_override:
                    verify_badge = '<span class="badge badge-high" style="margin-left:8px;">SYMPY VERIFIED</span>'

                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; align-items:center; margin-top:16px;">
                    <span style="color:#888; font-size:13px; font-weight:600; text-transform:uppercase;">Final Verified Answer</span>
                    <div><span class="badge {badge_cls}">{conf_text} CONFIDENCE</span>{verify_badge}</div>
                </div>
                """, unsafe_allow_html=True)

                cleaned_answer = clean_latex(answer)
                if has_latex(answer) or has_latex(cleaned_answer):
                    try:
                        st.latex(cleaned_answer)
                    except Exception:
                        st.code(answer, language=None)
                else:
                    st.markdown(f"### `{answer}`")

                # Model agreement
                if agreement and len(agreement) > 1:
                    st.markdown("##### Model Agreement")
                    for ans_text, model_list in agreement.items():
                        names = ", ".join(short_model(m) for m in model_list)
                        count = len(model_list)
                        st.markdown(f"**{count} model{'s' if count > 1 else ''}:** {names}")
                        cleaned = clean_latex(ans_text)
                        if has_latex(ans_text) or has_latex(cleaned):
                            try:
                                st.latex(cleaned)
                            except Exception:
                                st.code(ans_text, language=None)
                        else:
                            st.code(ans_text, language=None)

                # Steps
                if aligned:
                    st.markdown("##### Step-by-Step Verification")
                    for step in aligned:
                        snum = step["canonical_step_number"]
                        ratio = step.get("agreement_ratio", 0)
                        flagged = step.get("flagged", False)
                        symbolic = step.get("symbolic_verified")
                        desc = step.get("description", "")[:70]

                        if flagged:
                            tag = "DISPUTED"
                        elif symbolic is True:
                            tag = "VERIFIED"
                        elif ratio >= 0.8:
                            tag = "AGREED"
                        else:
                            tag = "PARTIAL"

                        with st.expander(f"Step {snum}: {desc} -- {ratio:.0%} [{tag}]"):
                            for mname, mstep in step.get("model_steps", {}).items():
                                sn = short_model(mname)
                                if mstep:
                                    st.markdown(f"**{sn}**")
                                    math_expr = mstep.get("mathematical_expression", "")
                                    result_val = mstep.get("result", "")

                                    if math_expr and math_expr != "N/A":
                                        st.caption("Math:")
                                        render_math(math_expr)

                                    if result_val and result_val != "N/A":
                                        st.caption("Result:")
                                        render_math(result_val)

                                    st.markdown("---")
                                else:
                                    st.markdown(f"**{sn}** -- _no step_")

                            if symbolic is True:
                                st.success("Symbolically verified")
                            elif symbolic is False:
                                st.error("Symbolic verification failed")

                # Audit
                if audit:
                    with st.expander("Audit Trail"):
                        st.code("\n".join(audit), language=None)

                # Raw
                if raw_solutions:
                    with st.expander("Raw Model Solutions"):
                        for mname, steps in raw_solutions.items():
                            st.markdown(f"**{short_model(mname)}** ({len(steps)} steps)")

                # PDF Download
                st.markdown("---")
                try:
                    # Ensure data is JSON-serializable (strip any non-serializable types)
                    import json as _json
                    clean_data = _json.loads(_json.dumps(final_result, default=str))
                    pdf_resp = httpx.post(
                        f"{API_URL}/report/pdf",
                        json=clean_data,
                        timeout=60.0,
                    )
                    if pdf_resp.status_code == 200:
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_resp.content,
                            file_name="mvm2_verification_report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    else:
                        st.caption(f"PDF error: {pdf_resp.status_code} - {pdf_resp.text[:200]}")
                except Exception as e:
                    st.caption(f"PDF error: {e}")

elif solve_btn:
    with col_out:
        st.warning("Enter a math problem or upload an image.")
