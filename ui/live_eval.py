"""AIYYARY — Streamlit browser UI for running a live evaluation session with
a real business owner.

Calls the same Agent 1 / Agent 2 / Agent 3 classes main.py uses, in-process
(no subprocess) — the pipeline runs inside this Streamlit process.

Run with:
    streamlit run ui/live_eval.py
"""

import glob
import json
import os
import sys
from datetime import datetime

import pandas as pd
import streamlit as st

# `streamlit run ui/live_eval.py` puts ui/ (not the project root) on
# sys.path[0], so `agents.*` won't import without this.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

from agents.agent1_capability import PILLARS, CapabilityAssessmentAgent, readiness_tier
from agents.agent2_opportunity import OpportunityResearchAgent
from agents.agent3_risk import RiskTransformationAgent
from agents.enrichment import EnrichmentAgent

st.set_page_config(page_title="AIYYARY — Live Evaluation", page_icon="🧵", layout="wide")

TEAL = "#0d9488"

PILLAR_DISPLAY_NAMES = {
    "data_foundations": "Data Foundations",
    "process_digitisation": "Process Digitisation",
    "technology_infrastructure": "Technology Infrastructure",
    "staff_digital_literacy": "Staff Digital Literacy",
    "governance_compliance": "Governance & Compliance",
    "ai_tool_experience": "AI Tool Experience",
}

PILLAR_COLORS = {
    "data_foundations": "#0d9488",
    "process_digitisation": "#2563eb",
    "technology_infrastructure": "#7c3aed",
    "staff_digital_literacy": "#db2777",
    "governance_compliance": "#d97706",
    "ai_tool_experience": "#16a34a",
}

TIER_COLORS = {"Early-Stage": "#dc2626", "Mid-Readiness": "#d97706", "High-Readiness": "#16a34a"}
SEVERITY_COLORS = {"HIGH": "#fecaca", "MEDIUM": "#fed7aa", "LOW": "#bbf7d0"}

CHALLENGE_OPTIONS = [
    "Managing rapid growth",
    "Reducing returns",
    "Improving online presence",
    "Reaching new customers",
    "Improving sustainability credibility",
    "Other — specify below",
]
STAFF_OPTIONS = ["1-5", "6-10", "11-25", "26-50", "50+"]
CHANNEL_OPTIONS = ["Shopify", "WooCommerce", "Wix", "Other"]

# The 24-question owner-interview instrument, 4 questions per pillar, each
# with anchor labels at 1/5/9. Order matches PILLARS in agents/agent1_capability.py.
QUESTIONS = {
    "data_foundations": [
        {
            "prompt": "How structured and queryable is your customer purchase history?",
            "anchors": ("paper/spreadsheet", "basic CRM", "fully structured database"),
        },
        {
            "prompt": "How complete and accurate is your product inventory data?",
            "anchors": ("manual count", "basic system", "automated real-time"),
        },
        {
            "prompt": "How well do you track customer behaviour and browsing patterns?",
            "anchors": ("no tracking", "basic Analytics", "full behavioural analytics"),
        },
        {
            "prompt": "How consistently do you collect and store returns reason data?",
            "anchors": ("not at all", "informal notes", "structured categorised database"),
        },
    ],
    "process_digitisation": [
        {
            "prompt": "How digitised is your returns and exchanges process?",
            "anchors": ("paper forms", "basic online form", "fully automated with reason capture"),
        },
        {
            "prompt": "How automated is your order fulfilment and inventory replenishment?",
            "anchors": ("fully manual", "partially automated", "fully automated with triggers"),
        },
        {
            "prompt": "How structured is your customer feedback collection process?",
            "anchors": ("no process", "occasional surveys", "systematic multi-channel collection"),
        },
        {
            "prompt": "How digitised are your supplier and supply chain communications?",
            "anchors": ("phone/email only", "shared spreadsheets", "integrated digital platform"),
        },
    ],
    "technology_infrastructure": [
        {
            "prompt": "How reliable and scalable is your e-commerce platform?",
            "anchors": ("basic template site", "standard Shopify/WooCommerce", "enterprise platform with custom dev"),
        },
        {
            "prompt": "How well integrated are your sales, inventory, and customer systems?",
            "anchors": ("completely separate", "some manual data transfer", "fully integrated via APIs"),
        },
        {
            "prompt": "How secure and compliant is your technology stack?",
            "anchors": ("no SSL/basic security", "standard SSL and backups", "full security audit and compliance"),
        },
        {
            "prompt": "How capable is your analytics and reporting infrastructure?",
            "anchors": ("no analytics", "Google Analytics only", "custom BI dashboards with real-time data"),
        },
    ],
    "staff_digital_literacy": [
        {
            "prompt": "How confident is the owner/manager in adopting new digital tools?",
            "anchors": ("resistant to change", "open but cautious", "actively seeks new tools"),
        },
        {
            "prompt": "How capable is your team at using your current digital tools effectively?",
            "anchors": ("struggle with basics", "use standard features", "use advanced features confidently"),
        },
        {
            "prompt": "How quickly does your team typically adopt new software or processes?",
            "anchors": ("very slowly, months", "a few weeks with training", "quickly, within days"),
        },
        {
            "prompt": "How well would your team identify and escalate an AI error or unusual output?",
            "anchors": ("would not notice", "might notice obvious errors", "would identify and escalate reliably"),
        },
    ],
    "governance_compliance": [
        {
            "prompt": "How comprehensive and current is your privacy policy?",
            "anchors": ("no policy", "basic policy, outdated", "current, covers AI processing"),
        },
        {
            "prompt": "How well do you manage cookie consent and data subject requests?",
            "anchors": ("no consent mechanism", "basic cookie banner", "full consent management platform"),
        },
        {
            "prompt": "How current are your Companies House filings and company records?",
            "anchors": ("overdue", "up to date", "up to date with additional governance documentation"),
        },
        {
            "prompt": "Do you have a staff AI use policy or guidelines?",
            "anchors": ("no policy", "informal guidelines", "formal written policy reviewed annually"),
        },
    ],
    "ai_tool_experience": [
        {
            "prompt": "What AI or personalisation tools does your business currently use?",
            "anchors": ("none", "basic email automation", "personalisation engine or gen-AI tool"),
        },
        {
            "prompt": "How actively do you use marketing automation in your customer communications?",
            "anchors": ("no automation", "basic welcome sequence", "full lifecycle automation with segmentation"),
        },
        {
            "prompt": "How much experience does your team have evaluating AI-generated outputs?",
            "anchors": ("no experience", "occasional use", "regular structured review process"),
        },
        {
            "prompt": "How familiar are you with the AI tools available for fashion retail specifically?",
            "anchors": ("not at all", "aware of a few", "actively researched and trialled several"),
        },
    ],
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def compute_pillar_score(scores4: list) -> float:
    """Mirrors agents.agent1_capability.compute_stream_a's per-pillar formula."""
    raw_mean = sum(scores4) / len(scores4)
    return round(((raw_mean - 1) / 8) * 9 + 1, 2)


def get_pillar_scores() -> dict:
    return {
        pillar: compute_pillar_score([st.session_state[f"{pillar}_{i}"] for i in range(4)])
        for pillar in PILLARS
    }


def tier_badge_html(tier: str) -> str:
    color = TIER_COLORS.get(tier, "#6b7280")
    return (
        f"<span style='background:{color}; color:white; padding:4px 14px; "
        f"border-radius:999px; font-weight:600; font-size:0.95rem;'>{tier}</span>"
    )


def render_logo():
    st.markdown(
        f"<div style='font-size:3rem; font-weight:800; color:{TEAL}; "
        f"letter-spacing:0.02em; line-height:1;'>AIYYARY</div>",
        unsafe_allow_html=True,
    )
    st.caption("AI Readiness & Opportunity Assessment for Fashion Retail")


def _load_json_safe(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _parse_timestamp(value: str):
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def list_saved_sessions() -> list:
    """Groups outputs/agent{1,2,3}_*.json files into complete three-agent
    sessions, so a past run can be loaded straight into screen 3 without
    re-running the pipeline. Filenames don't cross-reference each other, so
    sessions are matched by company_name + nearest-following timestamp
    (agent1 -> agent2 -> agent3 run in that order within a single session).
    """

    def _entries(pattern):
        out = []
        for path in sorted(glob.glob(os.path.join(OUTPUTS_DIR, pattern))):
            data = _load_json_safe(path)
            if not data:
                continue
            ts = _parse_timestamp(data.get("assessment_timestamp", ""))
            if ts is None:
                continue
            out.append({"path": path, "company_name": data.get("company_name", ""), "timestamp": ts, "data": data})
        return out

    a1_entries = _entries("agent1_*.json")
    a2_entries = _entries("agent2_*.json")
    a3_entries = _entries("agent3_*.json")

    sessions = []
    for e1 in a1_entries:
        candidates2 = [
            e for e in a2_entries if e["company_name"] == e1["company_name"] and e["timestamp"] >= e1["timestamp"]
        ]
        if not candidates2:
            continue
        e2 = min(candidates2, key=lambda e: e["timestamp"])

        candidates3 = [
            e for e in a3_entries if e["company_name"] == e1["company_name"] and e["timestamp"] >= e2["timestamp"]
        ]
        if not candidates3:
            continue
        e3 = min(candidates3, key=lambda e: e["timestamp"])

        sessions.append(
            {
                "company_name": e1["company_name"],
                "timestamp": e1["timestamp"],
                "overall_score": e1["data"].get("overall_score"),
                "readiness_tier": e1["data"].get("readiness_tier"),
                "agent1_path": e1["path"],
                "agent2_path": e2["path"],
                "agent3_path": e3["path"],
            }
        )

    sessions.sort(key=lambda s: s["timestamp"], reverse=True)
    return sessions


def _text_from_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if isinstance(content.get("instructions"), list):
            return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(content["instructions"]))
        if isinstance(content.get("steps"), list):
            return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(content["steps"]))
        return json.dumps(content, indent=2, ensure_ascii=False)
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def init_state():
    defaults = {
        "screen": 1,
        "company_name": "",
        "website_url": "",
        "challenge": CHALLENGE_OPTIONS[0],
        "challenge_other": "",
        "staff_count": STAFF_OPTIONS[0],
        "sales_channel": CHANNEL_OPTIONS[0],
        "pipeline_ran": False,
        "pipeline_error": None,
        "agent1_output": None,
        "agent2_output": None,
        "agent3_output": None,
        "profile": None,
        "session_start": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    for pillar in PILLARS:
        for i in range(4):
            st.session_state.setdefault(f"{pillar}_{i}", 5)


# ---------------------------------------------------------------------------
# Screen 1 — Business Profile
# ---------------------------------------------------------------------------


def render_load_saved_section():
    sessions = list_saved_sessions()
    if not sessions:
        return

    with st.expander(f"📂 Load a previous saved report ({len(sessions)} available)"):
        labels = [
            f"{s['company_name']} — {s['timestamp']:%Y-%m-%d %H:%M} — {s['overall_score']}/9 ({s['readiness_tier']})"
            for s in sessions
        ]
        idx = st.selectbox(
            "Saved reports", range(len(sessions)), format_func=lambda i: labels[i], key="saved_session_idx"
        )
        if st.button("Load Report →", use_container_width=True):
            session = sessions[idx]
            agent1_output = _load_json_safe(session["agent1_path"])
            agent2_output = _load_json_safe(session["agent2_path"])
            agent3_output = _load_json_safe(session["agent3_path"])
            if not (agent1_output and agent2_output and agent3_output):
                st.error("Could not load one or more saved files for this report.")
            else:
                st.session_state.company_name = session["company_name"]
                st.session_state.agent1_output = agent1_output
                st.session_state.agent2_output = agent2_output
                st.session_state.agent3_output = agent3_output
                st.session_state.pipeline_ran = True
                st.session_state.pipeline_error = None
                st.session_state.profile = None
                st.session_state.session_start = session["timestamp"]
                st.session_state.screen = 3
                st.rerun()

    st.divider()


def render_screen1():
    render_logo()
    render_load_saved_section()
    st.subheader("Business Profile")

    st.session_state.company_name = st.text_input("Company name *", value=st.session_state.company_name)
    st.session_state.website_url = st.text_input("Website URL *", value=st.session_state.website_url)

    st.session_state.challenge = st.selectbox(
        "Primary business challenge",
        CHALLENGE_OPTIONS,
        index=CHALLENGE_OPTIONS.index(st.session_state.challenge),
    )
    if st.session_state.challenge == "Other — specify below":
        st.session_state.challenge_other = st.text_input(
            "Please specify your primary challenge", value=st.session_state.challenge_other
        )

    st.session_state.staff_count = st.selectbox(
        "Number of staff", STAFF_OPTIONS, index=STAFF_OPTIONS.index(st.session_state.staff_count)
    )
    st.session_state.sales_channel = st.selectbox(
        "Primary sales channel", CHANNEL_OPTIONS, index=CHANNEL_OPTIONS.index(st.session_state.sales_channel)
    )

    st.write("")
    if st.button("Start Assessment →", type="primary", use_container_width=True):
        errors = []
        if not st.session_state.company_name.strip():
            errors.append("Company name is required.")
        if not st.session_state.website_url.strip():
            errors.append("Website URL is required.")
        if st.session_state.challenge == "Other — specify below" and not st.session_state.challenge_other.strip():
            errors.append("Please specify your primary challenge.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            st.session_state.screen = 2
            st.rerun()


# ---------------------------------------------------------------------------
# Screen 2 — 24-question instrument
# ---------------------------------------------------------------------------


def render_sidebar_live():
    scores = get_pillar_scores()
    overall = round(sum(scores.values()) / len(scores), 2)
    tier = readiness_tier(overall)

    st.sidebar.markdown(f"<div style='font-size:1.6rem; font-weight:800; color:{TEAL};'>AIYYARY</div>", unsafe_allow_html=True)
    st.sidebar.caption(st.session_state.company_name or "Live assessment")
    st.sidebar.divider()

    st.sidebar.metric("Overall score (live)", f"{overall} / 9")
    st.sidebar.markdown(tier_badge_html(tier), unsafe_allow_html=True)
    st.sidebar.write("")

    st.sidebar.markdown("**Pillar scores**")
    for pillar in PILLARS:
        st.sidebar.caption(f"{PILLAR_DISPLAY_NAMES[pillar]} — {scores[pillar]}")
        st.sidebar.progress(min(1.0, max(0.0, scores[pillar] / 9)))

    return scores, overall, tier


def render_pillar_section(pillar: str):
    color = PILLAR_COLORS[pillar]
    st.markdown(
        f"<h3 style='color:{color}; margin-bottom:0.2rem;'>{PILLAR_DISPLAY_NAMES[pillar]}</h3>",
        unsafe_allow_html=True,
    )

    for i, q in enumerate(QUESTIONS[pillar]):
        st.markdown(f"**Q{i + 1}. {q['prompt']}**")
        st.slider(f"{pillar}_{i}_slider", min_value=1, max_value=9, key=f"{pillar}_{i}", label_visibility="collapsed")
        low, mid, high = q["anchors"]
        c1, c2, c3 = st.columns(3)
        c1.caption(f"1 = {low}")
        c2.caption(f"5 = {mid}")
        c3.caption(f"9 = {high}")
        st.write("")

    vals = [st.session_state[f"{pillar}_{i}"] for i in range(4)]
    score = compute_pillar_score(vals)
    raw_mean = sum(vals) / len(vals)
    st.info(
        f"**{PILLAR_DISPLAY_NAMES[pillar]} score: {score} / 9**  "
        f"(weighted formula: ((mean {raw_mean:.2f} − 1) / 8) × 9 + 1)"
    )
    st.divider()


def render_screen2():
    render_sidebar_live()
    render_logo()
    st.subheader("24-Question Digital Readiness Instrument")
    st.caption(f"{st.session_state.company_name} — rate each question from 1 (lowest) to 9 (highest)")
    st.write("")

    for pillar in PILLARS:
        render_pillar_section(pillar)

    if st.button("Run AIYYARY →", type="primary", use_container_width=True):
        st.session_state.screen = 3
        st.session_state.pipeline_ran = False
        st.session_state.pipeline_error = None
        st.rerun()


# ---------------------------------------------------------------------------
# Screen 3 — Results
# ---------------------------------------------------------------------------


def build_instrument() -> dict:
    instrument = {}
    for pillar in PILLARS:
        answers = []
        for i, q in enumerate(QUESTIONS[pillar]):
            score = st.session_state[f"{pillar}_{i}"]
            answers.append({"question": q["prompt"], "answer": f"owner-provided score {score}", "score": score})
        instrument[pillar] = answers
    return instrument


def resolve_challenge_text() -> str:
    if st.session_state.challenge == "Other — specify below":
        return st.session_state.challenge_other.strip()
    return st.session_state.challenge


def build_profile() -> dict:
    company_name = st.session_state.company_name
    return {
        "business_name": company_name,
        "company_name": company_name,
        "website": st.session_state.website_url,
        "location": "UK",
        "instagram_handle": None,
        "primary_challenge": resolve_challenge_text(),
        "industry": "Fashion and Apparel",
        "staff_count": st.session_state.staff_count,
        "sales_channel": st.session_state.sales_channel,
    }


def run_pipeline():
    session_start = datetime.now()
    profile = build_profile()
    instrument = build_instrument()

    enrichment_report = EnrichmentAgent().run(profile)
    agent1_output = CapabilityAssessmentAgent().run(profile, instrument, enrichment_report=enrichment_report)
    agent2_output = OpportunityResearchAgent().run(profile, agent1_output, execute_top_n=1)
    agent3_output = RiskTransformationAgent().run(agent1_output, agent2_output, profile=profile)

    st.session_state.profile = profile
    st.session_state.session_start = session_start
    st.session_state.agent1_output = agent1_output
    st.session_state.agent2_output = agent2_output
    st.session_state.agent3_output = agent3_output
    st.session_state.pipeline_ran = True


def render_tab_capability(agent1: dict):
    overall = agent1.get("overall_score", 0)
    tier = agent1.get("readiness_tier", "Early-Stage")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(
            f"<div style='font-size:3.2rem; font-weight:800; color:{TEAL}; line-height:1;'>{overall} / 9</div>",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(tier_badge_html(tier), unsafe_allow_html=True)
    with c2:
        pillar_scores = agent1.get("pillar_scores", {})
        df = pd.DataFrame(
            {
                "Pillar": [PILLAR_DISPLAY_NAMES[p] for p in PILLARS],
                "Score": [pillar_scores.get(p, {}).get("final_score", 0) for p in PILLARS],
            }
        ).set_index("Pillar")
        st.bar_chart(df)

    st.markdown("#### Verdict")
    st.write(agent1.get("verdict", "") or "(no verdict available)")

    st.markdown("#### Data Confidence")
    st.info(agent1.get("data_confidence_note", "") or "(no confidence note available)")

    summary = agent1.get("enrichment_summary", {})
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Enrichment sources available**")
        available = summary.get("sources_available") or []
        st.write(", ".join(available) if available else "None")
    with c4:
        st.markdown("**Enrichment sources failed**")
        failed = summary.get("sources_failed") or []
        st.write(", ".join(failed) if failed else "None")


def render_tab_opportunities(agent2: dict):
    recommended = agent2.get("recommended") or []
    deferred = agent2.get("deferred") or []

    if recommended:
        top = recommended[0]
        st.markdown("#### 🏆 Top Recommendation")
        with st.container(border=True):
            st.markdown(f"**{top['title']}**  ·  `{top['id']}`")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Effort", top.get("effort", "n/a"))
            c2.metric("Executor mode", top.get("executor_mode", "n/a"))
            status = (top.get("executor_result") or {}).get("status", "not dispatched")
            c3.metric("Status", status)
            c4.metric("Est. weeks", top.get("implementation_weeks", "n/a"))
            st.write(top.get("justification", ""))

    quick_win = agent2.get("quick_win_summary", "")
    if quick_win:
        st.success(f"**Quick win:** {quick_win}")

    st.markdown("#### All Opportunities Considered")
    rows = []
    for r in recommended:
        status = (r.get("executor_result") or {}).get("status", "not dispatched")
        rows.append(
            {
                "Title": r.get("title", ""),
                "Type": "Recommended",
                "Effort": r.get("effort", ""),
                "Executor Mode": r.get("executor_mode", ""),
                "Status / Reason": status,
            }
        )
    for d in deferred:
        rows.append(
            {
                "Title": d.get("title", ""),
                "Type": "Deferred",
                "Effort": "-",
                "Executor Mode": "-",
                "Status / Reason": d.get("unlock_condition", d.get("reason", "")),
            }
        )

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.write("No opportunities were evaluated.")


def render_tab_risk(agent3: dict):
    risk_register = agent3.get("risk_register") or []

    st.markdown("#### Risk Register")
    if risk_register:
        df = pd.DataFrame(
            [
                {
                    "Category": r.get("label", ""),
                    "Severity": r.get("severity", ""),
                    "Pillar": PILLAR_DISPLAY_NAMES.get(r.get("pillar"), r.get("pillar", "")),
                    "Trigger": r.get("trigger_reason", ""),
                    "Mitigation": r.get("mitigation", ""),
                }
                for r in risk_register
            ]
        )

        def color_severity(val):
            return f"background-color: {SEVERITY_COLORS.get(val, '')}; font-weight:600;"

        styler = df.style
        style_fn = getattr(styler, "map", None) or styler.applymap
        styled = style_fn(color_severity, subset=["Severity"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.write("No risk register available.")

    st.markdown("#### Phased Roadmap")
    roadmap = agent3.get("roadmap") or []
    if not roadmap:
        st.write("No roadmap available.")
    for phase in roadmap:
        title = f"Phase {phase.get('phase', '?')}: {phase.get('title', '')} · {phase.get('timeframe_weeks', 0)} weeks"
        with st.expander(title, expanded=True):
            st.write(phase.get("description", ""))
            agents_list = phase.get("recommended_ai_agents") or []
            if agents_list:
                st.markdown("**AI marketplace agents:**")
                for a in agents_list:
                    cost = "Free tier" if a.get("free_tier") else a.get("estimated_monthly_cost_gbp", "n/a")
                    st.write(f"- **{a.get('name', '')}** — {a.get('what_it_does_here', '')} ({cost})")
            if phase.get("human_oversight_required"):
                st.caption("⚠ Human oversight required for this phase")

    st.markdown("#### Headline ROI")
    st.markdown(
        f"<div style='font-size:2.2rem; font-weight:800; color:{TEAL};'>{agent3.get('headline_metric', 'n/a')}</div>",
        unsafe_allow_html=True,
    )
    st.caption(agent3.get("roi_summary", ""))
    st.caption(f"Total roadmap: {agent3.get('total_roadmap_weeks', 0)} weeks across {len(roadmap)} phases")


def render_tab_deliverables(agent2: dict):
    recommended = agent2.get("recommended") or []
    top = recommended[0] if recommended else None

    if not top or not top.get("executor_result"):
        st.info("No executor deliverables were produced for this run.")
        return

    executor_result = top["executor_result"]
    st.markdown(f"### {top.get('title', 'Opportunity')}")
    st.caption(f"Executor mode: {top.get('executor_mode', 'n/a')}  ·  Status: {executor_result.get('status', 'unknown')}")

    if executor_result.get("error"):
        st.error(executor_result["error"])
        return

    if top.get("executor_mode") == "market_connect":
        tools = executor_result.get("market_tools_recommended") or []
        if tools:
            st.markdown("#### Ranked Tools")
            for t in tools:
                with st.container(border=True):
                    st.markdown(f"**Rank {t.get('rank', '?')}: {t.get('tool_name', 'Unknown tool')}**")
                    st.write(t.get("why_this_business", ""))
                    free = "Yes" if t.get("free_tier_available") else "No"
                    st.caption(f"Free tier: {free}  ·  Est. cost: {t.get('estimated_monthly_cost_gbp', 'n/a')}")

        deliverables = executor_result.get("deliverables") or []
        if deliverables:
            st.markdown("#### Configuration Guide")
            st.write(_text_from_content(deliverables[0].get("content")))
    else:
        deliverables = executor_result.get("deliverables") or []
        if not deliverables:
            st.write("No deliverables were produced for this opportunity.")
        for d in deliverables:
            name = d.get("name", "deliverable")
            name_l = name.lower()
            content = d.get("content")
            with st.expander(f"{name.replace('_', ' ').title()} ({d.get('type', '')})", expanded=True):
                if "reason_codes" in name_l and isinstance(content, dict):
                    for rc in content.get("reason_codes", []):
                        st.write(f"**[{rc.get('code', '?')}] {rc.get('label', '')}**")
                        subs = rc.get("sub_reasons") or []
                        if subs:
                            st.caption("Sub-reasons: " + ", ".join(str(s) for s in subs))
                elif ("form_structure" in name_l or "question" in name_l) and isinstance(content, dict):
                    form = content.get("form_structure", content)
                    questions = form.get("questions", []) if isinstance(form, dict) else []
                    for q in questions:
                        st.write(f"**{q.get('prompt', '')}** ({q.get('type', '')})")
                        opts = q.get("options")
                        if opts:
                            st.caption("Options: " + ", ".join(str(o) for o in opts))
                elif ("email" in name_l or "sequence" in name_l) and isinstance(content, dict):
                    for e in content.get("sequence", []):
                        st.write(f"**Email {e.get('email_number', '?')} — {e.get('send_timing', '')}**")
                        st.write(f"Subject: {e.get('subject_line', '')}")
                        st.write(e.get("body_copy", ""))
                        st.caption(f"CTA: {e.get('call_to_action', '')}")
                else:
                    st.write(_text_from_content(content))

    next_steps = executor_result.get("next_steps") or []
    if next_steps:
        st.markdown("#### Next Steps")
        for i, step in enumerate(next_steps, start=1):
            st.write(f"{i}. {step}")

    missing = executor_result.get("missing_prerequisites") or []
    if missing:
        st.markdown("#### Missing Prerequisites")
        for item in missing:
            st.write(f"- {item}")


def render_screen3():
    render_logo()
    company = st.session_state.company_name
    st.subheader(f"Results — {company}")

    if not st.session_state.pipeline_ran:
        with st.spinner("Running three-agent pipeline... this takes 60–90 seconds"):
            try:
                run_pipeline()
            except Exception as exc:
                st.session_state.pipeline_error = str(exc)

        if st.session_state.pipeline_error:
            st.error(f"Pipeline failed: {st.session_state.pipeline_error}")
            if st.button("← Back to instrument"):
                st.session_state.screen = 2
                st.session_state.pipeline_error = None
                st.rerun()
            return

    agent1 = st.session_state.agent1_output
    agent2 = st.session_state.agent2_output
    agent3 = st.session_state.agent3_output

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Capability Assessment", "💡 Opportunities", "⚠️ Risk & Roadmap", "📦 Deliverables"]
    )
    with tab1:
        render_tab_capability(agent1)
    with tab2:
        render_tab_opportunities(agent2)
    with tab3:
        render_tab_risk(agent3)
    with tab4:
        render_tab_deliverables(agent2)

    st.divider()
    report_json = json.dumps(agent3, indent=2, ensure_ascii=False)
    safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in (company or "unknown"))
    st.download_button(
        "⬇ Download Full Report (JSON)",
        data=report_json,
        file_name=f"aiyyary_report_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    init_state()
    if st.session_state.screen == 1:
        render_screen1()
    elif st.session_state.screen == 2:
        render_screen2()
    else:
        render_screen3()


main()
