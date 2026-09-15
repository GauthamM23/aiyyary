"""AIYYARY orchestrator: single-command runner for the full three-agent
pipeline against any business — not just the five pre-defined demo
companies covered by demo_five_companies.py / demo_agent2.py / demo_agent3.py.

    python main.py --company "Business Name" --url "https://their-website.com"
    python main.py --company "Business Name" --url "https://their-website.com" --answers answers.json
    python main.py --company "Business Name" --url "https://their-website.com" --interactive

Runs Agent 1 (capability assessment) -> Agent 2 (opportunity research +
executors) -> Agent 3 (risk-aware roadmap) sequentially for one company,
then prints the rank-1 opportunity's full executor deliverables (reusing
show_deliverables.py's renderer in-process, not via subprocess) and a
session summary.
"""

import argparse
import json
import sys
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from agents.agent1_capability import PILLARS, CapabilityAssessmentAgent
from agents.agent2_opportunity import OpportunityResearchAgent
from agents.agent3_risk import RiskTransformationAgent
from agents.enrichment import EnrichmentAgent
from demo_agent2 import print_company_summary as print_agent2_summary
from demo_agent3 import print_company_summary as print_agent3_summary
from demo_five_companies import print_company_summary as print_agent1_summary
from show_deliverables import print_company as print_deliverables_report

LINE_WIDTH = 70
HEAVY = "═" * LINE_WIDTH

PILLAR_DISPLAY_NAMES = {
    "data_foundations": "Data Foundations",
    "process_digitisation": "Process Digitisation",
    "technology_infrastructure": "Technology Infrastructure",
    "staff_digital_literacy": "Staff Digital Literacy",
    "governance_compliance": "Governance & Compliance",
    "ai_tool_experience": "AI Tool Experience",
}

# Four questions per pillar (24 total), each with a 1/5/9 anchor scale, used
# by --interactive mode. This is the same owner-interview instrument shape
# consumed by compute_stream_a() in agents/agent1_capability.py.
INSTRUMENT_QUESTIONS = {
    "data_foundations": [
        {
            "prompt": "How structured and queryable is your customer purchase history?",
            "anchors": "1=paper/spreadsheet, 5=basic CRM, 9=fully structured database",
        },
        {
            "prompt": "How complete and accurate is your product inventory data?",
            "anchors": "1=manual stock count, 5=basic inventory system, 9=automated real-time",
        },
        {
            "prompt": "How well do you track customer behaviour and browsing patterns?",
            "anchors": "1=no tracking, 5=basic Google Analytics, 9=full behavioural analytics",
        },
        {
            "prompt": "How consistently do you collect and store returns reason data?",
            "anchors": "1=not at all, 5=informal notes, 9=structured categorised database",
        },
    ],
    "process_digitisation": [
        {
            "prompt": "How much of your order fulfilment process is automated vs manual?",
            "anchors": "1=fully manual, 5=partially automated, 9=fully automated workflow",
        },
        {
            "prompt": "How digitised is your returns and exchanges process?",
            "anchors": "1=phone/email only, 5=basic online form, 9=fully automated returns portal",
        },
        {
            "prompt": "How integrated are your sales channels (online, in-store, marketplace)?",
            "anchors": "1=fully separate systems, 5=partially synced, 9=fully unified inventory/orders",
        },
        {
            "prompt": "How much of your customer service runs through digital channels vs phone/in-person?",
            "anchors": "1=mostly phone/in-person, 5=mixed, 9=mostly digital/self-serve",
        },
    ],
    "technology_infrastructure": [
        {
            "prompt": "What is the maturity of your e-commerce platform?",
            "anchors": "1=no online store, 5=basic Shopify/Wix store, 9=custom/enterprise platform",
        },
        {
            "prompt": "How reliable and fast is your website performance?",
            "anchors": "1=frequent downtime/slow, 5=generally stable, 9=fast with monitored uptime",
        },
        {
            "prompt": "How well integrated are your backend systems (inventory, POS, e-commerce)?",
            "anchors": "1=fully disconnected, 5=partially integrated, 9=single source of truth",
        },
        {
            "prompt": "How scalable is your current tech stack for growth?",
            "anchors": "1=would break under 2x traffic, 5=could handle moderate growth, 9=built to scale",
        },
    ],
    "staff_digital_literacy": [
        {
            "prompt": "How comfortable is your team using digital tools day-to-day?",
            "anchors": "1=very uncomfortable, 5=comfortable with basics, 9=highly proficient across tools",
        },
        {
            "prompt": "How much ongoing digital/technical training does your team receive?",
            "anchors": "1=none, 5=occasional informal training, 9=structured regular programme",
        },
        {
            "prompt": "How would you rate your own (owner/manager) digital confidence?",
            "anchors": "1=very low, 5=moderate, 9=very high",
        },
        {
            "prompt": "How willing is your team to adopt new digital tools or AI systems?",
            "anchors": "1=resistant, 5=neutral/willing to try, 9=enthusiastic early adopters",
        },
    ],
    "governance_compliance": [
        {
            "prompt": "How up to date and comprehensive is your privacy policy?",
            "anchors": "1=none/outdated, 5=basic GDPR-compliant policy, 9=covers AI/data processing explicitly",
        },
        {
            "prompt": "How well documented are your data handling and retention practices?",
            "anchors": "1=not documented, 5=informal practices, 9=fully documented and audited",
        },
        {
            "prompt": "How current are your company filings and regulatory compliance?",
            "anchors": "1=overdue/lapsed, 5=up to date but reactive, 9=proactively managed",
        },
        {
            "prompt": "Do you have a policy or plan for responsible AI use?",
            "anchors": "1=no awareness, 5=informal awareness, 9=documented AI use policy",
        },
    ],
    "ai_tool_experience": [
        {
            "prompt": "How much experience does your business have using AI-powered tools?",
            "anchors": "1=none, 5=tried one or two tools, 9=multiple AI tools in active use",
        },
        {
            "prompt": "How would you rate your understanding of what AI could do for your business?",
            "anchors": "1=very limited, 5=general understanding, 9=clear strategic view",
        },
        {
            "prompt": "How data-driven are your current marketing decisions?",
            "anchors": "1=gut feel only, 5=basic analytics informed, 9=fully data/AI-driven",
        },
        {
            "prompt": "How open is your business to investing in AI tools in the next 12 months?",
            "anchors": "1=not open, 5=open if ROI is clear, 9=actively planning investment",
        },
    ],
}


# ---------------------------------------------------------------------------
# Step 1 — argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AIYYARY — run the full three-agent pipeline for one business"
    )
    parser.add_argument("--company", required=True, help='Company name, e.g. "Lucy & Yak"')
    parser.add_argument("--url", required=True, help="Company website URL")
    parser.add_argument(
        "--answers",
        help="Path to a JSON file of pre-filled Stream A instrument answers (see README for format)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for all 24 Stream A instrument answers in the terminal",
    )
    parser.add_argument(
        "--challenge",
        help="Primary business challenge (prompted for if not supplied)",
    )
    return parser.parse_args()


def resolve_challenge(challenge_arg: str) -> str:
    if challenge_arg:
        return challenge_arg
    default = "Improving digital capability and identifying practical AI opportunities"
    try:
        raw = input(f"Primary business challenge [{default}]: ").strip()
    except EOFError:
        raw = ""
    return raw or default


# ---------------------------------------------------------------------------
# Step 2 — Stream A instrument collection
# ---------------------------------------------------------------------------


def _instrument_from_raw_scores(raw_scores: dict) -> dict:
    instrument = {}
    for pillar in PILLARS:
        scores = raw_scores.get(pillar)
        if not scores or len(scores) != 4:
            raise ValueError(f"Instrument data must provide exactly 4 scores for '{pillar}'")
        instrument[pillar] = [
            {"question": f"{pillar} question {i + 1}", "answer": f"provided score {s}", "score": s}
            for i, s in enumerate(scores)
        ]
    return instrument


def load_instrument_from_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw_scores = json.load(f)
    return _instrument_from_raw_scores(raw_scores)


def default_instrument() -> dict:
    print("[WARNING] No --answers file and --interactive not set — using a flat")
    print("          mid-range default (all scores = 5). This will NOT produce a")
    print("          valid assessment; supply --answers or --interactive for a")
    print("          real Stream A owner-interview result.")
    print()
    return _instrument_from_raw_scores({pillar: [5, 5, 5, 5] for pillar in PILLARS})


def _prompt_for_score() -> int:
    while True:
        try:
            raw = input("      Answer: ").strip()
        except EOFError:
            raise SystemExit("\n[ERROR] Interactive input ended unexpectedly — 24 answers were required.")
        try:
            value = int(raw)
        except ValueError:
            print("      Please enter a whole number from 1 to 9.")
            continue
        if 1 <= value <= 9:
            return value
        print("      Please enter a number from 1 to 9.")


def collect_instrument_interactive(company_name: str) -> dict:
    print(HEAVY)
    print("AIYYARY — Live Evaluation Session")
    print(f"Company: {company_name}")
    print(HEAVY)
    print()

    instrument = {}
    for pillar_idx, pillar in enumerate(PILLARS, start=1):
        print(f"PILLAR {pillar_idx} OF 6: {PILLAR_DISPLAY_NAMES[pillar]}")
        print("Rate each question 1-9 (1=very low, 5=mid, 9=very high)")
        print()
        answers = []
        for q_idx, question in enumerate(INSTRUMENT_QUESTIONS[pillar], start=1):
            print(f"  Q{q_idx}. {question['prompt']}")
            print(f"      ({question['anchors']})")
            score = _prompt_for_score()
            answers.append(
                {"question": question["prompt"], "answer": f"owner-provided score {score}", "score": score}
            )
            print()
        instrument[pillar] = answers

    return instrument


# ---------------------------------------------------------------------------
# Steps 3-5 — the three-agent pipeline
# ---------------------------------------------------------------------------


def build_profile(company_name: str, url: str, challenge: str) -> dict:
    return {
        "business_name": company_name,
        "company_name": company_name,
        "website": url,
        "location": "UK",
        "instagram_handle": None,
        "primary_challenge": challenge,
        "industry": "Fashion and Apparel",
    }


def print_progress_header(title: str, company_name: str, url: str) -> None:
    print(HEAVY)
    print(title)
    print(f"Company: {company_name}  |  URL: {url}")
    print(HEAVY)
    print()


# ---------------------------------------------------------------------------
# Step 7 — session summary
# ---------------------------------------------------------------------------


def print_session_summary(company_name: str, session_start: datetime, agent1_output: dict, agent2_output: dict, agent3_output: dict) -> None:
    recommended = agent2_output.get("recommended") or []
    top = recommended[0] if recommended else None
    top_line = f"{top['title']} ({top['id']})" if top else "(none survived the prerequisite filter)"
    roadmap = agent3_output.get("roadmap") or []

    print()
    print(HEAVY)
    print("AIYYARY SESSION COMPLETE")
    print(HEAVY)
    print(f"Company:        {company_name}")
    print(f"Assessment:     {session_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Overall score:  {agent1_output['overall_score']:.2f}   Tier: {agent1_output['readiness_tier']}")
    print()
    print("Output files saved:")
    print(f"  Agent 1: {agent1_output['traceability']['output_file']}")
    print(f"  Agent 2: {agent2_output['traceability']['output_file']}")
    print(f"  Agent 3: {agent3_output['traceability']['output_file']}")
    print()
    print(f"Top recommendation: {top_line}")
    print(f"Headline ROI:       {agent3_output.get('headline_metric', 'n/a')}")
    print(f"Total roadmap:      {agent3_output.get('total_roadmap_weeks', 0)} weeks across {len(roadmap)} phases")
    print()
    print("To view full deliverables at any time:")
    print(f'  python show_deliverables.py --company "{company_name}"')
    print(HEAVY)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    session_start = datetime.now()

    challenge = resolve_challenge(args.challenge)
    profile = build_profile(args.company, args.url, challenge)

    if args.answers:
        instrument = load_instrument_from_file(args.answers)
    elif args.interactive:
        instrument = collect_instrument_interactive(args.company)
    else:
        instrument = default_instrument()

    # ---------------- Agent 1 — Digital Capability Assessment ----------------
    print_progress_header("RUNNING AGENT 1 — Digital Capability Assessment", args.company, args.url)
    enrichment_report = EnrichmentAgent().run(profile)
    agent1_output = CapabilityAssessmentAgent().run(profile, instrument, enrichment_report=enrichment_report)
    print_agent1_summary(agent1_output)

    # ---------------- Agent 2 — AI Opportunity Research -----------------------
    print_progress_header("RUNNING AGENT 2 — AI Opportunity Research", args.company, args.url)
    agent2_output = OpportunityResearchAgent().run(profile, agent1_output, execute_top_n=1)
    print_agent2_summary(agent2_output)

    # ---------------- Agent 3 — Risk-Aware Transformation ---------------------
    print_progress_header("RUNNING AGENT 3 — Risk-Aware Transformation", args.company, args.url)
    agent3_output = RiskTransformationAgent().run(agent1_output, agent2_output, profile=profile)
    print_agent3_summary(agent3_output)

    # ---------------- Step 6 — executor deliverables (rank-1 opportunity) -----
    print(HEAVY)
    print("EXECUTOR DELIVERABLES — Rank 1 Opportunity")
    print(HEAVY)
    print()
    print_deliverables_report(agent2_output, False)

    # ---------------- Step 7 — session summary ---------------------------------
    print_session_summary(args.company, session_start, agent1_output, agent2_output, agent3_output)


if __name__ == "__main__":
    main()
