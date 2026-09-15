"""Demo runner: the full Agent 3 pipeline against the five companies from demo_five_companies.py.

For each company this script:
  1. Loads the most recent Agent 1 and Agent 2 output JSONs for that company
     from outputs/, matched by the company_name field inside each JSON
     (not by filename). If Agent 2 output does not exist for a company,
     prints a warning and skips — does not crash.
  2. Runs the full Agent 3 pipeline (deterministic risk register, LLM
     mitigations, deterministic phase structure, LLM roadmap, LLM ROI
     summary).
  3. Prints a per-company risk summary.
  4. Prints a final comparison table across all companies that ran successfully.
"""

import argparse
import glob
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from agents.agent3_risk import RiskTransformationAgent
from demo_five_companies import DEMO_COMPANIES


# Shared loader this script uses twice — once for "agent1" and once for
# "agent2" prefixes — to pull in the two upstream JSON contracts Agent 3
# depends on. Matches by the company_name field *inside* the JSON, not by
# filename, so it's immune to the timestamp/company-name-sanitisation
# differences between how Agent 1 and Agent 2 name their output files.
def _find_latest_output(prefix: str, company_name: str) -> tuple:
    """Return (path, data) for the most recently modified {prefix}_*.json in
    outputs/ whose company_name field matches. (None, None) if none found."""
    output_dir = os.getenv("OUTPUT_DIR", "./outputs")
    pattern = os.path.join(output_dir, f"{prefix}_*.json")
    candidates = []
    for path in glob.glob(pattern):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("company_name") == company_name:
            candidates.append((os.path.getmtime(path), path, data))

    if not candidates:
        return None, None
    candidates.sort(key=lambda c: c[0])
    _, path, data = candidates[-1]
    return path, data


# Per-company driver for this demo: loads this company's most recent Agent 1
# and Agent 2 outputs (skipping cleanly, not crashing, if either is missing)
# and runs the real RiskTransformationAgent pipeline against them — the
# deterministic risk register, the phased roadmap, and the ROI headline. This
# is the last of the three per-agent demo drivers; its output (agent3_*.json)
# is the final artefact summarised in print_company_summary() below.
def run_company(company: dict):
    company_name = company["company_name"]

    agent1_path, agent1_output = _find_latest_output("agent1", company_name)
    if agent1_output is None:
        print(f"[SKIP] No Agent 1 output found for {company_name} in outputs/. Run demo_five_companies.py first.")
        return None

    agent2_path, agent2_output = _find_latest_output("agent2", company_name)
    if agent2_output is None:
        print(f"[SKIP] No Agent 2 output found for {company_name} in outputs/. Run demo_agent2.py first.")
        return None

    profile = {
        "company_name": company_name,
        "business_name": company_name,
        "primary_challenge": company["primary_challenge"],
    }

    print(f"Using Agent 1 output: {agent1_path}")
    print(f"Using Agent 2 output: {agent2_path}")

    agent = RiskTransformationAgent()
    result = agent.run(agent1_output, agent2_output, profile=profile)
    return result


def print_company_summary(result: dict) -> None:
    print("-" * 78)
    print(result["company_name"])
    print("-" * 78)
    print(f"  Overall score: {result['overall_score']:.2f}   Tier: {result['readiness_tier']}")
    print(
        f"  Risk severity — HIGH: {result['high_severity_count']}  "
        f"MEDIUM: {result['medium_severity_count']}  LOW: {result['low_severity_count']}"
    )

    for r in result["risk_register"]:
        print(f"    [{r['severity']:<6s}] {r['label']}")

    print(f"\n  Phase 1 required (HIGH risk remediation): {result['phase_structure']['phase1_required']}")
    print("  Roadmap:")
    for p in result["roadmap"]:
        print(f"    Phase {p['phase']}: {p['title']} — {p['timeframe_weeks']} weeks")
    print(f"  Total roadmap weeks: {result['total_roadmap_weeks']}")
    print(f"  Headline metric: {result['headline_metric']}")

    print("\n  Recommended AI agents per phase:")
    for p in result["roadmap"]:
        agents = p.get("recommended_ai_agents", [])
        if not agents:
            print(f"    Phase {p['phase']}: (none recommended)")
            continue
        agent_labels = [
            f"{a['name']} ({'free tier' if a.get('free_tier') else a.get('estimated_monthly_cost_gbp', 'cost n/a')})"
            for a in agents
        ]
        print(f"    Phase {p['phase']}: {' + '.join(agent_labels)} — {p['title']}")

    summary = result.get("marketplace_summary", {})
    print(f"  Estimated monthly AI tool cost if all agents deployed: {summary.get('estimated_total_monthly_cost_gbp', 'n/a')}")
    print(f"  Free-tier agents available: {summary.get('free_tier_agents_count', 0)} of {summary.get('total_agents_recommended', 0)} recommended")
    print(f"  {summary.get('aiyyary_bridge_note', '')}")

    print(f"  Output saved to: {result['traceability']['output_file']}")
    print()


def print_comparison_table(results: list) -> None:
    print("=" * 84)
    print("AIYYARY AGENT 3 — RISK AND ROADMAP COMPARISON")
    print("=" * 84)
    header = (
        f"{'Company':<20s} {'Score':>6s} {'Tier':<15s} "
        f"{'HIGH':>5s} {'MED':>4s} {'LOW':>4s} {'Weeks':>6s} {'Headline metric':<40s}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        headline = r["headline_metric"] or "(none)"
        row = (
            f"{r['company_name']:<20s} {r['overall_score']:>6.2f} {r['readiness_tier']:<15s} "
            f"{r['high_severity_count']:>5d} {r['medium_severity_count']:>4d} {r['low_severity_count']:>4d} "
            f"{r['total_roadmap_weeks']:>6d} {headline:<40s}"
        )
        print(row)
    print("=" * 84)


def main() -> None:
    parser = argparse.ArgumentParser(description="AIYYARY Agent 3 — Live Demo: Risk and Roadmap Across Five Companies")
    parser.add_argument(
        "--company",
        help="Run only the named company (case-insensitive, exact match), e.g. --company \"Birdsong London\"",
    )
    args = parser.parse_args()

    companies = DEMO_COMPANIES
    if args.company:
        companies = [c for c in DEMO_COMPANIES if c["company_name"].lower() == args.company.lower()]
        if not companies:
            known = ", ".join(c["company_name"] for c in DEMO_COMPANIES)
            raise SystemExit(f"No demo company named '{args.company}'. Known companies: {known}")

    results = []
    print("=" * 78)
    print("AIYYARY Agent 3 — Live Demo: Risk and Roadmap Across Five Companies")
    print("=" * 78)

    for company in companies:
        print(f"\nRunning Agent 3 pipeline for: {company['company_name']} ...")
        try:
            result = run_company(company)
        except Exception as exc:
            print(f"[ERROR] {company['company_name']} failed unexpectedly: {exc}")
            continue
        if result is None:
            continue
        print_company_summary(result)
        results.append(result)

    if results:
        print_comparison_table(results)
    else:
        print("No companies completed successfully.")


if __name__ == "__main__":
    main()
