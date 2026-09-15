"""Demo runner: the full Agent 2 pipeline against the five companies from demo_five_companies.py.

For each company this script:
  1. Loads the most recent Agent 1 output JSON for that company from outputs/.
     If none exists, prints a clear message and skips the company (does not crash).
  2. Runs the full Agent 2 pipeline (semantic retrieval, deterministic filter,
     LLM justification, and executor dispatch for the single top-ranked
     opportunity only, to control LLM spend).
  3. Prints a per-company summary.
  4. Prints a final comparison table across all companies that ran successfully.
"""

import glob
import json
import os

from agents.agent2_opportunity import OpportunityResearchAgent
from demo_five_companies import DEMO_COMPANIES


def _safe_company(company_name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(company_name or "unknown"))


# Agent 2 never runs Agent 1 itself in this demo — it reuses whatever Agent 1
# already produced for this company. This finds the most recent one on disk
# (by file mtime, filtered by filename prefix) so this script can be re-run
# any time after demo_five_companies.py without re-scoring capability.
def _find_latest_agent1_output(company_name: str) -> str:
    output_dir = os.getenv("OUTPUT_DIR", "./outputs")
    pattern = os.path.join(output_dir, f"agent1_{_safe_company(company_name)}_*.json")
    matches = glob.glob(pattern)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


# Per-company driver for this demo: loads the cached Agent 1 output, rebuilds
# the same profile dict Agent 1 was run with, and runs the real
# OpportunityResearchAgent pipeline (semantic retrieval, deterministic
# prerequisite filter, LLM justification, and executor dispatch capped to the
# single top-ranked opportunity to control LLM spend across five companies).
# This is the function whose output feeds directly into agent2_*.json and,
# later, into demo_agent3.py's own _find_latest_output("agent2", ...).
def run_company(company: dict) -> dict:
    latest_file = _find_latest_agent1_output(company["company_name"])
    if latest_file is None:
        print(f"[SKIP] No Agent 1 output found for {company['company_name']} in outputs/. Run demo_five_companies.py first.")
        return None

    with open(latest_file, "r", encoding="utf-8") as f:
        agent1_output = json.load(f)

    profile = {
        "business_name": company["company_name"],
        "company_name": company["company_name"],
        "website": company["website"],
        "location": company["location"],
        "instagram_handle": company["instagram_handle"],
        "primary_challenge": company["primary_challenge"],
        "industry": "Fashion and Apparel",
    }

    print(f"Using Agent 1 output: {latest_file}")
    agent = OpportunityResearchAgent()
    result = agent.run(profile, agent1_output, execute_top_n=1)
    return result


def print_company_summary(result: dict) -> None:
    print("-" * 78)
    print(result["company_name"])
    print("-" * 78)
    print(f"  Overall score: {result['overall_score']:.2f}   Tier: {result['readiness_tier']}")

    if result["recommended"]:
        top = result["recommended"][0]
        print(f"  Top recommended: {top['title']} ({top['id']})")
        print(f"  Executor mode:   {top['executor_mode']}")
        executor_result = top.get("executor_result")
        if executor_result:
            print(f"  Executor status: {executor_result['status']}")
            deliverables = executor_result.get("deliverables") or []
            first_deliverable = deliverables[0]["name"] if deliverables else "(none)"
            print(f"  First deliverable: {first_deliverable}")
        else:
            print("  Executor status: not dispatched")
            print("  First deliverable: (none)")
    else:
        print("  Top recommended: (none survived the filter)")

    print(f"  Quick win: {result['quick_win_summary']}")
    print(f"  Output saved to: {result['traceability']['output_file']}")
    print()


def print_comparison_table(results: list) -> None:
    print("=" * 118)
    print("COMPARISON ACROSS ALL COMPANIES (AGENT 2)")
    print("=" * 118)
    header = (
        f"{'Company':<22s} {'Overall':>8s} {'Tier':>15s} "
        f"{'Survivors':>10s} {'Excluded':>10s} {'Top Opportunity':>34s} {'Exec Status':>12s}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        top_str = r["recommended"][0]["title"] if r["recommended"] else "(none)"
        exec_status = "n/a"
        if r["recommended"] and r["recommended"][0].get("executor_result"):
            exec_status = r["recommended"][0]["executor_result"]["status"]
        row = (
            f"{r['company_name']:<22s} {r['overall_score']:>8.2f} {r['readiness_tier']:>15s} "
            f"{r['survivors_count']:>10d} {r['excluded_count']:>10d} {top_str:>34s} {exec_status:>12s}"
        )
        print(row)
    print("=" * 118)


def main() -> None:
    results = []
    print("=" * 78)
    print("AIYYARY Agent 2 — Live Demo: Opportunity Research Across Five Companies")
    print("=" * 78)

    for company in DEMO_COMPANIES:
        print(f"\nRunning Agent 2 pipeline for: {company['company_name']} ...")
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
