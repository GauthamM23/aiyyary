"""Test suite for Agent 2 (OpportunityResearchAgent).

Mode 1 (default, always runs): deterministic prerequisite-filter tests. Pure
Python, no network connection, no API key, and no ChromaDB required —
exercises filter_by_prerequisites() directly against the real knowledge base.

Mode 2 (--live flag): live pipeline test. Runs the full OpportunityResearchAgent
pipeline (ChromaDB semantic retrieval + LLM justification call) against a
single constructed profile, with executor dispatch suppressed (execute_top_n=0)
to keep it fast and cheap. Requires a Groq API key.

Usage:
    python test_agent2.py
    python test_agent2.py --live
"""

import argparse
import json
import os
import sys

from agents.agent2_opportunity import filter_by_prerequisites, load_knowledge_base

# ---------------------------------------------------------------------------
# Mode 1: deterministic prerequisite-filter tests
# ---------------------------------------------------------------------------


def _make_agent1_output(company_name: str, pillar_final_scores: dict) -> dict:
    pillar_scores = {p: {"final_score": v} for p, v in pillar_final_scores.items()}
    overall = round(sum(pillar_final_scores.values()) / len(pillar_final_scores), 2)
    if overall >= 7.5:
        tier = "High-Readiness"
    elif overall >= 5.0:
        tier = "Mid-Readiness"
    else:
        tier = "Early-Stage"
    return {
        "agent": "Agent1_CapabilityAssessment",
        "schema_version": "2.0",
        "company_name": company_name,
        "pillar_scores": pillar_scores,
        "overall_score": overall,
        "readiness_tier": tier,
    }


PROFILES = [
    {
        "name": "Very Low Maturity",
        "agent1_output": _make_agent1_output(
            "Very Low Maturity Co",
            {
                "data_foundations": 1.5,
                "process_digitisation": 1.5,
                "technology_infrastructure": 1.5,
                "staff_digital_literacy": 1.5,
                "governance_compliance": 1.5,
                "ai_tool_experience": 1.5,
            },
        ),
        "check": lambda survivors, excluded: (
            {o["id"] for o in survivors} == {"uc_001"},
            f"expected only uc_001 to survive, got {sorted(o['id'] for o in survivors)}",
        ),
    },
    {
        "name": "Low-Mid Maturity",
        "agent1_output": _make_agent1_output(
            "Low-Mid Maturity Co",
            {
                "data_foundations": 4.09,
                "process_digitisation": 2.41,
                "technology_infrastructure": 5.22,
                "staff_digital_literacy": 4.09,
                "governance_compliance": 2.97,
                "ai_tool_experience": 2.97,
            },
        ),
        "check": lambda survivors, excluded: (
            {"uc_001", "uc_002"}.issubset({o["id"] for o in survivors}),
            f"expected uc_001 and uc_002 to survive, got {sorted(o['id'] for o in survivors)}",
        ),
    },
    {
        "name": "Mid Maturity",
        "agent1_output": _make_agent1_output(
            "Mid Maturity Co",
            {
                "data_foundations": 5.5,
                "process_digitisation": 5.5,
                "technology_infrastructure": 5.78,
                "staff_digital_literacy": 4.94,
                "governance_compliance": 6.06,
                "ai_tool_experience": 4.94,
            },
        ),
        "check": lambda survivors, excluded: (
            len(survivors) >= 5
            and "uc_009" not in {o["id"] for o in survivors}
            and "uc_010" not in {o["id"] for o in survivors},
            f"expected >=5 survivors with uc_009/uc_010 excluded, got {sorted(o['id'] for o in survivors)}",
        ),
    },
    {
        "name": "Mid-High Maturity",
        "agent1_output": _make_agent1_output(
            "Mid-High Maturity Co",
            {
                "data_foundations": 7.3,
                "process_digitisation": 7.2,
                "technology_infrastructure": 6.8,
                "staff_digital_literacy": 6.7,
                "governance_compliance": 6.6,
                "ai_tool_experience": 6.7,
            },
        ),
        "check": lambda survivors, excluded: (
            len(survivors) >= 7 and {o["id"] for o in excluded} == {"uc_010"},
            f"expected >=7 survivors with only uc_010 excluded, got excluded={sorted(o['id'] for o in excluded)}",
        ),
    },
    {
        "name": "High Maturity",
        "agent1_output": _make_agent1_output(
            "High Maturity Co",
            {
                "data_foundations": 9.44,
                "process_digitisation": 9.44,
                "technology_infrastructure": 9.44,
                "staff_digital_literacy": 9.44,
                "governance_compliance": 9.44,
                "ai_tool_experience": 9.44,
            },
        ),
        "check": lambda survivors, excluded: (
            len(survivors) == 10,
            f"expected all 10 to survive, got {len(survivors)}",
        ),
    },
]


def run_mode_1() -> bool:
    print("=" * 78)
    print("MODE 1: Deterministic prerequisite filter tests (offline, no API required)")
    print("=" * 78)

    knowledge_base = load_knowledge_base()
    all_passed = True
    table_rows = []

    for profile in PROFILES:
        agent1_output = profile["agent1_output"]
        survivors, excluded = filter_by_prerequisites(knowledge_base, agent1_output)
        passed, message = profile["check"](survivors, excluded)
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"

        overall = agent1_output["overall_score"]
        tier = agent1_output["readiness_tier"]
        print(f"\n[{status}] {profile['name']} | Overall: {overall:.2f} | Tier: {tier}")
        print(f"    Survivors ({len(survivors)}): {sorted(o['id'] for o in survivors)}")
        print(f"    Excluded  ({len(excluded)}): {sorted(o['id'] for o in excluded)}")
        if not passed:
            print(f"    FAILURE: {message}")

        table_rows.append({
            "profile": profile["name"],
            "overall": overall,
            "tier": tier,
            "survivors": sorted(o["id"] for o in survivors),
            "excluded": sorted(o["id"] for o in excluded),
            "status": status,
        })

    print("\n" + "=" * 78)
    print("FILTER RESULTS TABLE")
    print("=" * 78)
    for row in table_rows:
        print(f"\n{row['profile']} (overall={row['overall']:.2f}, tier={row['tier']}, status={row['status']})")
        print(f"  Survive : {row['survivors']}")
        print(f"  Exclude : {row['excluded']}")

    print("\n" + "=" * 78)
    print("ALL PROFILES PASSED" if all_passed else "ONE OR MORE PROFILES FAILED")
    print("=" * 78)
    return all_passed


# ---------------------------------------------------------------------------
# Mode 2: live pipeline test
# ---------------------------------------------------------------------------


def run_mode_2() -> None:
    if os.getenv("GROQ_API_KEY"):
        os.environ["LLM_MODE"] = "groq"

    from agents.agent2_opportunity import OpportunityResearchAgent

    print("=" * 78)
    print("MODE 2: Live pipeline test (ChromaDB retrieval + LLM justification, no executor dispatch)")
    print("=" * 78)

    profile = {
        "company_name": "Test Fashion Co",
        "business_name": "Test Fashion Co",
        "website": "https://example-test-fashion.co.uk",
        "location": "London, UK",
        "primary_challenge": "Reducing returns and improving repeat purchase rate through better size guidance",
        "industry": "Fashion and Apparel",
    }
    agent1_output = {
        "agent": "Agent1_CapabilityAssessment",
        "schema_version": "2.0",
        "company_name": "Test Fashion Co",
        "pillar_scores": {
            "data_foundations": {"final_score": 5.5},
            "process_digitisation": {"final_score": 5.5},
            "technology_infrastructure": {"final_score": 5.78},
            "staff_digital_literacy": {"final_score": 4.94},
            "governance_compliance": {"final_score": 6.06},
            "ai_tool_experience": {"final_score": 4.94},
        },
        "overall_score": 5.45,
        "readiness_tier": "Mid-Readiness",
    }

    agent = OpportunityResearchAgent()
    result = agent.run(profile, agent1_output, execute_top_n=0)

    print("\nRECOMMENDED:")
    for r in result["recommended"]:
        print(f"  [{r['rank']}] {r['title']} ({r['id']}) — {r['justification']}")

    print("\nDEFERRED:")
    for d in result["deferred"]:
        print(f"  {d['title']} ({d['id']}) — {d['reason']}")

    print(f"\nQuick win: {result['quick_win_summary']}")
    print(f"\nFull output saved to: {result['traceability']['output_file']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIYYARY Agent 2 test suite")
    parser.add_argument("--live", action="store_true", help="Run Mode 2 (live pipeline test)")
    args = parser.parse_args()

    mode_1_passed = run_mode_1()

    if args.live:
        run_mode_2()

    if not mode_1_passed:
        sys.exit(1)
