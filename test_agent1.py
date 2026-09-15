"""Test suite for Agent 1 (CapabilityAssessmentAgent).

Mode 1 (default, always runs): deterministic Stream A scoring tests.
Pure Python, no network connection or API keys required — exercises
compute_stream_a() and readiness_tier() directly.

Mode 2 (--enrich flag): live enrichment test. Runs EnrichmentAgent.run()
against a real URL/business name and prints the report. Requires internet
but no LLM API key (the three Stream C inference calls inside enrichment
are only triggered on specific conflict conditions, and any LLM failure
is caught and reported inline rather than raised).

Usage:
    python test_agent1.py
    python test_agent1.py --enrich --url https://example.co.uk --name "Example Boutique" --location "London"
"""

import argparse
import json
import sys

from agents.agent1_capability import PILLARS, compute_stream_a, readiness_tier


# ---------------------------------------------------------------------------
# Mode 1: deterministic Stream A tests
# ---------------------------------------------------------------------------


def make_instrument(pillar_scores: dict) -> dict:
    """Build a 6-pillar x 4-question instrument dict from per-pillar raw score lists."""
    instrument = {}
    for pillar, scores in pillar_scores.items():
        assert len(scores) == 4, f"{pillar} must have exactly 4 scores"
        instrument[pillar] = [
            {"question": f"{pillar} question {i + 1}", "answer": f"answer with score {score}", "score": score}
            for i, score in enumerate(scores)
        ]
    return instrument


PROFILES = [
    {
        "name": "Very Low Maturity (Birdsong London)",
        "company_name": "Birdsong London",
        "scores": {
            "data_foundations": [1, 1, 2, 1],
            "process_digitisation": [2, 1, 1, 2],
            "technology_infrastructure": [1, 2, 2, 1],
            "staff_digital_literacy": [2, 2, 1, 1],
            "governance_compliance": [1, 1, 1, 2],
            "ai_tool_experience": [2, 1, 2, 1],
        },
        # Hand-calculated via scaled = ((mean - 1) / 8) * 9 + 1, rounded to 2dp
        "expected_pillar_scores": {
            "data_foundations": 1.28,
            "process_digitisation": 1.56,
            "technology_infrastructure": 1.56,
            "staff_digital_literacy": 1.56,
            "governance_compliance": 1.28,
            "ai_tool_experience": 1.56,
        },
        "expected_overall_range": (1.1, 2.0),
        "expected_tier": "Early-Stage",
    },
    {
        "name": "Low-Mid Maturity (Gudrun Sjödén UK)",
        "company_name": "Gudrun Sjödén UK",
        "scores": {
            "data_foundations": [5, 4, 4, 2],
            "process_digitisation": [4, 2, 1, 2],
            "technology_infrastructure": [5, 3, 7, 4],
            "staff_digital_literacy": [4, 4, 5, 2],
            "governance_compliance": [4, 4, 1, 2],
            "ai_tool_experience": [2, 4, 1, 4],
        },
        "expected_pillar_scores": {
            "data_foundations": 4.09,
            "process_digitisation": 2.41,
            "technology_infrastructure": 5.22,
            "staff_digital_literacy": 4.09,
            "governance_compliance": 2.97,
            "ai_tool_experience": 2.97,
        },
        "expected_overall_range": None,  # exact overall asserted below instead of a range
        "expected_tier": "Early-Stage",
    },
    {
        "name": "Mid Maturity (Baukjen)",
        "company_name": "Baukjen",
        "scores": {
            "data_foundations": [4, 5, 5, 6],
            "process_digitisation": [5, 5, 4, 6],
            "technology_infrastructure": [5, 6, 5, 5],
            "staff_digital_literacy": [4, 5, 5, 4],
            "governance_compliance": [5, 6, 5, 6],
            "ai_tool_experience": [4, 5, 4, 5],
        },
        "expected_pillar_scores": {
            "data_foundations": 5.5,
            "process_digitisation": 5.5,
            "technology_infrastructure": 5.78,
            "staff_digital_literacy": 4.94,
            "governance_compliance": 6.06,
            "ai_tool_experience": 4.94,
        },
        "expected_overall_range": (5.0, 6.5),
        "expected_tier": "Mid-Readiness",
    },
    {
        "name": "Mid-High Maturity (Nobody's Child)",
        "company_name": "Nobody's Child",
        "scores": {
            "data_foundations": [6, 7, 6, 7],
            "process_digitisation": [6, 7, 7, 6],
            "technology_infrastructure": [7, 6, 6, 7],
            "staff_digital_literacy": [5, 6, 6, 5],
            "governance_compliance": [7, 6, 7, 6],
            "ai_tool_experience": [6, 5, 6, 6],
        },
        "expected_pillar_scores": {
            "data_foundations": 7.19,
            "process_digitisation": 7.19,
            "technology_infrastructure": 7.19,
            "staff_digital_literacy": 6.06,
            "governance_compliance": 7.19,
            "ai_tool_experience": 6.34,
        },
        "expected_overall_range": (6.5, 7.49),
        "expected_tier": "Mid-Readiness",
    },
    {
        "name": "High Maturity (Lucy & Yak)",
        "company_name": "Lucy & Yak",
        "scores": {
            "data_foundations": [8, 9, 8, 9],
            "process_digitisation": [8, 9, 9, 8],
            "technology_infrastructure": [9, 8, 9, 8],
            "staff_digital_literacy": [8, 9, 8, 9],
            "governance_compliance": [9, 8, 9, 8],
            "ai_tool_experience": [8, 9, 8, 9],
        },
        "expected_pillar_scores": {
            "data_foundations": 9.44,
            "process_digitisation": 9.44,
            "technology_infrastructure": 9.44,
            "staff_digital_literacy": 9.44,
            "governance_compliance": 9.44,
            "ai_tool_experience": 9.44,
        },
        "expected_overall_range": (7.5, 10.0),
        "expected_tier": "High-Readiness",
    },
]


def run_mode_1() -> bool:
    print("=" * 78)
    print("MODE 1: Deterministic Stream A scoring tests (offline, no API required)")
    print("=" * 78)

    all_passed = True
    table_rows = []

    for profile in PROFILES:
        instrument = make_instrument(profile["scores"])
        pillar_scores = compute_stream_a(instrument)
        overall = round(sum(pillar_scores[p] for p in PILLARS) / len(PILLARS), 2)
        tier = readiness_tier(overall)

        failure_reasons = []

        for pillar in PILLARS:
            expected = profile["expected_pillar_scores"][pillar]
            actual = pillar_scores[pillar]
            if abs(actual - expected) > 0.005:
                failure_reasons.append(f"{pillar}: got {actual}, expected {expected}")

        expected_overall_exact = round(
            sum(profile["expected_pillar_scores"][p] for p in PILLARS) / len(PILLARS), 2
        )
        if abs(overall - expected_overall_exact) > 0.01:
            failure_reasons.append(f"overall: got {overall}, expected {expected_overall_exact}")

        if profile["expected_overall_range"] is not None:
            low, high = profile["expected_overall_range"]
            if not (low <= overall <= high):
                failure_reasons.append(f"overall {overall} outside expected range [{low}, {high}]")

        if tier != profile["expected_tier"]:
            failure_reasons.append(f"tier: got {tier}, expected {profile['expected_tier']}")

        passed = not failure_reasons
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"

        print(f"\n[{status}] {profile['name']} | Overall: {overall:.2f} | Tier: {tier}")
        for pillar in PILLARS:
            print(f"    {pillar:30s}: {pillar_scores[pillar]:.2f}")
        if failure_reasons:
            for reason in failure_reasons:
                print(f"    FAILURE: {reason}")

        table_rows.append(
            {
                "profile": profile["name"],
                "company_name": profile["company_name"],
                "pillar_scores": pillar_scores,
                "overall": overall,
                "tier": tier,
                "status": status,
            }
        )

    print("\n" + "=" * 78)
    print("STREAM A SCORES TABLE")
    print("=" * 78)
    header = f"{'Profile':38s} " + " ".join(f"{p[:10]:>11s}" for p in PILLARS) + f" {'Overall':>8s} {'Tier':>15s}"
    print(header)
    for row in table_rows:
        line = f"{row['profile']:38s} "
        line += " ".join(f"{row['pillar_scores'][p]:>11.2f}" for p in PILLARS)
        line += f" {row['overall']:>8.2f} {row['tier']:>15s}"
        print(line)

    print("\n" + "=" * 78)
    print("ALL PROFILES PASSED" if all_passed else "ONE OR MORE PROFILES FAILED")
    print("=" * 78)
    return all_passed


# ---------------------------------------------------------------------------
# Mode 2: live enrichment test
# ---------------------------------------------------------------------------


def run_mode_2(url: str, name: str, location: str) -> None:
    from agents.enrichment import EnrichmentAgent

    print("=" * 78)
    print("MODE 2: Live enrichment test (requires internet, no LLM API key needed)")
    print("=" * 78)
    print(f"Business name: {name}")
    print(f"URL:           {url}")
    print(f"Location:      {location}\n")

    agent = EnrichmentAgent()
    profile = {"business_name": name, "website": url, "location": location}
    report = agent.run(profile)

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIYYARY Agent 1 test suite")
    parser.add_argument("--enrich", action="store_true", help="Run Mode 2 (live enrichment test)")
    parser.add_argument("--url", default=None, help="Business website URL for Mode 2")
    parser.add_argument("--name", default=None, help="Business name for Mode 2")
    parser.add_argument("--location", default="United Kingdom", help="Business location for Mode 2")
    args = parser.parse_args()

    mode_1_passed = run_mode_1()

    if args.enrich:
        if not args.url or not args.name:
            print("\n--enrich requires both --url and --name", file=sys.stderr)
            sys.exit(1)
        run_mode_2(args.url, args.name, args.location)

    if not mode_1_passed:
        sys.exit(1)
