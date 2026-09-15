"""
Test suite for Agent 1 (CapabilityAssessmentAgent) scoring logic.

Runs entirely offline: only compute_scores() and readiness_tier() are exercised,
both of which are pure deterministic Python with no API calls.
"""

from agents.agent1_capability import PILLAR_KEYS, CapabilityAssessmentAgent


def make_instrument(pillar_scores: dict[str, list[int]]) -> dict:
    """Build a 6-pillar x 4-question instrument dict from per-pillar raw score lists."""
    instrument = {}
    for pillar, scores in pillar_scores.items():
        assert len(scores) == 4, f"{pillar} must have exactly 4 scores"
        instrument[pillar] = [
            {
                "question": f"{pillar} question {i + 1}",
                "answer": f"answer with score {score}",
                "score": score,
            }
            for i, score in enumerate(scores)
        ]
    return instrument


def manual_expected_overall(pillar_scores: dict[str, list[int]]) -> tuple[dict, float]:
    """Independently re-derive pillar scaled scores and overall score for cross-check."""
    scaled = {}
    for pillar, scores in pillar_scores.items():
        raw_mean = sum(scores) / len(scores)
        scaled_score = ((raw_mean - 1) / 8) * 9 + 1
        scaled[pillar] = round(scaled_score, 2)
    overall = round(sum(scaled.values()) / len(scaled), 2)
    return scaled, overall


def expected_tier(overall: float) -> str:
    if overall >= 7.5:
        return "High-Readiness"
    elif overall >= 5.0:
        return "Mid-Readiness"
    else:
        return "Early-Stage"


PROFILES = [
    {
        "name": "Profile 1 - Very Low Maturity (Frayed Thread Basics)",
        "company_name": "Frayed Thread Basics",
        "scores": {
            "data_foundations": [1, 1, 2, 1],
            "process_digitisation": [1, 2, 1, 1],
            "technology_infrastructure": [2, 1, 1, 2],
            "staff_digital_literacy": [1, 1, 1, 2],
            "governance_compliance": [2, 2, 1, 1],
            "ai_tool_experience": [1, 1, 1, 1],
        },
    },
    {
        "name": "Profile 2 - Low-Mid Maturity (Thread & Co)",
        "company_name": "Thread & Co",
        "scores": {
            "data_foundations": [2, 3, 3, 4],
            "process_digitisation": [3, 4, 2, 3],
            "technology_infrastructure": [4, 3, 3, 4],
            "staff_digital_literacy": [2, 2, 3, 3],
            "governance_compliance": [3, 4, 4, 3],
            "ai_tool_experience": [2, 2, 3, 2],
        },
    },
    {
        "name": "Profile 3 - Mid Maturity (Meridian Apparel)",
        "company_name": "Meridian Apparel",
        "scores": {
            "data_foundations": [4, 5, 5, 6],
            "process_digitisation": [5, 5, 4, 6],
            "technology_infrastructure": [5, 6, 5, 5],
            "staff_digital_literacy": [4, 5, 5, 4],
            "governance_compliance": [5, 6, 5, 6],
            "ai_tool_experience": [4, 5, 4, 5],
        },
    },
    {
        "name": "Profile 4 - Mid-High Maturity (Northbank Fashion Group)",
        "company_name": "Northbank Fashion Group",
        "scores": {
            "data_foundations": [6, 7, 6, 7],
            "process_digitisation": [6, 7, 7, 6],
            "technology_infrastructure": [7, 6, 6, 7],
            "staff_digital_literacy": [5, 6, 6, 5],
            "governance_compliance": [7, 6, 7, 6],
            "ai_tool_experience": [6, 5, 6, 6],
        },
    },
    {
        "name": "Profile 5 - High Maturity (Vantage Digital Retail)",
        "company_name": "Vantage Digital Retail",
        "scores": {
            "data_foundations": [8, 9, 8, 9],
            "process_digitisation": [7, 8, 8, 9],
            "technology_infrastructure": [8, 8, 9, 8],
            "staff_digital_literacy": [7, 8, 7, 8],
            "governance_compliance": [8, 9, 9, 8],
            "ai_tool_experience": [7, 8, 8, 7],
        },
    },
]


def run_all_tests() -> bool:
    agent = CapabilityAssessmentAgent()
    all_passed = True

    for profile in PROFILES:
        name = profile["name"]
        scores = profile["scores"]
        instrument = make_instrument(scores)

        pillar_scores, overall_score = agent.compute_scores(instrument)
        tier = agent.readiness_tier(overall_score)

        expected_pillar_scores, expected_overall = manual_expected_overall(scores)
        expected_tier_value = expected_tier(expected_overall)

        print(f"\n{'=' * 70}")
        print(name)
        print(f"{'=' * 70}")
        for pillar in PILLAR_KEYS:
            print(f"  {pillar:30s}: {pillar_scores[pillar]:.2f}")
        print(f"  {'OVERALL':30s}: {overall_score:.2f}")
        print(f"  {'TIER':30s}: {tier}")

        passed = True
        failure_reasons = []

        if abs(overall_score - expected_overall) > 0.005:
            passed = False
            failure_reasons.append(
                f"overall score mismatch: got {overall_score}, expected {expected_overall}"
            )

        for pillar in PILLAR_KEYS:
            if abs(pillar_scores[pillar] - expected_pillar_scores[pillar]) > 0.005:
                passed = False
                failure_reasons.append(
                    f"{pillar} mismatch: got {pillar_scores[pillar]}, "
                    f"expected {expected_pillar_scores[pillar]}"
                )

        if tier != expected_tier_value:
            passed = False
            failure_reasons.append(
                f"tier mismatch: got {tier}, expected {expected_tier_value}"
            )

        try:
            assert set(pillar_scores.keys()) == set(PILLAR_KEYS)
        except AssertionError:
            passed = False
            failure_reasons.append("pillar_scores keys do not match PILLAR_KEYS exactly")

        status = "PASS" if passed else "FAIL"
        print(
            f"  RESULT: {status} | {name} | overall={overall_score:.2f} | tier={tier}"
        )
        if not passed:
            for reason in failure_reasons:
                print(f"    - {reason}")
            all_passed = False

    return all_passed


if __name__ == "__main__":
    print("Running Agent 1 capability scoring tests (offline, no API key required)...")
    success = run_all_tests()

    print(f"\n{'=' * 70}")
    if success:
        print("ALL PROFILES PASSED")
    else:
        print("ONE OR MORE PROFILES FAILED")
    print(f"{'=' * 70}")

    if not success:
        raise SystemExit(1)
