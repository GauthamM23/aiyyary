"""Test suite for Agent 3 (RiskTransformationAgent).

Mode 1 (default, always runs): deterministic severity tests. Pure Python,
no network connection, no API key required — exercises _assign_severity()
and _regulatory_override() directly against every branch of the taxonomy.

Mode 2 (--live flag): live pipeline test. Loads the most recent Agent 1
and Agent 2 output JSONs for Lucy & Yak from outputs/, runs the full
RiskTransformationAgent pipeline (3 LLM calls), and prints the risk
register and roadmap summary. Requires a Groq/Anthropic API key.

Usage:
    python test_agent3.py
    python test_agent3.py --live
"""

import argparse
import glob
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from agents.agent3_risk import RiskTransformationAgent

# ---------------------------------------------------------------------------
# Mode 1: deterministic severity tests
# ---------------------------------------------------------------------------

_agent = RiskTransformationAgent()
_categories_by_id = {c["id"]: c for c in _agent.taxonomy["categories"]}


def severity(cat_id: str, score: float) -> str:
    category = _categories_by_id[cat_id]
    pillar_scores = {category["pillar_mapped"]: {"final_score": score}}
    return _agent._assign_severity(category, pillar_scores, {})


def override(cat_id: str, privacy_found=None, consent_found=None, days_filing=None, base=None, score: float = 3.0) -> str:
    category = _categories_by_id[cat_id]
    pillar_scores = {category["pillar_mapped"]: {"final_score": score}}
    enrichment = {
        "privacy_policy_found": privacy_found,
        "cookie_consent_found": consent_found,
        "days_since_last_filing": days_filing,
    }

    if base is None:
        base = _agent._assign_severity(category, pillar_scores, enrichment)

    result, _reason = _agent._regulatory_override(base, enrichment)
    return result


def vendor_severity(tools: list, platform: str) -> str:
    category = _categories_by_id["vendor_risk"]
    enrichment = {"marketing_automation_tools": tools, "platform_detected": platform}
    return _agent._assign_severity(category, {}, enrichment)


def hallucination_severity(chat: bool, automation: bool, ai_score: float) -> str:
    category = _categories_by_id["hallucination_risk"]
    pillar_scores = {"ai_tool_experience": {"final_score": ai_score}}
    enrichment = {"chat_widget_found": chat, "marketing_automation_detected": automation}
    return _agent._assign_severity(category, pillar_scores, enrichment)


def ethical_severity(mentions_ai, gov_score: float) -> str:
    category = _categories_by_id["ethical_risk"]
    pillar_scores = {"governance_compliance": {"final_score": gov_score}}
    enrichment = {"privacy_policy_mentions_ai": mentions_ai}
    return _agent._assign_severity(category, pillar_scores, enrichment)


def _case(label, actual, expected):
    return {"label": label, "actual": actual, "expected": expected, "passed": actual == expected}


def run_mode_1() -> bool:
    print("=" * 78)
    print("MODE 1: Deterministic severity tests (offline, no API required)")
    print("=" * 78)

    cases = [
        # Data risk — score driven
        _case("data_risk HIGH", severity("data_risk", 2.0), "HIGH"),
        _case("data_risk MEDIUM", severity("data_risk", 5.0), "MEDIUM"),
        _case("data_risk LOW", severity("data_risk", 7.0), "LOW"),

        # Regulatory risk — score driven base, then override
        _case("regulatory_risk score-driven HIGH", severity("regulatory_risk", 3.0), "HIGH"),
        _case("regulatory_risk override privacy_found=False", override("regulatory_risk", privacy_found=False), "HIGH"),
        _case("regulatory_risk override consent_found=False", override("regulatory_risk", consent_found=False), "HIGH"),
        _case(
            "regulatory_risk override days_filing=400 base=LOW",
            override("regulatory_risk", days_filing=400, base="LOW"),
            "MEDIUM",
        ),
        _case(
            "regulatory_risk override privacy_found=None base=LOW",
            override("regulatory_risk", privacy_found=None, base="LOW"),
            "LOW",
        ),

        # Adoption risk — score driven
        _case("adoption_risk HIGH", severity("adoption_risk", 2.5), "HIGH"),
        _case("adoption_risk MEDIUM", severity("adoption_risk", 5.5), "MEDIUM"),
        _case("adoption_risk LOW", severity("adoption_risk", 8.0), "LOW"),

        # Vendor risk — enrichment driven
        _case("vendor_risk HIGH (single tool + Shopify)", vendor_severity(["Klaviyo"], "Shopify"), "HIGH"),
        _case("vendor_risk MEDIUM (two tools + Shopify)", vendor_severity(["Klaviyo", "Mailchimp"], "Shopify"), "MEDIUM"),
        _case("vendor_risk LOW (no tools + WooCommerce)", vendor_severity([], "WooCommerce"), "LOW"),

        # Hallucination risk — enrichment driven
        _case(
            "hallucination_risk HIGH (chat+automation, low ai score)",
            hallucination_severity(True, True, 3.0),
            "HIGH",
        ),
        _case(
            "hallucination_risk MEDIUM (automation only, mid ai score)",
            hallucination_severity(False, True, 5.0),
            "MEDIUM",
        ),
        _case(
            "hallucination_risk LOW (no chat/automation, high ai score)",
            hallucination_severity(False, False, 8.0),
            "LOW",
        ),

        # Ethical risk — enrichment driven
        _case("ethical_risk HIGH (no AI mention, low gov score)", ethical_severity(False, 3.0), "HIGH"),
        _case("ethical_risk MEDIUM (no AI mention, high gov score)", ethical_severity(False, 7.0), "MEDIUM"),
        _case("ethical_risk LOW (AI mentioned)", ethical_severity(True, 5.0), "LOW"),
    ]

    passed_count = 0
    for case in cases:
        status = "PASS" if case["passed"] else "FAIL"
        if case["passed"]:
            passed_count += 1
        else:
            print(f"  [FAIL DETAIL] {case['label']}: expected {case['expected']}, got {case['actual']}")

    print()
    print("═" * 40)
    print("AGENT 3 DETERMINISTIC TEST RESULTS")
    print("═" * 40)
    max_label_len = max(len(c["label"]) for c in cases)
    for case in cases:
        status = "PASS" if case["passed"] else "FAIL"
        print(f"{case['label']:<{max_label_len}s} : {status}")
    print("═" * 40)
    failed_count = len(cases) - passed_count
    print(f"Total: {passed_count} passed, {failed_count} failed")
    print("═" * 40)

    return failed_count == 0


# ---------------------------------------------------------------------------
# Mode 2: live pipeline test (Lucy & Yak only)
# ---------------------------------------------------------------------------


def _safe_company(company_name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(company_name or "unknown"))


def _find_latest_output(prefix: str, company_name: str) -> str:
    output_dir = os.getenv("OUTPUT_DIR", "./outputs")
    pattern = os.path.join(output_dir, f"{prefix}_{_safe_company(company_name)}_*.json")
    matches = glob.glob(pattern)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def run_mode_2() -> None:
    print("=" * 78)
    print("MODE 2: Live pipeline test — Lucy & Yak (3 LLM calls)")
    print("=" * 78)

    company_name = "Lucy & Yak"
    agent1_path = _find_latest_output("agent1", company_name)
    agent2_path = _find_latest_output("agent2", company_name)

    if agent1_path is None or agent2_path is None:
        print(
            f"[ERROR] Could not find both Agent 1 and Agent 2 output JSONs for '{company_name}' in outputs/. "
            "Run demo_five_companies.py and demo_agent2.py first."
        )
        sys.exit(1)

    print(f"Using Agent 1 output: {agent1_path}")
    print(f"Using Agent 2 output: {agent2_path}")

    with open(agent1_path, "r", encoding="utf-8") as f:
        agent1_output = json.load(f)
    with open(agent2_path, "r", encoding="utf-8") as f:
        agent2_output = json.load(f)

    profile = {
        "company_name": company_name,
        "business_name": company_name,
        "primary_challenge": (
            "Managing rapid growth in online sales while maintaining brand authenticity "
            "and sustainable supply chain transparency"
        ),
    }

    agent = RiskTransformationAgent()
    result = agent.run(agent1_output, agent2_output, profile=profile)

    print("\nRISK REGISTER:")
    for r in result["risk_register"]:
        print(f"  [{r['severity']:<6s}] {r['label']} ({r['category_id']}) — pillar={r['pillar']} score={r['pillar_score']}")
        print(f"           trigger: {r['trigger_reason']}")
        print(f"           mitigation: {r['mitigation']}")
        print(f"           regulatory_ref: {r['regulatory_ref']}")

    print(
        f"\nSeverity counts — HIGH: {result['high_severity_count']}  "
        f"MEDIUM: {result['medium_severity_count']}  LOW: {result['low_severity_count']}"
    )

    print("\nPHASE STRUCTURE:")
    print(f"  phase1_required: {result['phase_structure']['phase1_required']}")
    print(f"  phase1_targets: {result['phase_structure']['phase1_targets']}")
    print(f"  phase2_deploy_built: {[o['id'] for o in result['phase_structure']['phase2_deploy_built']]}")
    print(f"  phase3_deploy_pending: {[o['id'] for o in result['phase_structure']['phase3_deploy_pending']]}")
    print(f"  phase4_deferred: {[o['id'] for o in result['phase_structure']['phase4_deferred']]}")

    print("\nROADMAP:")
    for p in result["roadmap"]:
        print(f"  Phase {p['phase']}: {p['title']} ({p['timeframe_weeks']} weeks)")
        print(f"    {p['description']}")
        print(f"    Deliverables: {p['deliverables']}")
        print(f"    Human oversight required: {p['human_oversight_required']}")
        print(f"    Activates executor: {p['activates_executor']}")

    print(f"\nTotal roadmap weeks: {result['total_roadmap_weeks']}")
    print(f"\nROI summary: {result['roi_summary']}")
    print(f"Headline metric: {result['headline_metric']}")
    print(f"\nFull output saved to: {result['traceability']['output_file']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIYYARY Agent 3 test suite")
    parser.add_argument("--live", action="store_true", help="Run Mode 2 (live pipeline test)")
    args = parser.parse_args()

    mode_1_passed = run_mode_1()

    if args.live:
        run_mode_2()

    if not mode_1_passed:
        sys.exit(1)
