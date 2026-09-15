"""Cross-validation: researcher-constructed scores (test_agent1.py) vs live
enrichment scores (demo_five_companies.py).

Compares the five hand-constructed Stream-A-only profiles the researcher
built before any live enrichment ran against the actual Stream A + B + C
synthesis measured by the live pipeline, quantifies divergence per pillar,
generates academic findings for the dissertation results chapter, and
confirms exactly which scores/tiers are being handed to Agent 2.

This script is read-only with respect to every other file in the project —
it only writes to outputs/cross_validation_report.json and
outputs/cross_validation_table.md.

Usage:
    python cross_validate.py                # full run incl. LLM findings
    python cross_validate.py --no-llm        # skip LLM findings generation
    python cross_validate.py --summary-only  # print only summary + Agent 2 confirmation
    python cross_validate.py --export-table  # write only the markdown table, no terminal report
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

from agents.agent1_capability import PILLARS

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "outputs")
REPORT_JSON_PATH = os.path.join(OUTPUTS_DIR, "cross_validation_report.json")
TABLE_MD_PATH = os.path.join(OUTPUTS_DIR, "cross_validation_table.md")

COMPANIES = [
    "Birdsong London",
    "Gudrun Sjödén UK",
    "Baukjen",
    "Nobody's Child",
    "Lucy & Yak",
]

HEAVY_SEP = "═" * 79
LIGHT_SEP = "─" * 53

# ---------------------------------------------------------------------------
# PART 1: constructed scores store
#
# These are the exact Stream-A-only expected_pillar_scores and hand-calculated
# overall (mean of the six pillars, rounded to 2dp) from test_agent1.py's
# PROFILES list, representing the researcher's prior expectation before any
# live enrichment ran.
# ---------------------------------------------------------------------------

CONSTRUCTED_SCORES = {
    "Birdsong London": {
        "source": "researcher_constructed",
        "maturity_label": "Very Low Maturity",
        "pillar_scores": {
            "data_foundations": 1.28,
            "process_digitisation": 1.56,
            "technology_infrastructure": 1.56,
            "staff_digital_literacy": 1.56,
            "governance_compliance": 1.28,
            "ai_tool_experience": 1.56,
        },
        "overall_score": 1.47,
    },
    "Gudrun Sjödén UK": {
        "source": "researcher_constructed",
        "maturity_label": "Low-Mid Maturity",
        "pillar_scores": {
            "data_foundations": 4.09,
            "process_digitisation": 2.41,
            "technology_infrastructure": 5.22,
            "staff_digital_literacy": 4.09,
            "governance_compliance": 2.97,
            "ai_tool_experience": 2.97,
        },
        "overall_score": 3.62,
    },
    "Baukjen": {
        "source": "researcher_constructed",
        "maturity_label": "Mid Maturity",
        "pillar_scores": {
            "data_foundations": 5.5,
            "process_digitisation": 5.5,
            "technology_infrastructure": 5.78,
            "staff_digital_literacy": 4.94,
            "governance_compliance": 6.06,
            "ai_tool_experience": 4.94,
        },
        "overall_score": 5.45,
    },
    "Nobody's Child": {
        "source": "researcher_constructed",
        "maturity_label": "Mid-High Maturity",
        "pillar_scores": {
            "data_foundations": 7.19,
            "process_digitisation": 7.19,
            "technology_infrastructure": 7.19,
            "staff_digital_literacy": 6.06,
            "governance_compliance": 7.19,
            "ai_tool_experience": 6.34,
        },
        "overall_score": 6.86,
    },
    "Lucy & Yak": {
        "source": "researcher_constructed",
        "maturity_label": "High Maturity",
        "pillar_scores": {
            "data_foundations": 9.44,
            "process_digitisation": 9.44,
            "technology_infrastructure": 9.44,
            "staff_digital_literacy": 9.44,
            "governance_compliance": 9.44,
            "ai_tool_experience": 9.44,
        },
        "overall_score": 9.44,
    },
}


# ---------------------------------------------------------------------------
# PART 2: live scores loader
# ---------------------------------------------------------------------------


def load_live_scores(outputs_dir: str) -> dict:
    """Load the most recent live Agent 1 result per company from outputs_dir.

    Scans every *.json file in outputs_dir, keeps the ones whose content is
    an Agent 1 capability assessment, and for each company retains only the
    most recent run by assessment_timestamp. Companies with no matching file
    are returned as None with a warning printed to terminal.
    """
    candidates_by_company = {}

    for path in glob.glob(os.path.join(outputs_dir, "*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        if data.get("agent") != "Agent1_CapabilityAssessment":
            continue

        company_name = data.get("company_name")
        timestamp = data.get("assessment_timestamp")
        if not company_name or not timestamp:
            continue

        existing = candidates_by_company.get(company_name)
        if existing is None or timestamp > existing["assessment_timestamp"]:
            candidates_by_company[company_name] = {
                "assessment_timestamp": timestamp,
                "data": data,
            }

    live_scores = {}
    for company_name in COMPANIES:
        entry = candidates_by_company.get(company_name)
        if entry is None:
            print(f"[WARNING] No live Agent 1 output found for '{company_name}' in {outputs_dir} — skipping.")
            live_scores[company_name] = None
            continue

        data = entry["data"]
        pillar_scores = {pillar: data["pillar_scores"][pillar]["final_score"] for pillar in PILLARS}
        live_scores[company_name] = {
            "source": "live_enrichment",
            "pillar_scores": pillar_scores,
            "overall_score": data["overall_score"],
            "readiness_tier": data["readiness_tier"],
            "assessment_timestamp": data["assessment_timestamp"],
        }

    return live_scores


# ---------------------------------------------------------------------------
# PART 3: divergence calculator
# ---------------------------------------------------------------------------


def _severity(abs_divergence: float) -> str:
    if abs_divergence >= 2.0:
        return "HIGH"
    if abs_divergence >= 1.0:
        return "MEDIUM"
    return "LOW"


def calculate_divergence(constructed: dict, live: dict) -> dict:
    """Compute per-pillar and overall divergence for each company with live data."""
    report = {}

    for company_name, live_entry in live.items():
        if live_entry is None:
            continue
        constructed_entry = constructed[company_name]

        pillar_divergence = {}
        low_count = 0
        for pillar in PILLARS:
            constructed_score = constructed_entry["pillar_scores"][pillar]
            live_score = live_entry["pillar_scores"][pillar]
            divergence = round(live_score - constructed_score, 2)
            abs_divergence = abs(divergence)
            direction = "live_higher" if divergence > 0 else "live_lower" if divergence < 0 else "identical"
            severity = _severity(abs_divergence)
            if severity == "LOW":
                low_count += 1

            pillar_divergence[pillar] = {
                "constructed_score": constructed_score,
                "live_score": live_score,
                "divergence": divergence,
                "abs_divergence": abs_divergence,
                "direction": direction,
                "severity": severity,
            }

        constructed_overall = constructed_entry["overall_score"]
        live_overall = live_entry["overall_score"]
        overall_divergence = round(live_overall - constructed_overall, 2)
        overall_abs = abs(overall_divergence)
        overall_severity = _severity(overall_abs)
        agreement_rate = round(low_count / len(PILLARS) * 100, 1)

        if overall_severity == "LOW" and agreement_rate >= 80:
            verdict = "VALIDATED"
        elif overall_severity == "MEDIUM" or agreement_rate >= 50:
            verdict = "PARTIAL_MATCH"
        else:
            verdict = "DIVERGENT"

        report[company_name] = {
            "pillars": pillar_divergence,
            "constructed_overall": constructed_overall,
            "live_overall": live_overall,
            "overall_divergence": overall_divergence,
            "overall_abs_divergence": overall_abs,
            "overall_severity": overall_severity,
            "agreement_rate": agreement_rate,
            "verdict": verdict,
        }

    return report


# ---------------------------------------------------------------------------
# PART 4: academic finding generator
# ---------------------------------------------------------------------------

FINDINGS_SYSTEM_PROMPT = """You are an academic research assistant helping write the results chapter
of an MSc dissertation on agentic AI for SME digital transformation.

You will be given a cross-validation report comparing researcher-constructed
capability scores against live-measured scores for five UK fashion SMEs.

Generate up to 5 academic findings from this data. Each finding must:
1. State what was observed (specific numbers, specific companies)
2. Interpret what it means for the validity of the AIYYARY assessment instrument
3. Suggest one implication for the dissertation's methodology or limitations section

Findings should be written in academic prose suitable for a results chapter.
Reference specific pillar names, specific divergence values, and specific
company names. Do not speak in generalities.

Respond with JSON only:
{
  "findings": [
    {
      "finding_number": 1,
      "observation": "string — what was measured",
      "interpretation": "string — what it means",
      "implication": "string — for methodology or limitations"
    }
  ],
  "overall_instrument_validity_assessment": "string — one paragraph summary"
}"""


def generate_findings(divergence_report: dict, llm_client) -> dict:
    """Call the LLM once with the full divergence report to generate academic findings.

    Returns {"findings": [...], "overall_instrument_validity_assessment": "..."}
    or a "pending" placeholder if the LLM is unavailable or the call fails.
    """
    pending = {
        "findings": [],
        "overall_instrument_validity_assessment": None,
        "status": "pending",
    }

    if llm_client is None:
        print("[WARNING] No LLM client available — academic findings marked as pending.")
        return pending

    user_message = json.dumps(divergence_report, indent=2, default=str)

    try:
        result = llm_client.structured_chat(
            system=FINDINGS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            schema_description=(
                '{"findings": [{"finding_number": int, "observation": str, '
                '"interpretation": str, "implication": str}], '
                '"overall_instrument_validity_assessment": str}'
            ),
            max_tokens=2000,
        )
    except Exception as exc:
        print(f"[WARNING] LLM findings generation failed ({exc}) — academic findings marked as pending.")
        return pending

    result["status"] = "generated"
    return result


# ---------------------------------------------------------------------------
# PART 5: Agent 2 input confirmation
# ---------------------------------------------------------------------------


def confirm_agent2_inputs(live_scores: dict, divergence_report: dict) -> dict:
    """State exactly which score/tier is sent to Agent 2 per company, with confidence."""
    confidence_by_severity = {"LOW": "high", "MEDIUM": "medium", "HIGH": "low"}
    confirmation = {}

    for company_name in COMPANIES:
        live_entry = live_scores.get(company_name)
        if live_entry is None:
            confirmation[company_name] = None
            continue

        divergence_entry = divergence_report.get(company_name)
        high_divergence_pillars = []
        dissertation_flag = None

        if divergence_entry is not None:
            flag_parts = []
            for pillar, pillar_data in divergence_entry["pillars"].items():
                if pillar_data["severity"] == "HIGH":
                    high_divergence_pillars.append(pillar)
                    verb = "underestimated" if pillar_data["direction"] == "live_higher" else "overestimated"
                    flag_parts.append(
                        f"Researcher {verb} {pillar} by {pillar_data['abs_divergence']:.2f} points"
                    )
            if flag_parts:
                dissertation_flag = "; ".join(flag_parts) + " — discuss in limitations"

            confidence = confidence_by_severity[divergence_entry["overall_severity"]]
        else:
            confidence = "unknown"

        confirmation[company_name] = {
            "score_sent_to_agent2": live_entry["overall_score"],
            "readiness_tier_sent_to_agent2": live_entry["readiness_tier"],
            "score_source": "live_enrichment",
            "confidence": confidence,
            "high_divergence_pillars": high_divergence_pillars,
            "dissertation_flag": dissertation_flag,
        }

    return confirmation


# ---------------------------------------------------------------------------
# PART 6: terminal output
# ---------------------------------------------------------------------------


def _pillar_label(pillar: str) -> str:
    return pillar.replace("_", " ").capitalize()


def print_company_detail(company_name: str, divergence_entry: dict, agent2_entry: dict) -> None:
    print(f"\n{company_name.upper()}")
    print(LIGHT_SEP)
    print(f"{'Pillar':<26s}{'Constructed':>12s}{'Live':>10s}{'Diff':>10s}    Severity")
    for pillar in PILLARS:
        p = divergence_entry["pillars"][pillar]
        flag = "  ← FLAG" if p["severity"] == "HIGH" else ""
        print(
            f"{pillar:<26s}{p['constructed_score']:>12.2f}{p['live_score']:>10.2f}"
            f"{p['divergence']:>+10.2f}    {p['severity']:<8s}{flag}"
        )
    print(LIGHT_SEP)
    print(
        f"Overall:    Constructed {divergence_entry['constructed_overall']:.2f}  →  "
        f"Live {divergence_entry['live_overall']:.2f}  (diff {divergence_entry['overall_divergence']:+.2f})"
    )
    print(f"Verdict:    {divergence_entry['verdict']}  |  Agreement rate: {divergence_entry['agreement_rate']}%")
    if agent2_entry is not None:
        print(
            f"Agent 2 receives: {agent2_entry['score_sent_to_agent2']:.2f} / "
            f"{agent2_entry['readiness_tier_sent_to_agent2']}  [confidence: {agent2_entry['confidence']}]"
        )


def print_summary_table(divergence_report: dict, agent2_inputs: dict) -> None:
    print(HEAVY_SEP)
    print("SUMMARY")
    print(HEAVY_SEP)
    print(f"{'Company':<20s}{'Constructed':>12s}{'Live':>8s}{'Diff':>8s}  {'Verdict':<16s}{'A2 Confidence'}")
    for company_name in COMPANIES:
        d = divergence_report.get(company_name)
        if d is None:
            continue
        confidence = agent2_inputs[company_name]["confidence"]
        print(
            f"{company_name:<20s}{d['constructed_overall']:>12.2f}{d['live_overall']:>8.2f}"
            f"{d['overall_divergence']:>+8.2f}  {d['verdict']:<16s}{confidence}"
        )
    print(HEAVY_SEP)

    all_pillar_entries = [
        pillar_data for d in divergence_report.values() for pillar_data in d["pillars"].values()
    ]
    high_flags = [p for p in all_pillar_entries if p["severity"] == "HIGH"]
    low_count = sum(1 for p in all_pillar_entries if p["severity"] == "LOW")
    overall_agreement = round(low_count / len(all_pillar_entries) * 100, 1) if all_pillar_entries else 0.0
    companies_with_high = len({
        company_name
        for company_name, d in divergence_report.items()
        for pillar_data in d["pillars"].values()
        if pillar_data["severity"] == "HIGH"
    })
    print(f"HIGH divergence flags: {len(high_flags)} pillar(s) across {companies_with_high} company/companies")
    print(f"Overall instrument agreement: {overall_agreement}% of pillars within LOW divergence")


def print_agent2_confirmation(agent2_inputs: dict) -> None:
    print(HEAVY_SEP)
    print("SCORES CONFIRMED FOR AGENT 2")
    print(HEAVY_SEP)
    for company_name in COMPANIES:
        entry = agent2_inputs.get(company_name)
        if entry is None:
            print(f"{company_name:<18s} → [no live score available — not sent to Agent 2]")
            continue
        print(
            f"{company_name:<18s} → {entry['score_sent_to_agent2']:<6.2f}"
            f"{entry['readiness_tier_sent_to_agent2']:<16s}[{entry['confidence']} confidence]"
        )
    print(HEAVY_SEP)
    print("All scores sourced from live enrichment. Constructed scores retained as")
    print("comparative baseline for dissertation results chapter only.")


def print_findings(findings_result: dict) -> None:
    print(HEAVY_SEP)
    print("ACADEMIC FINDINGS")
    print(HEAVY_SEP)
    if findings_result.get("status") != "generated":
        print("(pending — LLM unavailable or --no-llm specified)")
        return
    for finding in findings_result.get("findings", []):
        print(f"\nFinding {finding.get('finding_number')}:")
        print(f"  Observation:    {finding.get('observation')}")
        print(f"  Interpretation: {finding.get('interpretation')}")
        print(f"  Implication:    {finding.get('implication')}")
    overall = findings_result.get("overall_instrument_validity_assessment")
    if overall:
        print(f"\nOverall instrument validity assessment:\n  {overall}")


# ---------------------------------------------------------------------------
# PART 7: saved outputs
# ---------------------------------------------------------------------------


def build_report(
    constructed_scores: dict,
    live_scores: dict,
    divergence_report: dict,
    agent2_inputs: dict,
    findings_result: dict,
) -> dict:
    all_pillar_entries = [
        pillar_data for d in divergence_report.values() for pillar_data in d["pillars"].values()
    ]
    low_count = sum(1 for p in all_pillar_entries if p["severity"] == "LOW")
    overall_agreement = round(low_count / len(all_pillar_entries) * 100, 1) if all_pillar_entries else 0.0

    high_divergence_flags = [
        {
            "company": company_name,
            "pillar": pillar,
            "constructed_score": pillar_data["constructed_score"],
            "live_score": pillar_data["live_score"],
            "divergence": pillar_data["divergence"],
        }
        for company_name, d in divergence_report.items()
        for pillar, pillar_data in d["pillars"].items()
        if pillar_data["severity"] == "HIGH"
    ]

    return {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "constructed_scores": constructed_scores,
        "live_scores": live_scores,
        "divergence_report": divergence_report,
        "agent2_inputs": agent2_inputs,
        "academic_findings": findings_result,
        "instrument_validity_summary": {
            "overall_agreement_rate": overall_agreement,
            "companies_evaluated": len(divergence_report),
            "total_pillar_comparisons": len(all_pillar_entries),
        },
        "high_divergence_flags": high_divergence_flags,
    }


def write_report_json(report: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)


def write_table_markdown(divergence_report: dict, agent2_inputs: dict, findings_result: dict, path: str) -> None:
    lines = []
    lines.append("## Cross-Validation: Constructed vs Live Capability Scores")
    lines.append("")
    lines.append("| Company | Pillar | Constructed | Live | Divergence | Severity |")
    lines.append("|---|---|---|---|---|---|")
    for company_name in COMPANIES:
        d = divergence_report.get(company_name)
        if d is None:
            continue
        for pillar in PILLARS:
            p = d["pillars"][pillar]
            lines.append(
                f"| {company_name} | {_pillar_label(pillar)} | {p['constructed_score']:.2f} | "
                f"{p['live_score']:.2f} | {p['divergence']:+.2f} | {p['severity']} |"
            )

    lines.append("")
    lines.append("### Overall Score Comparison")
    lines.append("")
    lines.append("| Company | Constructed | Live | Divergence | Verdict | Agent 2 Input |")
    lines.append("|---|---|---|---|---|---|")
    for company_name in COMPANIES:
        d = divergence_report.get(company_name)
        if d is None:
            continue
        lines.append(
            f"| {company_name} | {d['constructed_overall']:.2f} | {d['live_overall']:.2f} | "
            f"{d['overall_divergence']:+.2f} | {d['verdict']} | {d['live_overall']:.2f} |"
        )

    lines.append("")
    lines.append("### Academic Findings")
    lines.append("")
    if findings_result.get("status") == "generated":
        for finding in findings_result.get("findings", []):
            lines.append(f"**Finding {finding.get('finding_number')}.** {finding.get('observation')}")
            lines.append("")
            lines.append(f"*Interpretation:* {finding.get('interpretation')}")
            lines.append("")
            lines.append(f"*Implication:* {finding.get('implication')}")
            lines.append("")
        overall = findings_result.get("overall_instrument_validity_assessment")
        if overall:
            lines.append(f"**Overall instrument validity assessment.** {overall}")
    else:
        lines.append("_Findings pending — LLM unavailable or generation skipped for this run._")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="AIYYARY cross-validation: constructed vs live scores")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM academic findings generation")
    parser.add_argument(
        "--summary-only", action="store_true", help="Print only the summary table and Agent 2 confirmation"
    )
    parser.add_argument(
        "--export-table", action="store_true", help="Write only outputs/cross_validation_table.md and exit"
    )
    args = parser.parse_args()

    live_scores = load_live_scores(OUTPUTS_DIR)
    divergence_report = calculate_divergence(CONSTRUCTED_SCORES, live_scores)
    agent2_inputs = confirm_agent2_inputs(live_scores, divergence_report)

    skip_llm = args.no_llm or args.export_table
    findings_result = {"findings": [], "overall_instrument_validity_assessment": None, "status": "skipped"}
    if not skip_llm:
        try:
            from agents.llm_client import LLMClient

            llm_client = LLMClient()
        except Exception as exc:
            print(f"[WARNING] Could not initialise LLM client ({exc}) — academic findings marked as pending.")
            llm_client = None
        findings_result = generate_findings(divergence_report, llm_client)

    if args.export_table:
        write_table_markdown(divergence_report, agent2_inputs, findings_result, TABLE_MD_PATH)
        print(f"Saved: {os.path.relpath(TABLE_MD_PATH, SCRIPT_DIR)}")
        return

    if not args.summary_only:
        print(HEAVY_SEP)
        print("AIYYARY CROSS-VALIDATION REPORT")
        print("Constructed scores (test_agent1.py) vs Live scores (demo_five_companies.py)")
        print(HEAVY_SEP)
        for company_name in COMPANIES:
            d = divergence_report.get(company_name)
            if d is None:
                continue
            print_company_detail(company_name, d, agent2_inputs.get(company_name))
        print()

    print_summary_table(divergence_report, agent2_inputs)
    print()
    print_agent2_confirmation(agent2_inputs)

    if not args.summary_only:
        print()
        print_findings(findings_result)

    report = build_report(CONSTRUCTED_SCORES, live_scores, divergence_report, agent2_inputs, findings_result)
    write_report_json(report, REPORT_JSON_PATH)
    write_table_markdown(divergence_report, agent2_inputs, findings_result, TABLE_MD_PATH)

    print()
    print(f"Saved: {os.path.relpath(REPORT_JSON_PATH, SCRIPT_DIR)}")
    print(f"Saved: {os.path.relpath(TABLE_MD_PATH, SCRIPT_DIR)}")


if __name__ == "__main__":
    main()
