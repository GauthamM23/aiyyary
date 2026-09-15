"""Agent 1: Capability Assessment Agent for AIYYARY.

Synthesises three evidence streams into a single, traceable capability
assessment for a UK fashion/apparel SME:

  Stream A (60% weight) — owner interview instrument, pure Python scoring.
  Stream B (30% weight) — live public-data enrichment (EnrichmentAgent).
  Stream C (10% weight) — LLM reconciliation, used only when A and B
                           diverge by more than 2.0 points on the 10-point
                           scale for a given pillar.

Output is the exact schema Agent 2 and Agent 3 consume directly.
"""

import json
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

from agents.enrichment import EnrichmentAgent
from agents.llm_client import LLMClient

load_dotenv()

PILLARS = [
    "data_foundations",
    "process_digitisation",
    "technology_infrastructure",
    "staff_digital_literacy",
    "governance_compliance",
    "ai_tool_experience",
]

RECONCILIATION_SYSTEM_PROMPT = """You are AIYYARY's capability reconciliation analyst. You will be given two independently
derived scores for a single capability pillar of a UK fashion SME, along with the evidence
behind each score. Your job is to produce a single reconciled score with explicit reasoning.

Rules:
1. The owner interview score (Stream A) carries more weight than public data (Stream B)
   because internal operational data is not visible from outside the business.
2. However, if the public data directly contradicts a factual claim
   (e.g. no privacy policy found but owner claims strong governance), Stream B should
   reduce the final score.
3. Your reconciled_score must be a float between 1.0 and 10.0.
4. Respond with JSON only: {"reconciled_score": float, "reasoning": "string"}"""

VERDICT_SYSTEM_PROMPT = """You are AIYYARY's capability assessment analyst for UK fashion and apparel SMEs.
You will be given ALREADY-CALCULATED and ALREADY-FINALISED pillar scores and an overall
score for a specific business. These scores reflect a weighted synthesis of a live owner
interview, real-time public data enrichment, and where applicable an LLM reconciliation.
Do not recalculate or modify them.

Your job:
1. Write a 2-3 sentence verdict explaining what these specific numbers mean for this
   specific business. Reference their company name, primary challenge, and the most
   significant finding from the public data enrichment.
2. Identify the 3 lowest-scoring pillars as top_gaps. For each, cite the exact pillar
   name and finalised score, identify which evidence stream drove the score, and write
   one sentence on the operational impact for a small fashion retailer specifically.
3. Identify the 2 highest-scoring pillars as strengths. For each, cite the exact pillar
   name and finalised score and write one sentence on how this can be leveraged in the
   transformation roadmap.
4. Write a data_confidence_note: a single sentence assessing how much confidence to
   place in this overall assessment, based on how many enrichment sources were available
   and whether any Stream C reconciliations were required. If stream_c_reconciliations is
   greater than 0, data_confidence_note must explicitly state how many reconciliations
   occurred and which pillars were affected. Do not say "no Stream C reconciliations" if
   the count is greater than zero.

Respond with valid JSON only. No markdown fences. No prose outside the JSON.
Schema:
{
  "verdict": "string",
  "top_gaps": [{"pillar": "string", "score": float, "dominant_stream": "A/B/C", "impact": "string"}],
  "strengths": [{"pillar": "string", "score": float, "dominant_stream": "A/B/C", "advantage": "string"}],
  "data_confidence_note": "string"
}"""

FASHION_SYSTEM_PROMPT = """You are a specialist in AI adoption for UK independent fashion and apparel retailers.
You will be given a complete capability profile for a specific fashion SME.

Identify up to 3 fashion-specific readiness signals that are particularly relevant
for this business's profile — things that matter in fashion AI deployment that the
six generic pillars do not fully capture. Examples: sizing data structure, returns
pattern visibility, seasonal buying cycle maturity, supplier data integration.

For each signal, state: what it is, why it matters specifically for fashion AI,
and what the available evidence suggests about this business's position on it.

Respond with JSON only:
{"fashion_signals": [{"signal": "string", "relevance": "string", "evidence": "string"}]}"""


# Turns the 24-question owner interview into the six Stream A pillar scores.
# This is the 60%-weighted anchor of every final pillar score Agent 1
# produces — Agent 2's prerequisite filter and Agent 3's risk severity both
# ultimately trace back through this number, so an error here propagates
# through the entire pipeline. Deliberately pure Python: no network call, no
# LLM, so it always succeeds even if every external service is down.
def compute_stream_a(instrument: dict) -> dict:
    """Compute Stream A pillar scores from the 24-question owner interview instrument.

    Pure Python — no network calls, no API keys required. `instrument[pillar]` must be
    a list of 4 raw scores (1-9), either bare numbers or dicts carrying a "score" key.
    """
    pillar_scores = {}
    for pillar in PILLARS:
        answers = instrument[pillar]
        raw_scores = [a["score"] if isinstance(a, dict) else a for a in answers]
        raw_mean = sum(raw_scores) / len(raw_scores)
        scaled = ((raw_mean - 1) / 8) * 9 + 1
        pillar_scores[pillar] = round(scaled, 2)
    return pillar_scores


def compute_stream_a_overall(pillar_scores: dict) -> float:
    """Overall Stream A score = mean of the 6 pillar Stream A scores."""
    return round(sum(pillar_scores[p] for p in PILLARS) / len(PILLARS), 2)


# Maps the overall score to the High/Mid/Early-Stage tier that Agent 2 and
# Agent 3 both read directly (agent1_output["readiness_tier"]) — it is what
# ultimately gates which AI marketplace agents Agent 3 is allowed to
# recommend and what ceiling is applied to Agent 3's headline ROI figure.
def readiness_tier(overall_score: float) -> str:
    if overall_score >= 7.5:
        return "High-Readiness"
    if overall_score >= 5.0:
        return "Mid-Readiness"
    return "Early-Stage"


def _safe_get(d, *keys):
    """Nested dict lookup that returns None instead of raising on a missing/non-dict path."""
    current = d
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def extract_structured_enrichment(enrichment_report: dict) -> dict:
    """Flatten a fixed set of structured fields out of enrichment_report['raw_findings'].

    Downstream agents (e.g. Agent 3) need clean booleans/scalars rather than parsing
    stream_b_signals prose. Every field defaults to None if the underlying enrichment
    data is missing; the whole dict is empty if enrichment_report itself is missing.
    """
    if not enrichment_report:
        return {}

    raw_findings = enrichment_report.get("raw_findings") or {}
    website = raw_findings.get("website") or {}
    google_profile = raw_findings.get("google_profile") or {}
    companies_house = raw_findings.get("companies_house") or {}
    facebook_ads = raw_findings.get("facebook_ads") or {}

    marketing_count = _safe_get(website, "marketing_automation", "count")

    return {
        "privacy_policy_found": _safe_get(website, "privacy_policy", "found"),
        "privacy_policy_mentions_ai": _safe_get(website, "privacy_policy", "mentions_ai"),
        "cookie_consent_found": _safe_get(website, "cookie_consent", "found"),
        "cookie_consent_provider": _safe_get(website, "cookie_consent", "provider"),
        "marketing_automation_detected": None if marketing_count is None else bool(marketing_count),
        "marketing_automation_tools": _safe_get(website, "marketing_automation", "tools_detected"),
        "analytics_tools_detected": _safe_get(website, "analytics", "tools_detected"),
        "chat_widget_found": _safe_get(website, "chat_widget", "found"),
        "chat_widget_provider": _safe_get(website, "chat_widget", "provider"),
        "ssl_enabled": _safe_get(website, "ssl", "enabled"),
        "platform_detected": _safe_get(website, "platform", "name"),
        "company_status": _safe_get(companies_house, "company_status"),
        "days_since_last_filing": _safe_get(companies_house, "days_since_last_filing"),
        "google_review_count": _safe_get(google_profile, "total_reviews"),
        "has_active_fb_ads": _safe_get(facebook_ads, "has_active_ads"),
    }


class CapabilityAssessmentAgent:
    """Orchestrates the full Agent 1 three-stream capability assessment."""

    def __init__(self):
        self.llm = LLMClient()

    # ------------------------------------------------------------------
    # Stream C reconciliation (LLM Call 1, conditional, per-pillar)
    # ------------------------------------------------------------------
    # Fires only when Stream A (owner) and Stream B (public data) disagree by
    # more than 2.0 points on a pillar — i.e. when the owner's self-report and
    # what's actually visible online tell two different stories. Builds
    # Stream C, the tie-breaking 10% of the final weighted score for that
    # pillar. Never raises: on any LLM failure it falls back to a
    # deterministic 60/40 A/B blend so the pipeline keeps running.
    def _reconcile_stream_c(
        self, pillar: str, stream_a_score: float, stream_b_score: float, evidence_signals: list
    ) -> dict:
        schema_description = '{"reconciled_score": float, "reasoning": "string"}'
        evidence_text = (
            "\n".join(f"- {s}" for s in evidence_signals) if evidence_signals else "- No public data evidence available"
        )
        user_message = (
            f"Pillar: {pillar}\n"
            f"Stream A (owner interview) score: {stream_a_score}\n"
            f"Stream B (public data) score: {stream_b_score}\n"
            f"Stream B evidence:\n{evidence_text}"
        )

        try:
            result = self.llm.structured_chat(
                system=RECONCILIATION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
                schema_description=schema_description,
                max_tokens=500,
            )
        except Exception as exc:
            fallback_score = round((stream_a_score * 0.6) + (stream_b_score * 0.4), 2)
            result = {
                "reconciled_score": fallback_score,
                "reasoning": f"[LLM reconciliation failed ({exc}); used a weighted Stream A/B fallback]",
            }

        reconciled_score = result.get("reconciled_score")
        if reconciled_score is None:
            reconciled_score = round((stream_a_score * 0.6) + (stream_b_score * 0.4), 2)
        reconciled_score = max(1.0, min(10.0, float(reconciled_score)))
        result["reconciled_score"] = round(reconciled_score, 2)
        result.setdefault("reasoning", "")
        return result

    # ------------------------------------------------------------------
    # LLM Call 2 — qualitative verdict (always called, once per run)
    # ------------------------------------------------------------------
    def _call_verdict(
        self,
        profile: dict,
        pillar_results: dict,
        overall_score: float,
        tier: str,
        enrichment_report: dict,
        stream_c_count: int,
    ) -> dict:
        schema_description = (
            '{"verdict": "string", '
            '"top_gaps": [{"pillar": "string", "score": float, "dominant_stream": "A/B/C", "impact": "string"}], '
            '"strengths": [{"pillar": "string", "score": float, "dominant_stream": "A/B/C", "advantage": "string"}], '
            '"data_confidence_note": "string"}'
        )
        scores_summary = {
            p: {"final_score": r["final_score"], "dominant_stream": r["dominant_stream"]}
            for p, r in pillar_results.items()
        }
        user_message = (
            f"Company name: {profile.get('company_name') or profile.get('business_name')}\n"
            f"Primary challenge: {profile.get('primary_challenge', 'Not specified')}\n"
            f"Finalised pillar scores and dominant evidence stream:\n{json.dumps(scores_summary, indent=2)}\n"
            f"Overall score: {overall_score}\n"
            f"Readiness tier: {tier}\n"
            f"Enrichment sources available: {enrichment_report.get('sources_available', [])}\n"
            f"Enrichment sources failed: {enrichment_report.get('sources_failed', [])}\n"
            f"Stream C reconciliations performed: {stream_c_count}\n"
            "Write the verdict, top_gaps, strengths, and data_confidence_note as instructed."
        )

        try:
            return self.llm.structured_chat(
                system=VERDICT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
                schema_description=schema_description,
                max_tokens=1500,
            )
        except Exception as exc:
            sorted_pillars = sorted(pillar_results.items(), key=lambda kv: kv[1]["final_score"])
            gaps = sorted_pillars[:3]
            strengths = sorted_pillars[-2:]
            return {
                "verdict": f"[LLM verdict call failed: {exc}]",
                "top_gaps": [
                    {"pillar": p, "score": r["final_score"], "dominant_stream": r["dominant_stream"], "impact": "Not available"}
                    for p, r in gaps
                ],
                "strengths": [
                    {"pillar": p, "score": r["final_score"], "dominant_stream": r["dominant_stream"], "advantage": "Not available"}
                    for p, r in strengths
                ],
                "data_confidence_note": "LLM verdict call failed; confidence assessment unavailable.",
            }

    # ------------------------------------------------------------------
    # LLM Call 3 — fashion-specific contextualisation (always called, once per run)
    # ------------------------------------------------------------------
    def _call_fashion_context(self, profile: dict, pillar_results: dict, overall_score: float) -> dict:
        schema_description = '{"fashion_signals": [{"signal": "string", "relevance": "string", "evidence": "string"}]}'
        user_message = (
            f"Company: {profile.get('company_name') or profile.get('business_name')}\n"
            f"Industry: {profile.get('industry', 'Fashion and Apparel')}\n"
            f"Finalised pillar scores: {json.dumps({p: r['final_score'] for p, r in pillar_results.items()}, indent=2)}\n"
            f"Overall score: {overall_score}\n"
            "Identify up to 3 fashion-specific AI readiness signals as instructed."
        )

        try:
            return self.llm.structured_chat(
                system=FASHION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
                schema_description=schema_description,
                max_tokens=1000,
            )
        except Exception:
            return {"fashion_signals": []}

    # ------------------------------------------------------------------
    # Output persistence
    # ------------------------------------------------------------------
    def _output_filepath(self, company_name: str) -> str:
        output_dir = os.getenv("OUTPUT_DIR", "./outputs")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_company = "".join(
            c if c.isalnum() or c in ("-", "_") else "_" for c in str(company_name or "unknown")
        )
        filename = f"agent1_{safe_company}_{timestamp}.json"
        return os.path.join(output_dir, filename)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    # Main entry point for Agent 1 — the whole pipeline starts here. Combines
    # Stream A (owner interview), Stream B (live enrichment), and conditional
    # Stream C into six final pillar scores and one overall score, then
    # writes agent1_*.json to outputs/. That JSON file's exact schema
    # (pillar_scores, overall_score, readiness_tier, enrichment_structured)
    # is the entire contract Agent 2 and Agent 3 are built against — nothing
    # downstream re-derives a capability score from scratch.
    def run(self, profile: dict, instrument: dict, enrichment_report: dict = None) -> dict:
        if enrichment_report is None:
            enrichment_report = EnrichmentAgent().run(profile)

        # 1. Stream A — pure Python, no network
        stream_a_scores = compute_stream_a(instrument)

        # 2. Stream B — extracted from enrichment report
        stream_b_signals_map = enrichment_report.get("stream_b_signals", {})

        # 3 & 4. Divergence check, conditional Stream C, final weighted score per pillar
        pillar_results = {}
        llm_calls_made = 0
        stream_c_count = 0

        for pillar in PILLARS:
            a_score = stream_a_scores[pillar]
            b_info = stream_b_signals_map.get(pillar, {})
            b_score = b_info.get("stream_b_score")
            b_signals = b_info.get("signals", [])

            stream_c_reconciled = None
            stream_c_reasoning = None

            if b_score is None:
                final_score = a_score
                dominant_stream = "A"
            elif abs(a_score - b_score) > 2.0:
                reconciliation = self._reconcile_stream_c(pillar, a_score, b_score, b_signals)
                llm_calls_made += 1
                stream_c_count += 1
                stream_c_reconciled = reconciliation["reconciled_score"]
                stream_c_reasoning = reconciliation.get("reasoning")
                final_score = (a_score * 0.6) + (b_score * 0.3) + (stream_c_reconciled * 0.1)
                dominant_stream = "C"
            else:
                final_score = (a_score * 0.7) + (b_score * 0.3)
                dominant_stream = "A"

            pillar_results[pillar] = {
                "stream_a_score": a_score,
                "stream_b_score": b_score,
                "stream_c_reconciled": stream_c_reconciled,
                "final_score": round(final_score, 2),
                "dominant_stream": dominant_stream,
                "stream_b_signals": b_signals,
                "stream_c_reasoning": stream_c_reasoning,
            }

        # 5. Overall final score
        overall_score = round(sum(r["final_score"] for r in pillar_results.values()) / len(PILLARS), 2)

        # 6. Readiness tier
        tier = readiness_tier(overall_score)

        # 7. LLM verdict
        verdict_data = self._call_verdict(profile, pillar_results, overall_score, tier, enrichment_report, stream_c_count)
        llm_calls_made += 1

        # 8. Fashion contextualisation
        fashion_data = self._call_fashion_context(profile, pillar_results, overall_score)
        llm_calls_made += 1

        # 9. Structured enrichment fields for downstream agents (e.g. Agent 3)
        enrichment_structured = extract_structured_enrichment(enrichment_report)

        # 10. Assemble output
        company_name = profile.get("company_name") or profile.get("business_name")
        output_file = self._output_filepath(company_name)

        instrument_answers_count = sum(len(instrument[p]) for p in PILLARS)
        enrichment_signals_count = sum(len(stream_b_signals_map.get(p, {}).get("signals", [])) for p in PILLARS)

        result = {
            "agent": "Agent1_CapabilityAssessment",
            "schema_version": "2.1",
            "assessment_timestamp": datetime.now(timezone.utc).isoformat(),
            "company_name": company_name,
            "industry": profile.get("industry", "Fashion and Apparel"),
            "llm_mode": self.llm.llm_mode,
            "pillar_scores": {
                p: {
                    "stream_a_score": r["stream_a_score"],
                    "stream_b_score": r["stream_b_score"],
                    "stream_c_reconciled": r["stream_c_reconciled"],
                    "final_score": r["final_score"],
                    "dominant_stream": r["dominant_stream"],
                    "stream_b_signals": r["stream_b_signals"],
                    "stream_c_reasoning": r["stream_c_reasoning"],
                }
                for p, r in pillar_results.items()
            },
            "overall_score": overall_score,
            "readiness_tier": tier,
            "verdict": verdict_data.get("verdict", ""),
            "top_gaps": verdict_data.get("top_gaps", []),
            "strengths": verdict_data.get("strengths", []),
            "data_confidence_note": verdict_data.get("data_confidence_note", ""),
            "fashion_signals": fashion_data.get("fashion_signals", []),
            "enrichment_structured": enrichment_structured,
            "enrichment_summary": {
                "sources_available": enrichment_report.get("sources_available", []),
                "sources_failed": enrichment_report.get("sources_failed", []),
                "stream_c_reconciliations": stream_c_count,
                "enrichment_duration_seconds": enrichment_report.get("enrichment_duration_seconds", 0.0),
            },
            "traceability": {
                "instrument_answers_count": instrument_answers_count,
                "enrichment_signals_count": enrichment_signals_count,
                "llm_calls_made": llm_calls_made,
                "output_file": output_file,
            },
        }

        # 11. Save to outputs/
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved output to: {output_file}")

        return result
