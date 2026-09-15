import json
import os
from datetime import datetime

from dotenv import load_dotenv

from agents.llm_client import chat, parse_json

load_dotenv()

PILLAR_KEYS = [
    "data_foundations",
    "process_digitisation",
    "technology_infrastructure",
    "staff_digital_literacy",
    "governance_compliance",
    "ai_tool_experience",
]

SYSTEM_PROMPT = (
    "You are AIYYARY's capability assessment analyst for UK fashion and apparel SMEs.\n"
    "You will be given ALREADY-CALCULATED pillar scores and an overall score for a specific business.\n"
    "These scores are arithmetically correct - do not recalculate or modify them under any circumstances.\n"
    "Your only job is:\n"
    "1. Write a 2-3 sentence verdict explaining what these specific numbers mean for this specific business, "
    "referencing their company name, industry, and primary challenge.\n"
    "2. Identify the 3 lowest-scoring pillars as top_gaps. For each, cite the exact pillar name and score, "
    "and write one sentence on the operational impact for a small fashion retailer.\n"
    "3. Identify the 2 highest-scoring pillars as strengths. For each, cite the exact pillar name and score, "
    "and write one sentence on how this can be leveraged.\n"
    "Respond with valid JSON only. No markdown fences. No prose outside the JSON.\n"
    "Exact schema required:\n"
    '{"verdict": "string", "top_gaps": [{"pillar": "string", "score": float, "impact": "string"}], '
    '"strengths": [{"pillar": "string", "score": float, "advantage": "string"}]}\n'
    "These scores are already calculated and arithmetically correct. Do not recalculate them. "
    "Your only job is to write the verdict, gaps, and strengths based on these exact numbers."
)


class CapabilityAssessmentAgent:
    def compute_scores(self, instrument: dict) -> tuple[dict, float]:
        """Deterministically compute pillar scores and overall score. No API calls."""
        pillar_scores = {}
        for pillar in PILLAR_KEYS:
            answers = instrument[pillar]
            raw_scores = [a["score"] for a in answers]
            raw_mean = sum(raw_scores) / len(raw_scores)
            scaled = ((raw_mean - 1) / 8) * 9 + 1
            pillar_scores[pillar] = round(scaled, 2)

        overall_score = round(
            sum(pillar_scores.values()) / len(pillar_scores), 2
        )
        return pillar_scores, overall_score

    def readiness_tier(self, overall: float) -> str:
        if overall >= 7.5:
            return "High-Readiness"
        elif overall >= 5.0:
            return "Mid-Readiness"
        else:
            return "Early-Stage"

    def run(self, profile: dict, instrument: dict) -> dict:
        pillar_scores, overall_score = self.compute_scores(instrument)
        tier = self.readiness_tier(overall_score)

        user_prompt = (
            f"Company name: {profile.get('company_name')}\n"
            f"Industry: {profile.get('industry')}\n"
            f"Primary challenge: {profile.get('primary_challenge')}\n\n"
            f"Already-calculated pillar scores (1-10 scale, do not recalculate):\n"
            f"{json.dumps(pillar_scores, indent=2)}\n\n"
            f"Already-calculated overall score: {overall_score}\n"
            f"Readiness tier: {tier}\n\n"
            "Write the verdict, top_gaps, and strengths JSON as instructed."
        )

        raw_response = chat(
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=2048,
        )
        llm_output = parse_json(raw_response)

        result = {
            "agent": "Agent1_CapabilityAssessment",
            "company_name": profile.get("company_name"),
            "pillar_scores": pillar_scores,
            "overall_score": overall_score,
            "readiness_tier": tier,
            "verdict": llm_output.get("verdict", ""),
            "top_gaps": llm_output.get("top_gaps", []),
            "strengths": llm_output.get("strengths", []),
        }

        self._save_output(result)
        return result

    def _save_output(self, result: dict) -> str:
        output_dir = os.getenv("OUTPUT_DIR", "./outputs")
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_company = "".join(
            c if c.isalnum() or c in ("-", "_") else "_"
            for c in str(result.get("company_name", "unknown"))
        )
        filename = f"agent1_{safe_company}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(f"Saved output to: {filepath}")
        return filepath
