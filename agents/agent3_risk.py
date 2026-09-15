"""Agent 3: Risk-Aware Workflow Transformation Agent for AIYYARY.

Consumes Agent 1's capability assessment output and Agent 2's opportunity
research + executor output directly. Three stages, three LLM calls:

  Stage 1 — Deterministic risk register: six taxonomy categories, each
            assigned a severity (HIGH/MEDIUM/LOW) from pillar scores and
            enrichment_structured booleans, pure Python, no LLM.
  Stage 2 — LLM Call 1: one concrete, fashion-specific mitigation sentence
            per risk. Severity is never touched by the LLM.
  Stage 3 — Deterministic phase structure: sequenced by risk severity first,
            opportunity effort second (effort ordering already baked into
            Agent 2's survivor list). Phase 1 is always remediation if any
            HIGH severity risk exists.
  Stage 4 — LLM Call 2: phase titles/descriptions/timeframes/deliverables
            for the fixed phase structure. Phase order is never touched.
  Stage 5 — LLM Call 3: a 12-month ROI summary grounded in the knowledge
            base roi_evidence/roi_source fields of Agent 2's recommended
            opportunities.

Agent 3 never re-scores pillars, never re-ranks opportunities, and never
overrides Agent 2's executor results.
"""

import json
import os
import re
from datetime import datetime, timezone

from dotenv import load_dotenv

from agents.llm_client import LLMClient

load_dotenv()

TAXONOMY_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "taxonomy", "risk_taxonomy.json")
KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base", "fashion_use_cases.json")
MARKETPLACE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "taxonomy", "ai_agent_marketplace.json")

CUSTOMER_FACING_EXECUTOR_MODES = {"full_build", "market_connect"}

# Effort levels (from ai_agent_marketplace.json "effort_to_deploy") appropriate
# for each Agent 1 readiness tier. Used to deterministically sanitize whatever
# the roadmap LLM call proposes, so tier-inappropriate agents can never reach
# the final output regardless of what the LLM does.
TIER_ALLOWED_EFFORTS = {
    "Early-Stage": {"low"},
    "Mid-Readiness": {"low", "medium"},
    "High-Readiness": {"low", "medium", "high"},
}

# Default risk categories used to backfill recommended_ai_agents for a phase
# when the LLM omits them or proposes only invalid/inappropriate agents.
# Phase 1 is handled separately (its categories come from phase_structure).
PHASE_KIND_DEFAULT_CATEGORIES = {
    "phase2_deploy_built": ["adoption_risk", "hallucination_risk"],
    "phase3_deploy_pending": ["vendor_risk", "hallucination_risk"],
    "phase4_deferred": ["ethical_risk", "data_risk"],
}

_COST_AMOUNT_RE = re.compile(r"£\s?(\d+(?:,\d{3})*)")

# Constraint: for Early-Stage businesses, Phase 1 (risk remediation) and
# Phase 2 (activate built outputs) must only ever surface free-tier or
# low-cost agents — reducing effort AND cost in the first phase, per
# AIYYARY's bridge positioning. £15/month is the low-cost cutoff.
LOW_COST_MONTHLY_THRESHOLD_GBP = 15
EARLY_STAGE_RESTRICTED_KINDS = {"phase1_risk_remediation", "phase2_deploy_built"}

MITIGATION_SYSTEM_PROMPT = """You are AIYYARY's risk specialist for UK fashion SMEs.
You will be given a risk register with ALREADY-DETERMINED severity levels.
Do NOT change any severity level. Your only job is to write one concrete,
fashion-specific mitigation sentence per risk — not generic advice.
Every sentence must reference something specific about this business:
their platform, their scores, their primary challenge, or their
enrichment signals.

Respond with JSON only:
{"mitigations": [{"category_id": "string", "mitigation": "string", "regulatory_ref": "string"}]}"""

ROADMAP_SYSTEM_PROMPT = """You are AIYYARY's transformation roadmap specialist for UK fashion SMEs.
You will be given a FIXED phase structure — do not change the phase order
or add new phases. Your job is to write a title, 2-sentence description,
realistic timeframe in weeks for a small team with no developer, and
2-3 concrete deliverables per phase.

Phase 1 is always remediation of HIGH-severity risks if any exist.
Phase 2 is activation of Agent 2 executor outputs already built.
Phase 3 is deployment of remaining executable opportunities.
Phase 4 is deferred opportunities (unlock conditions must be stated).

The total roadmap duration must be variable — derived from this
business's actual risk profile, not a fixed template.

For each roadmap phase, include a "recommended_ai_agents" array containing
1-2 verified AI agents from the provided marketplace list that can help
deliver that phase. For each agent include:
- name: the agent name exactly as provided
- what_it_does_here: one sentence on what this specific agent does for
  THIS business in THIS phase (reference their platform and challenge)
- effort: low/medium/high
- free_tier: true/false
- estimated_monthly_cost_gbp: from the marketplace data

Only recommend agents appropriate for this business's readiness tier.
Do not invent agents not in the provided marketplace list.

CRITICAL RULES FOR recommended_ai_agents:

- Phase 1 agents must address the specific HIGH-severity risk being remediated
  (regulatory risk → compliance tools like CookieYes, Termly, Iubenda;
   vendor risk → diversification tools like Loop Returns, Mailchimp backup;
   data risk → data quality tools like Prediko, Gorgias;
   adoption risk → staff onboarding tools like Lyro by Tidio)

- Phase 2 agents must help the business activate the executor output already built
  (returns form → Gorgias AI Agent for helpdesk integration;
   email sequence → Klaviyo or Lyro by Tidio for activation support)

- Phase 3 agents must be DIFFERENT from Phase 1 agents and must correspond
  specifically to the opportunities being deployed in Phase 3
  (if Phase 3 deploys size guide chatbot → recommend Fin by Intercom or Lyro;
   if Phase 3 deploys email marketing → recommend Klaviyo;
   if Phase 3 deploys review collection → recommend Gorgias AI Agent or Yotpo)

- Phase 4 agents must address the deferred opportunities and their unlock conditions
  (deferred personalisation → recommend ViSenze when data foundations improve;
   deferred demand forecasting → recommend Prediko when data matures)

- NEVER repeat the same agent name in more than two phases across one company's roadmap
- NEVER recommend a vendor risk remediation agent (Loop Returns, Mailchimp backup)
  in any phase other than Phase 1 for a company whose HIGH risk is vendor dependency

Respond with JSON only:
{
  "roadmap": [
    {
      "phase": 1,
      "title": "string",
      "timeframe_weeks": 0,
      "description": "string",
      "deliverables": ["string"],
      "human_oversight_required": true,
      "activates_executor": "executor_id or null",
      "recommended_ai_agents": [
        {
          "name": "string",
          "what_it_does_here": "string",
          "effort": "low|medium|high",
          "free_tier": true,
          "estimated_monthly_cost_gbp": "string"
        }
      ]
    }
  ],
  "total_roadmap_weeks": 0
}"""

ROI_SYSTEM_PROMPT = """You are AIYYARY's commercial analyst for UK fashion SMEs.
Write a single paragraph (4-6 sentences) estimating the 12-month
commercial outcome if this business implements the full roadmap.
Ground every claim in the ROI evidence from the recommended
opportunities — do not invent figures.
Reference the business by name. Reference specific use cases by title.
State which phase delivers which benefit and in what approximate timeframe.
Do not use hedging language like "could potentially" — be direct and specific.

headline_metric must be a single absolute GBP annual revenue figure in this
exact format: £X,XXX additional annual revenue — never a ratio, never weekly,
never a range, always annual, always GBP.

Respond with JSON only: {"roi_summary": "string", "headline_metric": "string"}"""


def load_taxonomy(path: str = TAXONOMY_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_knowledge_base(path: str = KNOWLEDGE_BASE_PATH) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_marketplace(path: str = MARKETPLACE_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class RiskTransformationAgent:
    """Orchestrates the full Agent 3 pipeline: risk register, roadmap, ROI summary."""

    def __init__(self):
        self.llm = LLMClient()
        self.taxonomy = load_taxonomy()
        self.high_threshold = self.taxonomy["thresholds"]["HIGH"]
        self.medium_threshold = self.taxonomy["thresholds"]["MEDIUM"]
        self.knowledge_base = load_knowledge_base()
        self.use_cases_by_id = {uc["id"]: uc for uc in self.knowledge_base}
        self.marketplace = load_marketplace()
        self._agents_by_name = {
            agent["name"]: agent
            for agents in self.marketplace.get("agents_by_risk_category", {}).values()
            for agent in agents
        }

    # ------------------------------------------------------------------
    # Deterministic severity assignment — no LLM, callable with no API key
    # ------------------------------------------------------------------
    # The core deterministic risk-scoring rule, called once per taxonomy
    # category. HIGH/MEDIUM/LOW severity is decided here purely from Agent
    # 1's pillar scores and enrichment_structured booleans — never by an LLM
    # — so a risk register can't be talked into understating a real risk.
    # This severity is what later decides whether Phase 1 (risk remediation)
    # exists in the roadmap at all.
    def _assign_severity(self, category: dict, pillar_scores: dict, enrichment_structured: dict) -> str:
        cat_id = category["id"]
        pillar = category["pillar_mapped"]
        score = pillar_scores.get(pillar, {}).get("final_score", 5.0)

        # Score-driven categories: severity from pillar score alone
        if category["score_driven"]:
            if score < self.high_threshold:
                return "HIGH"
            elif score < self.medium_threshold:
                return "MEDIUM"
            return "LOW"

        # Non-score-driven categories: severity from enrichment_structured signals
        # with pillar score as a secondary tiebreaker

        if cat_id == "vendor_risk":
            tools = enrichment_structured.get("marketing_automation_tools") or []
            platform = enrichment_structured.get("platform_detected")
            single_vendor = len(tools) == 1
            shopify_only = platform == "Shopify"
            if single_vendor and shopify_only:
                return "HIGH"
            elif single_vendor or shopify_only:
                return "MEDIUM"
            return "LOW"

        if cat_id == "hallucination_risk":
            # Risk increases when customer-facing AI is already active
            chat_active = enrichment_structured.get("chat_widget_found") is True
            automation_active = enrichment_structured.get("marketing_automation_detected") is True
            ai_exp_score = pillar_scores.get("ai_tool_experience", {}).get("final_score", 5.0)
            if (chat_active or automation_active) and ai_exp_score < self.high_threshold:
                return "HIGH"
            elif automation_active and ai_exp_score < self.medium_threshold:
                return "MEDIUM"
            return "LOW"

        if cat_id == "ethical_risk":
            mentions_ai = enrichment_structured.get("privacy_policy_mentions_ai")
            gov_score = pillar_scores.get("governance_compliance", {}).get("final_score", 5.0)
            if mentions_ai is False and gov_score < self.high_threshold:
                return "HIGH"
            elif mentions_ai is False:
                return "MEDIUM"
            return "LOW"

        # Default fallback
        if score < self.high_threshold:
            return "HIGH"
        elif score < self.medium_threshold:
            return "MEDIUM"
        return "LOW"

    # ------------------------------------------------------------------
    # Regulatory risk override — enrichment_structured booleans take precedence
    # ------------------------------------------------------------------
    # Lets hard evidence from the live website/Companies House scan overrule
    # the pillar-score-driven severity for regulatory_risk specifically: a
    # confirmed-missing privacy policy or cookie consent mechanism forces
    # HIGH no matter how well the owner scored on governance in the
    # interview, because that's a live, checkable regulatory fact rather
    # than a self-reported one.
    def _regulatory_override(self, base_severity: str, enrichment_structured: dict) -> tuple:
        privacy_found = enrichment_structured.get("privacy_policy_found")
        consent_found = enrichment_structured.get("cookie_consent_found")
        days_filing = enrichment_structured.get("days_since_last_filing")

        # Confirmed absence of privacy policy or consent mechanism
        # always triggers HIGH regardless of pillar score
        if privacy_found is False or consent_found is False:
            return "HIGH", "privacy_policy_found=False or cookie_consent_found=False confirmed by live website scan"

        # Long gap since last Companies House filing is a MEDIUM floor
        if days_filing is not None and days_filing > 365:
            if base_severity == "LOW":
                return "MEDIUM", f"days_since_last_filing={days_filing} exceeds 365-day threshold"

        # null means unknown — do not upgrade or downgrade
        return base_severity, "score-driven"

    def _score_trigger_reason(self, category: dict, pillar_scores: dict, enrichment_structured: dict) -> str:
        cat_id = category["id"]
        pillar = category["pillar_mapped"]
        score = pillar_scores.get(pillar, {}).get("final_score", 5.0)

        if category["score_driven"]:
            return (
                f"{pillar} final_score={score} vs thresholds HIGH<{self.high_threshold} "
                f"MEDIUM<{self.medium_threshold}"
            )

        if cat_id == "vendor_risk":
            tools = enrichment_structured.get("marketing_automation_tools") or []
            platform = enrichment_structured.get("platform_detected")
            return f"marketing_automation_tools={tools}, platform_detected={platform}"

        if cat_id == "hallucination_risk":
            chat = enrichment_structured.get("chat_widget_found")
            automation = enrichment_structured.get("marketing_automation_detected")
            ai_score = pillar_scores.get("ai_tool_experience", {}).get("final_score", 5.0)
            return (
                f"chat_widget_found={chat}, marketing_automation_detected={automation}, "
                f"ai_tool_experience final_score={ai_score}"
            )

        if cat_id == "ethical_risk":
            mentions_ai = enrichment_structured.get("privacy_policy_mentions_ai")
            gov_score = pillar_scores.get("governance_compliance", {}).get("final_score", 5.0)
            return f"privacy_policy_mentions_ai={mentions_ai}, governance_compliance final_score={gov_score}"

        return f"{pillar} final_score={score}"

    # ------------------------------------------------------------------
    # Stage 1 — build the deterministic risk register (no LLM)
    # ------------------------------------------------------------------
    # Builds the full six-category risk register for this business — no LLM
    # involved. This is Stage 1 of Agent 3 and its output (severity per
    # category) drives everything after it: which marketplace agents get
    # recommended, whether Phase 1 exists, and how long the roadmap runs.
    def _build_risk_register(self, pillar_scores: dict, enrichment_structured: dict) -> list:
        risk_register = []
        for category in self.taxonomy["categories"]:
            base_severity = self._assign_severity(category, pillar_scores, enrichment_structured)
            base_reason = self._score_trigger_reason(category, pillar_scores, enrichment_structured)

            if category["id"] == "regulatory_risk":
                severity, override_reason = self._regulatory_override(base_severity, enrichment_structured)
                trigger_reason = base_reason if override_reason == "score-driven" else override_reason
            else:
                severity = base_severity
                trigger_reason = base_reason

            pillar = category["pillar_mapped"]
            pillar_score = pillar_scores.get(pillar, {}).get("final_score", 5.0)

            risk_register.append({
                "category_id": category["id"],
                "label": category["label"],
                "severity": severity,
                "pillar": pillar,
                "pillar_score": pillar_score,
                "trigger_reason": trigger_reason,
                "mitigation": category["mitigations"][severity],
                "regulatory_ref": category["regulatory_refs"][0] if category["regulatory_refs"] else "",
                "fashion_extensions_applied": category["fashion_extensions"],
                "human_oversight_required": severity == "HIGH",
            })
        return risk_register

    # ------------------------------------------------------------------
    # Phase structure builder — pure Python, no LLM
    # ------------------------------------------------------------------
    # Turns the risk register plus Agent 2's executor results into the fixed
    # 4-phase skeleton (risk remediation → activate what's built → deploy
    # what's pending → defer the rest) before any LLM is involved. The LLM
    # roadmap call later only fills in titles/descriptions for phases this
    # function already decided must exist, in the order this function fixed.
    def _build_phase_structure(self, risk_register: list, agent2_output: dict) -> dict:
        high_risks = [r for r in risk_register if r["severity"] == "HIGH"]
        phase1_required = len(high_risks) > 0

        # Executor results already built by Agent 2
        built_executors = []
        pending_executors = []
        for opp in agent2_output.get("recommended", []):
            er = opp.get("executor_result") or {}
            if er and er.get("status") in ["complete", "connected"]:
                built_executors.append(opp)
            elif opp.get("executable"):
                pending_executors.append(opp)

        deferred = agent2_output.get("deferred", [])

        return {
            "phase1_required": phase1_required,
            "phase1_targets": [r["category_id"] for r in high_risks],
            "phase1_high_risk_pillars": [r["pillar"] for r in high_risks],
            "phase2_deploy_built": built_executors,
            "phase3_deploy_pending": pending_executors,
            "phase4_deferred": deferred,
        }

    def _phase_kind_sequence(self, phase_structure: dict) -> list:
        """Fixed logical phase order, skipping any phase with no content —
        mirrors the conditional construction in _fallback_roadmap so a
        roadmap's positional "phase" entries can be mapped back to their
        logical kind regardless of whether the LLM or the fallback built it."""
        kinds = []
        if phase_structure["phase1_required"]:
            kinds.append("phase1_risk_remediation")
        if phase_structure["phase2_deploy_built"]:
            kinds.append("phase2_deploy_built")
        if phase_structure["phase3_deploy_pending"]:
            kinds.append("phase3_deploy_pending")
        if phase_structure["phase4_deferred"]:
            kinds.append("phase4_deferred")
        return kinds

    # ------------------------------------------------------------------
    # AI agent marketplace helpers — bridge AIYYARY to third-party AI agents
    # that can solve a business's risk/opportunity gaps immediately, while
    # AIYYARY builds its own equivalents over time.
    # ------------------------------------------------------------------
    def _get_recommended_agents(self, risk_category_id: str, effort_filter: str = None) -> list:
        """Agents mapped to a single risk category, optionally filtered by effort_to_deploy."""
        agents = self.marketplace.get("agents_by_risk_category", {}).get(risk_category_id, [])
        if effort_filter:
            agents = [a for a in agents if a.get("effort_to_deploy") == effort_filter]
        return agents

    def _get_recommended_agents_for_phase(self, risk_category_ids: list, effort_filter: str = None) -> list:
        """Agents across multiple risk categories (e.g. all Phase 1 HIGH-risk
        categories), deduplicated by name, optionally filtered by effort."""
        seen = set()
        agents = []
        for cat_id in risk_category_ids or []:
            for agent in self._get_recommended_agents(cat_id, effort_filter=effort_filter):
                if agent["name"] not in seen:
                    seen.add(agent["name"])
                    agents.append(agent)
        return agents

    def _to_recommendation(self, marketplace_agent: dict, what_it_does_here: str = None) -> dict:
        return {
            "name": marketplace_agent["name"],
            "what_it_does_here": what_it_does_here or marketplace_agent.get("what_it_does", ""),
            "effort": marketplace_agent.get("effort_to_deploy"),
            "free_tier": marketplace_agent.get("free_tier", False),
            "estimated_monthly_cost_gbp": marketplace_agent.get("pricing", "Not specified"),
        }

    def _is_low_cost(self, marketplace_agent: dict) -> bool:
        if marketplace_agent.get("free_tier"):
            return True
        text = marketplace_agent.get("pricing", "") or ""
        is_annual = "/year" in text.lower() or "per year" in text.lower()
        amounts = [int(m.replace(",", "")) for m in _COST_AMOUNT_RE.findall(text)]
        if not amounts:
            return False
        amounts = [a / 12 if is_annual else a for a in amounts]
        return min(amounts) <= LOW_COST_MONTHLY_THRESHOLD_GBP

    def _filter_for_early_stage_restricted_phase(self, agents: list, readiness_tier: str, kind: str) -> list:
        """Constraint: Early-Stage businesses only see free-tier/low-cost
        agents in Phase 1 and Phase 2, regardless of effort_to_deploy."""
        if readiness_tier != "Early-Stage" or kind not in EARLY_STAGE_RESTRICTED_KINDS:
            return agents
        return [a for a in agents if self._is_low_cost(a)]

    def _sanitize_phase_agents(
        self, llm_agents: list, readiness_tier: str, kind: str = None, max_agents: int = 2,
        exclude_names: set = None,
    ) -> list:
        """Validate LLM-proposed agents against the verified marketplace list.
        Drops any agent name the LLM invented (hallucination_risk applies to
        Agent 3's own outputs too) and any agent whose effort_to_deploy is not
        appropriate for this business's readiness tier. Marketplace data
        (effort, free_tier, pricing) always wins over whatever the LLM said.
        exclude_names guarantees Phase 3 never repeats a Phase 1 agent, even if
        the LLM ignores the CRITICAL RULES instructions in the system prompt."""
        allowed_efforts = TIER_ALLOWED_EFFORTS.get(readiness_tier, {"low", "medium", "high"})
        exclude_names = exclude_names or set()
        sanitized = []
        seen_names = set()
        for entry in llm_agents or []:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            marketplace_agent = self._agents_by_name.get(name)
            if not marketplace_agent or name in seen_names or name in exclude_names:
                continue
            if marketplace_agent.get("effort_to_deploy") not in allowed_efforts:
                continue
            if not self._filter_for_early_stage_restricted_phase([marketplace_agent], readiness_tier, kind):
                continue
            seen_names.add(name)
            sanitized.append(self._to_recommendation(marketplace_agent, entry.get("what_it_does_here")))
            if len(sanitized) >= max_agents:
                break
        return sanitized

    def _backfill_phase_agents(
        self, kind: str, phase_structure: dict, readiness_tier: str, max_agents: int = 2,
        exclude_names: set = None,
    ) -> list:
        """Deterministic fallback when the LLM omitted recommended_ai_agents or
        proposed only invalid/inappropriate ones — guarantees every phase still
        carries verified, tier-appropriate marketplace agents.
        exclude_names guarantees Phase 3 never repeats a Phase 1 agent."""
        allowed_efforts = TIER_ALLOWED_EFFORTS.get(readiness_tier, {"low", "medium", "high"})
        exclude_names = exclude_names or set()
        if kind == "phase1_risk_remediation":
            candidates = self._get_recommended_agents_for_phase(
                phase_structure["phase1_targets"], effort_filter="low"
            )
        else:
            category_ids = PHASE_KIND_DEFAULT_CATEGORIES.get(kind, [])
            candidates = [
                a for a in self._get_recommended_agents_for_phase(category_ids)
                if a.get("effort_to_deploy") in allowed_efforts
            ]
        candidates = [a for a in candidates if a["name"] not in exclude_names]
        candidates = self._filter_for_early_stage_restricted_phase(candidates, readiness_tier, kind)
        return [self._to_recommendation(agent) for agent in candidates[:max_agents]]

    def _estimate_total_monthly_cost(self, agents) -> str:
        """Rough £ range from marketplace pricing strings, skipping free-tier
        agents and converting /year figures to a monthly equivalent."""
        agents = list(agents)
        non_free = [a for a in agents if not a.get("free_tier")]
        if not agents:
            return "£0"
        if not non_free:
            return "£0 (all free-tier)"

        low_total, high_total, any_cost_found = 0, 0, False
        for agent in non_free:
            text = agent.get("estimated_monthly_cost_gbp", "") or ""
            is_annual = "/year" in text.lower() or "per year" in text.lower()
            amounts = [int(m.replace(",", "")) for m in _COST_AMOUNT_RE.findall(text)]
            if not amounts:
                continue
            amounts = [round(a / 12) if is_annual else a for a in amounts]
            any_cost_found = True
            low_total += min(amounts)
            high_total += max(amounts)

        if not any_cost_found:
            return "£0 (all free-tier)"
        if low_total == high_total:
            return f"£{low_total}/month"
        return f"£{low_total}-£{high_total}/month"

    # ------------------------------------------------------------------
    # LLM Call 1 — risk mitigations
    # ------------------------------------------------------------------
    def _call_mitigations(self, profile: dict, agent1_output: dict, risk_register: list) -> dict:
        schema_description = (
            '{"mitigations": [{"category_id": "string", "mitigation": "string", "regulatory_ref": "string"}]}'
        )
        company_name = profile.get("company_name") or profile.get("business_name") or agent1_output.get("company_name")
        enrichment_structured = agent1_output.get("enrichment_structured", {})
        risk_summary = [
            {
                "category_id": r["category_id"],
                "label": r["label"],
                "severity": r["severity"],
                "trigger_reason": r["trigger_reason"],
                "pillar": r["pillar"],
                "pillar_score": r["pillar_score"],
            }
            for r in risk_register
        ]
        user_message = (
            f"Business name: {company_name}\n"
            f"Primary challenge: {profile.get('primary_challenge', 'Not specified')}\n"
            f"Platform detected: {enrichment_structured.get('platform_detected')}\n"
            f"Risk register (severities already determined, do not change):\n{json.dumps(risk_summary, indent=2)}\n"
            f"Enrichment signals:\n{json.dumps(enrichment_structured, indent=2)}\n"
            "Write one mitigation and regulatory_ref per category as instructed."
        )

        try:
            return self.llm.structured_chat(
                system=MITIGATION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
                schema_description=schema_description,
                max_tokens=1500,
            )
        except Exception as exc:
            return {
                "mitigations": [
                    {
                        "category_id": r["category_id"],
                        "mitigation": f"[LLM mitigation call failed ({exc})]; {r['mitigation']}",
                        "regulatory_ref": r["regulatory_ref"],
                    }
                    for r in risk_register
                ]
            }

    # ------------------------------------------------------------------
    # LLM Call 2 — phased roadmap descriptions
    # ------------------------------------------------------------------
    def _call_roadmap(self, profile: dict, agent1_output: dict, agent2_output: dict, phase_structure: dict) -> dict:
        schema_description = (
            '{"roadmap": [{"phase": 1, "title": "string", "timeframe_weeks": 0, "description": "string", '
            '"deliverables": ["string"], "human_oversight_required": true, "activates_executor": "executor_id or null", '
            '"recommended_ai_agents": [{"name": "string", "what_it_does_here": "string", "effort": "low|medium|high", '
            '"free_tier": true, "estimated_monthly_cost_gbp": "string"}]}], '
            '"total_roadmap_weeks": 0}'
        )
        company_name = profile.get("company_name") or profile.get("business_name") or agent1_output.get("company_name")
        enrichment_structured = agent1_output.get("enrichment_structured", {})
        readiness_tier = agent1_output.get("readiness_tier")

        # Build marketplace recommendations per phase
        phase1_agents = self._get_recommended_agents_for_phase(
            phase_structure["phase1_targets"],  # list of HIGH risk category IDs
            effort_filter="low"  # Phase 1 should only suggest low-effort agents
        )

        marketplace_context = f"""
AVAILABLE AI AGENTS FOR PHASE RECOMMENDATIONS:
For each roadmap phase, suggest 1-2 of the most relevant agents from this
verified list. Only suggest agents appropriate for the business's maturity
level and budget. For Early-Stage businesses suggest only low-effort, low-cost
or free-tier agents. For Mid-Readiness suggest low-to-medium effort agents.
For High-Readiness all effort levels are appropriate.
This business's readiness tier is: {readiness_tier}

Phase 1 relevant agents (risk remediation): {json.dumps(phase1_agents, indent=2)}
All marketplace agents available: {json.dumps(self.marketplace['agents_by_risk_category'], indent=2)}
"""
        phase_structure_summary = {
            "phase1_required": phase_structure["phase1_required"],
            "phase1_targets": phase_structure["phase1_targets"],
            "phase1_high_risk_pillars": phase_structure["phase1_high_risk_pillars"],
            "phase2_deploy_built": [
                {"id": o["id"], "title": o["title"], "executor_mode": o["executor_mode"],
                 "status": (o.get("executor_result") or {}).get("status")}
                for o in phase_structure["phase2_deploy_built"]
            ],
            "phase3_deploy_pending": [
                {"id": o["id"], "title": o["title"], "executor_mode": o["executor_mode"],
                 "implementation_weeks": o.get("implementation_weeks")}
                for o in phase_structure["phase3_deploy_pending"]
            ],
            "phase4_deferred": [
                {"id": o["id"], "title": o["title"], "unlock_condition": o.get("unlock_condition")}
                for o in phase_structure["phase4_deferred"]
            ],
        }
        user_message = (
            f"Business name: {company_name}\n"
            f"Readiness tier: {agent1_output.get('readiness_tier')}\n"
            f"Overall score: {agent1_output.get('overall_score')}\n"
            f"Fixed phase structure (do not change order or add phases):\n{json.dumps(phase_structure_summary, indent=2)}\n"
            f"Enrichment signals:\n{json.dumps(enrichment_structured, indent=2)}\n"
            f"Agent 2 quick win summary: {agent2_output.get('quick_win_summary', '')}\n"
            f"{marketplace_context}\n"
            "Write the roadmap as instructed."
        )

        try:
            return self.llm.structured_chat(
                system=ROADMAP_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
                schema_description=schema_description,
                max_tokens=3200,
            )
        except Exception as exc:
            return self._fallback_roadmap(phase_structure, exc)

    def _fallback_roadmap(self, phase_structure: dict, exc: Exception = None) -> dict:
        note = f" [LLM roadmap call failed ({exc})]" if exc else ""
        roadmap = []
        phase_number = 1

        if phase_structure["phase1_required"]:
            roadmap.append({
                "phase": phase_number,
                "title": "Risk Remediation",
                "timeframe_weeks": max(2, min(8, 2 * len(phase_structure["phase1_targets"]))),
                "description": (
                    f"Address the {len(phase_structure['phase1_targets'])} HIGH-severity risk(s) identified in the "
                    f"risk register before any Agent 2 executor output goes live.{note}"
                ),
                "deliverables": [f"Remediation of {t}" for t in phase_structure["phase1_targets"]] or ["Risk remediation plan"],
                "human_oversight_required": True,
                "activates_executor": None,
                "recommended_ai_agents": [],
            })
            phase_number += 1

        if phase_structure["phase2_deploy_built"]:
            ids = [o["id"] for o in phase_structure["phase2_deploy_built"]]
            roadmap.append({
                "phase": phase_number,
                "title": "Activate Built Executor Outputs",
                "timeframe_weeks": 1,
                "description": f"Go live with the {len(ids)} opportunity(ies) Agent 2 has already built and connected.{note}",
                "deliverables": [o["title"] for o in phase_structure["phase2_deploy_built"]],
                "human_oversight_required": True,
                "activates_executor": ids[0] if ids else None,
                "recommended_ai_agents": [],
            })
            phase_number += 1

        if phase_structure["phase3_deploy_pending"]:
            ids = [o["id"] for o in phase_structure["phase3_deploy_pending"]]
            weeks = sum(o.get("implementation_weeks") or 2 for o in phase_structure["phase3_deploy_pending"])
            roadmap.append({
                "phase": phase_number,
                "title": "Deploy Remaining Executable Opportunities",
                "timeframe_weeks": min(max(weeks, 2), 6),
                "description": f"Build out the remaining {len(ids)} executable opportunity(ies) survivors identified by Agent 2.{note}",
                "deliverables": [o["title"] for o in phase_structure["phase3_deploy_pending"]],
                "human_oversight_required": True,
                "activates_executor": ids[0] if ids else None,
                "recommended_ai_agents": [],
            })
            phase_number += 1

        if phase_structure["phase4_deferred"]:
            roadmap.append({
                "phase": phase_number,
                "title": "Future Opportunities (Deferred)",
                "timeframe_weeks": 0,
                "description": f"{len(phase_structure['phase4_deferred'])} opportunity(ies) remain deferred until prerequisite pillar scores improve.{note}",
                "deliverables": [d.get("unlock_condition", d.get("title", "")) for d in phase_structure["phase4_deferred"]],
                "human_oversight_required": False,
                "activates_executor": None,
                "recommended_ai_agents": [],
            })

        total_weeks = sum(p["timeframe_weeks"] for p in roadmap)
        return {"roadmap": roadmap, "total_roadmap_weeks": total_weeks}

    def _phase_requires_oversight(self, phase_entry: dict, phase1_required: bool, agent2_output: dict) -> bool:
        if phase_entry.get("phase") == 1 and phase1_required:
            return True
        executor_id = phase_entry.get("activates_executor")
        if not executor_id:
            return False
        opp = next((o for o in agent2_output.get("recommended", []) if o.get("id") == executor_id), None)
        if not opp:
            return False
        return opp.get("executor_mode") in CUSTOMER_FACING_EXECUTOR_MODES

    # ------------------------------------------------------------------
    # LLM Call 3 — ROI summary
    # ------------------------------------------------------------------
    def _call_roi_summary(
        self, profile: dict, agent1_output: dict, agent2_output: dict, total_roadmap_weeks: int
    ) -> dict:
        schema_description = '{"roi_summary": "string", "headline_metric": "string"}'
        company_name = profile.get("company_name") or profile.get("business_name") or agent1_output.get("company_name")

        roi_opportunities = []
        for opp in agent2_output.get("recommended", []):
            kb_entry = self.use_cases_by_id.get(opp["id"], {})
            roi_opportunities.append({
                "id": opp["id"],
                "title": opp["title"],
                "roi_evidence": kb_entry.get("roi_evidence"),
                "roi_source": kb_entry.get("roi_source"),
                "implementation_weeks": opp.get("implementation_weeks"),
            })

        user_message = (
            f"Business name: {company_name}\n"
            f"Readiness tier: {agent1_output.get('readiness_tier')}\n"
            f"Total roadmap weeks: {total_roadmap_weeks}\n"
            f"Recommended opportunities with knowledge base ROI evidence:\n{json.dumps(roi_opportunities, indent=2)}\n"
            "Write the roi_summary and headline_metric as instructed."
        )

        try:
            return self.llm.structured_chat(
                system=ROI_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
                schema_description=schema_description,
                max_tokens=800,
            )
        except Exception as exc:
            return {
                "roi_summary": (
                    f"[LLM ROI summary call failed ({exc})]; see recommended opportunities and their "
                    "roi_evidence fields for grounding."
                ),
                "headline_metric": "Not available",
            }

    # Guards against the ROI LLM call inventing an implausible number: forces
    # the exact "£X,XXX additional annual revenue" format and caps the value
    # by readiness tier (Early-Stage £100k / Mid £500k / High £2m). This is
    # the last line of defence before headline_metric reaches the final
    # agent3_*.json output a reader might actually quote.
    def _enforce_headline_format(self, metric: str, overall_score: float = 5.0) -> str:
        if not metric:
            return "£0 additional annual revenue"
        cleaned = metric.strip().rstrip(".,;:!").strip()
        if not cleaned:
            return "£0 additional annual revenue"
        if cleaned.lower().endswith("additional annual revenue"):
            result = cleaned
        else:
            result = f"{cleaned} additional annual revenue"

        # Sanity cap: extract the numeric value and cap based on readiness tier
        # Early-Stage businesses (score < 5.0): cap at £100,000
        # Mid-Readiness businesses (score 5.0-7.5): cap at £500,000
        # High-Readiness businesses (score >= 7.5): cap at £2,000,000
        numbers = re.findall(r'[\d,]+', result)
        if numbers:
            try:
                value = int(numbers[0].replace(',', ''))
                if overall_score < 5.0 and value > 100000:
                    result = f"£{100000:,} additional annual revenue"
                elif overall_score < 7.5 and value > 500000:
                    result = f"£{500000:,} additional annual revenue"
                elif value > 2000000:
                    result = f"£{2000000:,} additional annual revenue"
            except ValueError:
                pass
        return result

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
        filename = f"agent3_{safe_company}_{timestamp}.json"
        return os.path.join(output_dir, filename)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    # Main entry point for Agent 3 — the final stage of the pipeline. Reads
    # Agent 1's pillar scores and Agent 2's recommended/executed opportunities
    # directly and never re-scores or re-ranks either. Produces the risk
    # register, the phased roadmap with marketplace agent recommendations,
    # and the ROI headline, then writes agent3_*.json — the artefact a
    # business owner or dissertation examiner actually reads at the end.
    def run(self, agent1_output: dict, agent2_output: dict, profile: dict = None) -> dict:
        if profile is None:
            profile = {}

        company_name = agent1_output.get("company_name")
        pillar_scores = agent1_output.get("pillar_scores", {})
        enrichment_structured = agent1_output.get("enrichment_structured", {}) or {}
        overall_score = agent1_output.get("overall_score")
        readiness_tier = agent1_output.get("readiness_tier")

        # Stage 1 — deterministic risk register
        risk_register = self._build_risk_register(pillar_scores, enrichment_structured)

        # Stage 2 — LLM mitigations (do not touch severity)
        mitigations_data = self._call_mitigations(profile, agent1_output, risk_register)
        mitigation_by_id = {m["category_id"]: m for m in mitigations_data.get("mitigations", []) if "category_id" in m}
        for entry in risk_register:
            override = mitigation_by_id.get(entry["category_id"])
            if override and override.get("mitigation"):
                entry["mitigation"] = override["mitigation"]
            if override and override.get("regulatory_ref"):
                entry["regulatory_ref"] = override["regulatory_ref"]

        high_severity_count = sum(1 for r in risk_register if r["severity"] == "HIGH")
        medium_severity_count = sum(1 for r in risk_register if r["severity"] == "MEDIUM")
        low_severity_count = sum(1 for r in risk_register if r["severity"] == "LOW")

        # Stage 3 — deterministic phase structure
        phase_structure = self._build_phase_structure(risk_register, agent2_output)

        # Stage 4 — LLM roadmap descriptions
        roadmap_data = self._call_roadmap(profile, agent1_output, agent2_output, phase_structure)
        roadmap = roadmap_data.get("roadmap")
        expected_phase_count = len(self._phase_kind_sequence(phase_structure))
        if (
            not roadmap
            or not isinstance(roadmap, list)
            or not all(isinstance(p, dict) for p in roadmap)
            or len(roadmap) != expected_phase_count
        ):
            # Malformed or truncated LLM roadmap (e.g. a stray string element,
            # wrong phase count) — never trust raw LLM structure, fall back
            # to the deterministic roadmap builder instead of crashing.
            roadmap = self._fallback_roadmap(phase_structure)["roadmap"]

        # Enforce phase duration rules deterministically, regardless of what the
        # LLM (or fallback) produced: Phase 1 is 2 weeks per HIGH-severity risk
        # (min 2, max 8 weeks); Phase 3 and Phase 4 are capped at 6 weeks each
        # no matter how many opportunities they contain.
        phase_kinds = self._phase_kind_sequence(phase_structure)
        for phase_entry, kind in zip(roadmap, phase_kinds):
            if kind == "phase1_risk_remediation":
                num_high = len(phase_structure["phase1_targets"])
                phase_entry["timeframe_weeks"] = max(2, min(8, 2 * num_high))
            elif kind == "phase2_deploy_built":
                phase_entry["timeframe_weeks"] = 1
            elif kind == "phase3_deploy_pending":
                phase_entry["timeframe_weeks"] = min(phase_entry.get("timeframe_weeks", 0), 6)
            elif kind == "phase4_deferred":
                num_deferred = len(phase_structure["phase4_deferred"])
                if num_deferred == 0:
                    phase_entry["timeframe_weeks"] = 0
                elif num_deferred <= 2:
                    phase_entry["timeframe_weeks"] = 3
                else:
                    phase_entry["timeframe_weeks"] = 6

        total_roadmap_weeks = sum(p.get("timeframe_weeks", 0) for p in roadmap)

        # Enforce human_oversight_required deterministically per the stated rule:
        # true for any phase activating a customer-facing executor output, and
        # for any phase that is HIGH-severity risk remediation (Phase 1).
        for phase_entry in roadmap:
            phase_entry["human_oversight_required"] = self._phase_requires_oversight(
                phase_entry, phase_structure["phase1_required"], agent2_output
            )

        # Sanitize recommended_ai_agents deterministically: drop any agent the
        # LLM invented or that isn't tier-appropriate, then backfill from the
        # verified marketplace if a phase ends up with none. Never trust the
        # LLM's agent list unchecked — same principle as hallucination_risk.
        # Phase 3 additionally excludes whatever Phase 1 already recommended:
        # Phase 1 addresses risk remediation, Phase 3 addresses deployment of
        # remaining opportunities — they are different problems and must not
        # share tools, even if the LLM's own output repeats one.
        phase1_agent_names = set()
        for phase_entry, kind in zip(roadmap, phase_kinds):
            exclude_names = phase1_agent_names if kind == "phase3_deploy_pending" else None
            sanitized = self._sanitize_phase_agents(
                phase_entry.get("recommended_ai_agents"), readiness_tier, kind, exclude_names=exclude_names
            )
            if not sanitized:
                sanitized = self._backfill_phase_agents(
                    kind, phase_structure, readiness_tier, exclude_names=exclude_names
                )
            phase_entry["recommended_ai_agents"] = sanitized
            if kind == "phase1_risk_remediation":
                phase1_agent_names = {a["name"] for a in sanitized}

        all_recommended = [a for p in roadmap for a in p.get("recommended_ai_agents", [])]
        unique_agents_by_name = {}
        for a in all_recommended:
            unique_agents_by_name.setdefault(a["name"], a)
        marketplace_summary = {
            "total_agents_recommended": len(unique_agents_by_name),
            "free_tier_agents_count": sum(1 for a in unique_agents_by_name.values() if a.get("free_tier")),
            "estimated_total_monthly_cost_gbp": self._estimate_total_monthly_cost(unique_agents_by_name.values()),
            "aiyyary_bridge_note": (
                "These agents are recommended as immediate solutions. As AIYYARY grows, "
                "custom-built equivalents will replace third-party dependencies where appropriate."
            ),
        }

        # Stage 5 — LLM ROI summary
        roi_data = self._call_roi_summary(profile, agent1_output, agent2_output, total_roadmap_weeks)
        headline_metric = self._enforce_headline_format(
            roi_data.get("headline_metric", ""),
            overall_score=agent1_output.get("overall_score", 5.0)
        )

        output_file = self._output_filepath(company_name)

        result = {
            "agent": "Agent3_RiskTransformation",
            "schema_version": "1.0",
            "assessment_timestamp": datetime.now(timezone.utc).isoformat(),
            "company_name": company_name,
            "overall_score": overall_score,
            "readiness_tier": readiness_tier,
            "risk_register": risk_register,
            "high_severity_count": high_severity_count,
            "medium_severity_count": medium_severity_count,
            "low_severity_count": low_severity_count,
            "phase_structure": phase_structure,
            "roadmap": roadmap,
            "total_roadmap_weeks": total_roadmap_weeks,
            "roi_summary": roi_data.get("roi_summary", ""),
            "headline_metric": headline_metric,
            "marketplace_summary": marketplace_summary,
            "enrichment_signals_used": {
                "privacy_policy_found": enrichment_structured.get("privacy_policy_found"),
                "cookie_consent_found": enrichment_structured.get("cookie_consent_found"),
                "marketing_automation_tools": enrichment_structured.get("marketing_automation_tools"),
                "platform_detected": enrichment_structured.get("platform_detected"),
                "company_status": enrichment_structured.get("company_status"),
                "days_since_last_filing": enrichment_structured.get("days_since_last_filing"),
            },
            "traceability": {
                "agent1_schema_version": agent1_output.get("schema_version", "2.1"),
                "agent2_schema_version": agent2_output.get("schema_version", "2.0"),
                "severity_method": "deterministic_threshold_plus_enrichment_override",
                "llm_calls_made": 3,
                "output_file": output_file,
            },
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved output to: {output_file}")

        return result
