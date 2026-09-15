"""Mode 1 executor: connects a business to existing market tools rather than building custom."""

import json

from agents.executors.base_executor import BaseExecutor, ExecutorResult

SYSTEM_PROMPT = """You are AIYYARY's market integration specialist for UK fashion SMEs.
You will be given a list of existing market tools that solve a specific problem
for a fashion retailer, along with that retailer's capability profile.

Your job: rank these tools for THIS specific business (not generically), write
a configuration guide that references their actual platform and primary challenge,
and give them three concrete first steps they can take today.

Do not describe tools generically. Every sentence must reference something specific
about this business — their platform, their score, their challenge.

Respond with JSON only matching this schema:
{
  "ranked_tools": [
    {
      "rank": 1,
      "tool_name": "string",
      "why_this_business": "string — one sentence referencing their specific situation",
      "free_tier_available": true,
      "estimated_monthly_cost_gbp": "string",
      "shopify_native": true
    }
  ],
  "configuration_guide": "string — specific to this business, references their platform",
  "activation_steps": ["step 1", "step 2", "step 3"]
}"""


def _detect_shopify(agent1_output: dict) -> bool:
    for pillar_data in agent1_output.get("pillar_scores", {}).values():
        for signal in pillar_data.get("stream_b_signals", []):
            if "shopify" in signal.lower():
                return True
    return False


def _budget_tier(agent1_output: dict) -> str:
    tech_score = agent1_output.get("pillar_scores", {}).get("technology_infrastructure", {}).get("final_score", 5.0)
    if tech_score >= 7.0:
        return "higher (established tech budget)"
    if tech_score >= 4.0:
        return "moderate (mid-tier SME budget)"
    return "constrained (prioritise free/low-cost tiers)"


def _build_tool_profiles(market_tools: list, shopify_detected: bool, budget_tier: str) -> list:
    descriptions = {
        "ChatGPT API": "General-purpose AI writing for drafting on-brand product copy at scale.",
        "Claude API": "AI writing assistant well suited to nuanced, brand-voice-consistent fashion copy.",
        "Jasper": "Marketing-focused AI writing tool with fashion/retail content templates.",
        "Judge.me": "Lightweight, budget-friendly review collection app built for Shopify stores.",
        "Yotpo": "Full-featured reviews and loyalty platform for scaling fashion e-commerce brands.",
        "Trustpilot": "Independent, trust-signal-focused review platform recognisable to UK shoppers.",
        "Klaviyo": "E-commerce-native email/SMS marketing automation with deep Shopify integration.",
        "Mailchimp": "Accessible, low-setup-complexity email marketing platform for smaller teams.",
        "Omnisend": "Omnichannel (email + SMS) marketing automation aimed at growing D2C fashion brands.",
    }
    complexity = {
        "ChatGPT API": "low", "Claude API": "low", "Jasper": "low",
        "Judge.me": "low", "Yotpo": "medium", "Trustpilot": "low",
        "Klaviyo": "medium", "Mailchimp": "low", "Omnisend": "medium",
    }
    free_tier = {
        "ChatGPT API": False, "Claude API": False, "Jasper": False,
        "Judge.me": True, "Yotpo": False, "Trustpilot": True,
        "Klaviyo": True, "Mailchimp": True, "Omnisend": True,
    }
    shopify_native = {
        "ChatGPT API": False, "Claude API": False, "Jasper": False,
        "Judge.me": True, "Yotpo": True, "Trustpilot": True,
        "Klaviyo": True, "Mailchimp": True, "Omnisend": True,
    }

    profiles = []
    for tool in market_tools:
        profiles.append({
            "tool_name": tool,
            "pricing_tier_for_this_business": budget_tier,
            "shopify_integration_available": shopify_native.get(tool, False) and shopify_detected,
            "setup_complexity": complexity.get(tool, "medium"),
            "free_tier_available": free_tier.get(tool, False),
            "description": descriptions.get(tool, f"{tool} — market tool for this use case."),
        })
    return profiles


class MarketConnector(BaseExecutor):
    # market_connect mode: for opportunities better served by an existing
    # tool than a custom build, this ranks real market tools for this
    # specific business and produces a configuration guide + first steps.
    # Result lands in agent2_output["recommended"][i]["executor_result"] and
    # is what Agent 3's Phase 2 ("activate what's built") points to.
    def execute(self, profile: dict, opportunity: dict, agent1_output: dict) -> ExecutorResult:
        try:
            market_tools = opportunity.get("market_tools", [])
            shopify_detected = _detect_shopify(agent1_output)
            budget_tier = _budget_tier(agent1_output)
            tool_profiles = _build_tool_profiles(market_tools, shopify_detected, budget_tier)

            company_name = profile.get("company_name") or profile.get("business_name")
            user_message = (
                f"Business: {company_name}\n"
                f"Primary challenge: {profile.get('primary_challenge', 'Not specified')}\n"
                f"Opportunity: {opportunity.get('title')} — {opportunity.get('description')}\n"
                f"Shopify detected: {shopify_detected}\n"
                f"Budget tier: {budget_tier}\n"
                f"Candidate tool profiles:\n{json.dumps(tool_profiles, indent=2)}\n"
                "Rank these tools for this specific business and produce the configuration guide "
                "and activation steps as instructed."
            )

            schema_description = (
                '{"ranked_tools": [{"rank": 1, "tool_name": "string", "why_this_business": "string", '
                '"free_tier_available": true, "estimated_monthly_cost_gbp": "string", "shopify_native": true}], '
                '"configuration_guide": "string", "activation_steps": ["string"]}'
            )

            llm_result = self.llm.structured_chat(
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
                schema_description=schema_description,
                max_tokens=1500,
            )

            ranked_tools = llm_result.get("ranked_tools", [])
            configuration_guide = llm_result.get("configuration_guide", "")
            activation_steps = llm_result.get("activation_steps", [])

            any_free_tier = any(t.get("free_tier_available") for t in ranked_tools) or any(
                p["free_tier_available"] for p in tool_profiles
            )
            activation_time = "today" if any_free_tier else "this week"

            deliverables = [
                {
                    "name": f"configuration_guide_{t.get('tool_name', 'tool').lower().replace(' ', '_')}",
                    "type": "markdown",
                    "content": configuration_guide,
                    "instructions": f"Configuration guide for {t.get('tool_name')}, ranked #{t.get('rank')} for this business.",
                }
                for t in ranked_tools
            ]

            return ExecutorResult(
                mode="market_connect",
                status="connected",
                deliverables=deliverables,
                missing_prerequisites=[],
                next_steps=activation_steps,
                estimated_activation_time=activation_time,
                market_tools_recommended=ranked_tools,
                llm_calls_made=1,
                error=None,
            )
        except Exception as exc:
            return ExecutorResult(
                mode="market_connect",
                status="failed",
                deliverables=[],
                missing_prerequisites=[],
                next_steps=[],
                estimated_activation_time="",
                market_tools_recommended=[],
                llm_calls_made=0,
                error=str(exc),
            )
