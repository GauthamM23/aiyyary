"""Mode 2 full-build executor: builds a three-email abandoned cart recovery sequence."""

from agents.executors.base_executor import BaseExecutor, ExecutorResult

SEQUENCE_SYSTEM_PROMPT = """You are a fashion email marketing specialist writing for a UK independent fashion brand.
Write a three-email abandoned cart recovery sequence in this brand's specific voice.
Email 1: sent 1 hour after abandonment — gentle reminder, no discount.
Email 2: sent 24 hours after abandonment — social proof, highlight brand values.
Email 3: sent 72 hours after abandonment — final nudge with a small incentive.

Each email must:
- Reference the specific brand name and its values
- Sound like a real person wrote it, not a marketing template
- Have a subject line, preview text, and body copy
- Be under 150 words body copy

Return JSON:
{
  "sequence": [
    {
      "email_number": 1,
      "send_timing": "1 hour after abandonment",
      "subject_line": "string",
      "preview_text": "string",
      "body_copy": "string",
      "call_to_action": "string"
    }
  ]
}

Respond with valid JSON only — no markdown code fences, no explanation before or after the JSON."""

SETUP_SYSTEM_PROMPT = (
    "You are a Klaviyo and Shopify Email specialist. Write exact step-by-step setup instructions "
    "for implementing a three-email abandoned cart flow. Reference Klaviyo's current interface "
    "specifically. Respond with valid JSON only, in the exact form {\"steps\": [\"string\"]} — no "
    "markdown code fences, no explanation before or after the JSON."
)


def _detect_email_platform(agent1_output: dict) -> str:
    for pillar_data in agent1_output.get("pillar_scores", {}).values():
        for signal in pillar_data.get("stream_b_signals", []):
            lowered = signal.lower()
            if "klaviyo" in lowered:
                return "Klaviyo (already detected on this store)"
            if "shopify email" in lowered:
                return "Shopify Email (already detected on this store)"
    return "No email platform detected; recommend Klaviyo as the default"


class EmailSequenceExecutor(BaseExecutor):
    # full_build mode: writes a complete, brand-voiced three-email abandoned-
    # cart recovery sequence plus setup steps for whatever email platform
    # scrape_website() detected (Klaviyo/Shopify Email) — a real deliverable
    # the business can load in directly, not a generic template.
    def execute(self, profile: dict, opportunity: dict, agent1_output: dict) -> ExecutorResult:
        llm_calls_made = 0
        try:
            company_name = profile.get("company_name") or profile.get("business_name")
            primary_challenge = profile.get("primary_challenge", "Not specified")
            brand_voice = profile.get("brand_voice", "warm, values-led, independent fashion brand voice")
            product_categories = profile.get("product_categories", "womenswear/fashion apparel")

            sequence_result = self.llm.structured_chat(
                system=SEQUENCE_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Business name: {company_name}\n"
                        f"Primary challenge: {primary_challenge}\n"
                        f"Brand voice: {brand_voice}\n"
                        f"Product categories: {product_categories}"
                    ),
                }],
                schema_description=(
                    '{"sequence": [{"email_number": 1, "send_timing": "string", "subject_line": "string", '
                    '"preview_text": "string", "body_copy": "string", "call_to_action": "string"}]}'
                ),
                max_tokens=2000,
            )
            llm_calls_made += 1

            platform = _detect_email_platform(agent1_output)
            setup_result = self.llm.structured_chat(
                system=SETUP_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": f"Detected email platform: {platform}\nWrite the setup steps for the three-email abandoned cart flow.",
                }],
                schema_description='{"steps": ["string"]}',
                max_tokens=1200,
            )
            llm_calls_made += 1

            measurement_framework = {
                "kpis": ["recovery_rate", "revenue_recovered", "unsubscribe_rate"],
                "thresholds": {"recovery_rate_target": "5-15%", "review_after_days": 30},
            }

            deliverables = [
                {
                    "name": "email_sequence",
                    "type": "json",
                    "content": sequence_result,
                    "instructions": "Load these three emails into your email platform's abandoned cart flow.",
                },
                {
                    "name": "setup_instructions",
                    "type": "markdown",
                    "content": setup_result,
                    "instructions": f"Follow these steps in {platform.split(' (')[0]} to activate the flow.",
                },
                {
                    "name": "measurement_framework",
                    "type": "json",
                    "content": measurement_framework,
                    "instructions": "Review these KPIs after 30 days to judge whether the flow is working.",
                },
            ]

            return ExecutorResult(
                mode="full_build",
                status="complete",
                deliverables=deliverables,
                missing_prerequisites=[],
                next_steps=[
                    "Connect your email platform (Klaviyo or Shopify Email) if not already connected",
                    "Create the three-email flow using the sequence JSON below",
                    "Set the send-timing triggers (1 hour, 24 hours, 72 hours after abandonment)",
                    "Turn the flow on and monitor recovery rate for 30 days",
                ],
                estimated_activation_time="this week",
                market_tools_recommended=[],
                llm_calls_made=llm_calls_made,
                error=None,
            )
        except Exception as exc:
            return ExecutorResult(
                mode="full_build",
                status="failed",
                deliverables=[],
                missing_prerequisites=[],
                next_steps=[],
                estimated_activation_time="",
                market_tools_recommended=[],
                llm_calls_made=llm_calls_made,
                error=str(exc),
            )
