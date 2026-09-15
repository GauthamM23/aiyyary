"""Mode 2 full-build executor: builds a deployable returns-reason capture system."""

import json

from agents.executors.base_executor import BaseExecutor, ExecutorResult

REASON_CODES_SYSTEM_PROMPT = (
    "You are a fashion retail operations specialist. Generate a structured set of returns "
    "reason codes for a UK fashion SME. The codes must be specific to fashion (sizing, fabric, "
    "colour, style) not generic e-commerce. Respond with valid JSON only, in the exact form: "
    '{"reason_codes": [{"code": "string", "label": "string", "sub_reasons": ["string"]}]} '
    "— no markdown code fences, no explanation before or after the JSON."
)

FORM_SYSTEM_PROMPT = (
    "You are building a customer-facing returns form for a UK fashion retailer. Generate the "
    "exact question sequence for a returns form that captures structured reason data while "
    "maintaining a warm, brand-appropriate customer experience. Respond with valid JSON only "
    "describing the full form structure — no markdown code fences, no explanation before or "
    "after the JSON."
)

SHOPIFY_SYSTEM_PROMPT = (
    "You are an e-commerce platform integration specialist. Write step-by-step instructions for "
    "embedding a returns reason capture form into this business's returns flow, specific to the "
    "current admin interface of the platform named in the user message. You must write the guide "
    "only for the platform stated in the user message. If the platform is Shopify, write "
    "Shopify-specific instructions. Never reference Wix, WooCommerce, or any other platform unless "
    "that is the platform stated. Respond with valid JSON only, in the exact form "
    '{"instructions": ["string"]} — no markdown code fences, no explanation before or after the JSON.'
)


class ReturnsCaptureExecutor(BaseExecutor):
    # full_build mode: actually builds the returns-reason capture system —
    # fashion-specific reason codes, a customer-facing form structure, and a
    # Shopify embed guide — rather than just recommending one. This is the
    # opportunity most demo companies (Birdsong, Gudrun Sjödén, Nobody's
    # Child) end up executing, since Returns Reason Capture has the lowest
    # prerequisite bar in the knowledge base.
    def execute(self, profile: dict, opportunity: dict, agent1_output: dict) -> ExecutorResult:
        llm_calls_made = 0
        try:
            company_name = profile.get("company_name") or profile.get("business_name")
            product_categories = profile.get("product_categories", "womenswear/fashion apparel")
            primary_challenge = profile.get("primary_challenge", "Not specified")
            brand_voice = profile.get("brand_voice", "warm, friendly, and on-brand for an independent fashion retailer")

            # LLM Call 1 — reason codes (with retry)
            try:
                reason_codes_result = self.llm.structured_chat(
                    system=REASON_CODES_SYSTEM_PROMPT,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Business name: {company_name}\n"
                            f"Product categories: {product_categories}\n"
                            f"Primary challenge: {primary_challenge}"
                        ),
                    }],
                    schema_description='{"reason_codes": [{"code": "string", "label": "string", "sub_reasons": ["string"]}]}',
                    max_tokens=1200,
                )
                llm_calls_made += 1
            except Exception:
                # one retry on transient parse failure
                reason_codes_result = self.llm.structured_chat(
                    system=REASON_CODES_SYSTEM_PROMPT,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Business name: {company_name}\n"
                            f"Product categories: {product_categories}\n"
                            f"Primary challenge: {primary_challenge}"
                        ),
                    }],
                    schema_description='{"reason_codes": [{"code": "string", "label": "string", "sub_reasons": ["string"]}]}',
                    max_tokens=1200,
                )
                llm_calls_made += 1

            # LLM Call 2 — form question sequence (with retry)
            try:
                form_result = self.llm.structured_chat(
                    system=FORM_SYSTEM_PROMPT,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Reason codes:\n{json.dumps(reason_codes_result, indent=2)}\n"
                            f"Brand voice: {brand_voice}"
                        ),
                    }],
                    schema_description='{"form_structure": {"title": "string", "questions": [{"id": "string", "type": "string", "prompt": "string", "options": ["string"]}]}}',
                    max_tokens=1500,
                )
                llm_calls_made += 1
            except Exception:
                # one retry on transient parse failure
                form_result = self.llm.structured_chat(
                    system=FORM_SYSTEM_PROMPT,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Reason codes:\n{json.dumps(reason_codes_result, indent=2)}\n"
                            f"Brand voice: {brand_voice}"
                        ),
                    }],
                    schema_description='{"form_structure": {"title": "string", "questions": [{"id": "string", "type": "string", "prompt": "string", "options": ["string"]}]}}',
                    max_tokens=1500,
                )
                llm_calls_made += 1

            # LLM Call 3 — platform-specific implementation instructions
            platform_detected = agent1_output.get("enrichment_structured", {}).get("platform_detected") or "your website"
            if platform_detected == "custom_or_unknown":
                # enrichment's sentinel for "no known platform markers found" —
                # treat the same as no detection at all, not a literal platform name.
                platform_detected = "your website"
            try:
                shopify_result = self.llm.structured_chat(
                    system=SHOPIFY_SYSTEM_PROMPT,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Platform detected on this company's website: {platform_detected}. "
                            f"Write the integration guide specifically for {platform_detected}. "
                            "If the platform is 'your website', write generic HTML/JavaScript instructions "
                            "that work across any website platform without referencing any specific platform by name. "
                            "Do not reference any other platform than the one stated. "
                            "Write the embed instructions for a returns reason capture form."
                        ),
                    }],
                    schema_description='{"instructions": ["string"]}',
                    max_tokens=1200,
                )
                llm_calls_made += 1
            except Exception:
                # one retry on transient parse failure
                shopify_result = self.llm.structured_chat(
                    system=SHOPIFY_SYSTEM_PROMPT,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Platform detected on this company's website: {platform_detected}. "
                            f"Write the integration guide specifically for {platform_detected}. "
                            "If the platform is 'your website', write generic HTML/JavaScript instructions "
                            "that work across any website platform without referencing any specific platform by name. "
                            "Do not reference any other platform than the one stated. "
                            "Write the embed instructions for a returns reason capture form."
                        ),
                    }],
                    schema_description='{"instructions": ["string"]}',
                    max_tokens=1200,
                )
                llm_calls_made += 1

            weekly_summary_template = (
                f"# Weekly Returns Summary — {company_name}\n\n"
                "Subject: Your weekly returns snapshot\n\n"
                "Hi team,\n\n"
                "Here is this week's returns-reason breakdown:\n\n"
                "- Top return reason: {{top_reason}} ({{top_reason_pct}}% of returns)\n"
                "- Total returns this week: {{total_returns}}\n"
                "- Week-over-week change: {{wow_change}}\n\n"
                "Recommended action: review whether {{top_reason}} points to a sizing, description, "
                "or quality fix on the affected product(s).\n\n"
                "— Generated by AIYYARY"
            )

            deliverables = [
                {
                    "name": "returns_reason_codes",
                    "type": "json",
                    "content": reason_codes_result,
                    "instructions": "Import these reason codes into your returns form or helpdesk tool.",
                },
                {
                    "name": "customer_form_structure",
                    "type": "json",
                    "content": form_result,
                    "instructions": "Copy this structure into Typeform, Google Forms, or your returns portal.",
                },
                {
                    "name": "shopify_integration_guide",
                    "type": "markdown",
                    "content": shopify_result,
                    "instructions": "Follow these steps to embed the form in your Shopify returns flow.",
                },
                {
                    "name": "weekly_summary_template",
                    "type": "markdown",
                    "content": weekly_summary_template,
                    "instructions": "Use this template for a weekly email reviewing returns data.",
                },
            ]

            return ExecutorResult(
                mode="full_build",
                status="complete",
                deliverables=deliverables,
                missing_prerequisites=[],
                next_steps=[
                    "Create Typeform account at typeform.com",
                    "Copy the form structure JSON below into a new form",
                    f"Add the embed code to your returns page on {platform_detected}",
                    "Set up a weekly calendar reminder to review the summary report",
                ],
                estimated_activation_time="today",
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
