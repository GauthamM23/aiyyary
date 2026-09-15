"""Mode 2 partial-build executor: scaffolds the size guide chatbot ahead of the business providing size data."""

from agents.executors.base_executor import BaseExecutor, ExecutorResult

FLOW_SYSTEM_PROMPT = (
    "You are a conversational AI designer building a size guide assistant for a UK fashion "
    "retailer. Design the complete conversation flow — the questions the chatbot asks, the logic "
    "for mapping answers to size recommendations, and the fallback responses when data is "
    "insufficient. Respond with valid JSON only describing the full conversation tree — no "
    "markdown code fences, no explanation before or after the JSON."
)

TEMPLATE_SYSTEM_PROMPT = (
    "You are a fashion retail data specialist. Generate a size data standardisation template the "
    "business owner fills in to power the size guide chatbot. The template must cover UK sizing "
    "conventions for women's fashion, as a CSV string with headers and example rows. Respond with "
    'valid JSON only, in the exact form {"csv_template": "string"} — no markdown code fences, no '
    "explanation before or after the JSON."
)

DEPLOYMENT_GUIDE = (
    "# Deploying the Size Guide Chatbot\n\n"
    "This chatbot's conversation logic is ready now. Full deployment is paused on one thing: "
    "your product size data in a standard format.\n\n"
    "1. Complete `size_data_template.csv` with measurements for each product/size combination.\n"
    "2. Return the completed file — AIYYARY will validate it and wire it into the conversation flow.\n"
    "3. Once connected, embed the chatbot widget on your product pages.\n"
    "4. Monitor size-related return rate for 4-6 weeks after launch to measure impact.\n"
)


class PartialBuildExecutor(BaseExecutor):
    # partial_build mode: builds everything that doesn't depend on data the
    # business hasn't handed over yet (the size-guide chatbot's conversation
    # logic is ready now) and scaffolds the rest (a CSV template + deployment
    # guide) so full activation unblocks the moment the owner returns the
    # filled-in size data — status="partial", not "complete" or "failed".
    def execute(self, profile: dict, opportunity: dict, agent1_output: dict) -> ExecutorResult:
        llm_calls_made = 0
        try:
            company_name = profile.get("company_name") or profile.get("business_name")
            primary_challenge = profile.get("primary_challenge", "Not specified")
            product_categories = profile.get("product_categories", "womenswear/fashion apparel")

            flow_result = self.llm.structured_chat(
                system=FLOW_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Business name: {company_name}\n"
                        f"Product categories: {product_categories}\n"
                        f"Primary challenge: {primary_challenge}"
                    ),
                }],
                schema_description='{"conversation_tree": {"start": {}, "nodes": []}}',
                max_tokens=2000,
            )
            llm_calls_made += 1

            template_result = self.llm.structured_chat(
                system=TEMPLATE_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": f"Generate the UK women's fashion size data standardisation CSV template for {company_name}.",
                }],
                schema_description='{"csv_template": "string"}',
                max_tokens=1200,
            )
            llm_calls_made += 1

            size_data_template = template_result.get("csv_template") if isinstance(template_result, dict) else str(template_result)

            deliverables = [
                {
                    "name": "chatbot_conversation_flow",
                    "type": "json",
                    "content": flow_result,
                    "instructions": "This is the ready-to-deploy conversation logic for the size guide chatbot.",
                },
                {
                    "name": "size_data_template",
                    "type": "csv",
                    "content": size_data_template,
                    "instructions": "Fill this in with your product measurements to activate full deployment.",
                },
                {
                    "name": "deployment_guide",
                    "type": "markdown",
                    "content": DEPLOYMENT_GUIDE,
                    "instructions": "Follow these steps once the size data template is complete.",
                },
            ]

            return ExecutorResult(
                mode="partial_build",
                status="partial",
                deliverables=deliverables,
                missing_prerequisites=[
                    "Complete the size_data_template.csv with your product measurements",
                    "Return the completed template to activate full deployment",
                ],
                next_steps=[
                    "Review the chatbot conversation flow below",
                    "Fill in the size_data_template.csv with your product measurements",
                    "Return the completed template to activate full deployment",
                ],
                estimated_activation_time="next week (after size data is provided)",
                market_tools_recommended=[],
                llm_calls_made=llm_calls_made,
                error=None,
            )
        except Exception as exc:
            return ExecutorResult(
                mode="partial_build",
                status="failed",
                deliverables=[],
                missing_prerequisites=[],
                next_steps=[],
                estimated_activation_time="",
                market_tools_recommended=[],
                llm_calls_made=llm_calls_made,
                error=str(exc),
            )
