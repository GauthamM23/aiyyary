"""Agent 2: AI Opportunity Research Agent (with integrated Executor layer) for AIYYARY.

Consumes Agent 1's capability assessment output directly. Four-plus-one stage pipeline:

  Stage 1 — Load the fashion AI use-case knowledge base and index it in ChromaDB.
  Stage 2 — Semantic retrieval: pull the 8 most relevant use cases for this business.
  Stage 3 — Deterministic prerequisite filter (pure Python, no LLM) — survivors vs excluded.
  Stage 4 — LLM justification call: one sentence per survivor/deferred item, no re-ranking.
  Stage 5 — Executor dispatch: actually activate the top-ranked executable survivors.

Output is the exact schema Agent 3 consumes directly.
"""

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone

from dotenv import load_dotenv

from agents.executors.base_executor import ExecutorResult
from agents.executors.email_sequence_executor import EmailSequenceExecutor
from agents.executors.market_connector import MarketConnector
from agents.executors.partial_build_executor import PartialBuildExecutor
from agents.executors.returns_form_executor import ReturnsCaptureExecutor
from agents.llm_client import LLMClient

load_dotenv()

KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base", "fashion_use_cases.json")
CHROMA_COLLECTION_NAME = "fashion_opportunities"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

EFFORT_ORDER = {"low": 0, "medium": 1, "high": 2}

EXECUTOR_REGISTRY = {
    "MarketConnector": MarketConnector,
    "ReturnsCaptureExecutor": ReturnsCaptureExecutor,
    "EmailSequenceExecutor": EmailSequenceExecutor,
    "PartialBuildExecutor": PartialBuildExecutor,
}

JUSTIFICATION_SYSTEM_PROMPT = """You are AIYYARY's opportunity research analyst for UK fashion SMEs.
You will be given a list of AI opportunities already filtered and ranked by a
deterministic scoring system. Do NOT re-rank or add new ones.

For each RECOMMENDED opportunity write ONE sentence explaining why it is
specifically right for this business right now, referencing their actual
pillar score for the relevant prerequisite and their primary challenge.

For each DEFERRED opportunity write ONE sentence explaining the specific
score gap preventing deployment and what needs to improve first.

Also write a quick_win_summary: one sentence identifying the single highest
priority action for this business to take this week. quick_win_summary must
always reference the rank 1 opportunity by its exact title and ID. Do not
reference any other opportunity.

Respond with JSON only:
{
  "recommended": [
    {
      "rank": 1,
      "id": "string",
      "title": "string",
      "justification": "string",
      "effort": "string",
      "implementation_weeks": 0,
      "cost_estimate_gbp": "string",
      "executor_mode": "string",
      "executable": true
    }
  ],
  "deferred": [
    {
      "id": "string",
      "title": "string",
      "reason": "string",
      "unlock_condition": "string — what score on which pillar needs to reach what level"
    }
  ],
  "quick_win_summary": "string"
}"""


def load_knowledge_base(path: str = KNOWLEDGE_BASE_PATH) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# The deterministic gatekeeper that makes AIYYARY's recommendations business-
# specific rather than generic: every AI use case in the knowledge base has
# a minimum pillar score requirement, and this function is the only place
# that decides survives vs excluded — no LLM involved, so it can never be
# talked into recommending something the business isn't actually ready for.
# Its output (survivors/excluded) is what the LLM justification call and the
# executor dispatch stage both operate on downstream.
def filter_by_prerequisites(opportunities: list, agent1_output: dict) -> tuple:
    """Pure-Python deterministic prerequisite filter — no LLM, no ChromaDB required.

    Returns (survivors, excluded), both lists of dicts. Survivors are sorted by
    effort (low -> medium -> high) then by phase. Excluded entries carry the
    business's actual_score and the score_gap still to close.
    """
    pillar_scores = agent1_output["pillar_scores"]
    survivors = []
    excluded = []

    for opportunity in opportunities:
        pillar = opportunity["requires_pillar"]
        min_required = opportunity["min_score_required"]
        actual_score = pillar_scores[pillar]["final_score"]

        if actual_score >= min_required:
            survivors.append(opportunity)
        else:
            excluded.append({
                **opportunity,
                "actual_score": actual_score,
                "score_gap": round(min_required - actual_score, 2),
            })

    survivors.sort(key=lambda o: (EFFORT_ORDER.get(o.get("effort", "high"), 2), o.get("phase", 99)))
    excluded.sort(key=lambda o: o["score_gap"])
    return survivors, excluded


class OpportunityResearchAgent:
    """Orchestrates the full Agent 2 pipeline: retrieval, filtering, justification, execution."""

    _embedder = None  # lazy-loaded, shared across instances so the model loads once per process

    def __init__(self):
        self.llm = LLMClient()
        self.execute_top_n = int(os.getenv("EXECUTE_TOP_N", "2"))
        self.chroma_db_path = os.getenv("CHROMA_DB_PATH", "./outputs/chroma_db")
        self.knowledge_base = load_knowledge_base()
        self.use_cases_by_id = {uc["id"]: uc for uc in self.knowledge_base}
        self._collection = None

    # ------------------------------------------------------------------
    # Stage 1 — knowledge base + ChromaDB index
    # ------------------------------------------------------------------
    @classmethod
    def _get_embedder(cls):
        if cls._embedder is None:
            from sentence_transformers import SentenceTransformer

            cls._embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return cls._embedder

    def _get_collection(self):
        if self._collection is not None:
            return self._collection

        import chromadb

        os.makedirs(self.chroma_db_path, exist_ok=True)
        client = chromadb.PersistentClient(path=self.chroma_db_path)
        collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

        if collection.count() == 0:
            embedder = self._get_embedder()
            documents = [uc["description"] for uc in self.knowledge_base]
            ids = [uc["id"] for uc in self.knowledge_base]
            metadatas = [
                {"title": uc["title"], "requires_pillar": uc["requires_pillar"], "effort": uc["effort"]}
                for uc in self.knowledge_base
            ]
            embeddings = embedder.encode(documents).tolist()
            collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

        self._collection = collection
        return collection

    # ------------------------------------------------------------------
    # Stage 2 — semantic retrieval
    # ------------------------------------------------------------------
    # Narrows the 10-entry knowledge base down to the 8 most relevant use
    # cases for this business before the deterministic filter runs, using
    # ChromaDB embedding similarity over company name, primary challenge, and
    # readiness tier. This is a relevance pre-filter only — it never decides
    # what the business can actually deploy, that's filter_by_prerequisites().
    def semantic_retrieval(self, profile: dict, agent1_output: dict, n_results: int = 8) -> list:
        collection = self._get_collection()
        embedder = self._get_embedder()

        company_name = profile.get("company_name") or profile.get("business_name")
        query = f"{company_name} {profile.get('primary_challenge', '')} fashion retail {agent1_output['readiness_tier']}"
        query_embedding = embedder.encode([query]).tolist()

        n_results = min(n_results, len(self.knowledge_base))
        results = collection.query(query_embeddings=query_embedding, n_results=n_results)
        retrieved_ids = results.get("ids", [[]])[0]

        return [self.use_cases_by_id[uid] for uid in retrieved_ids if uid in self.use_cases_by_id]

    # ------------------------------------------------------------------
    # Stage 4 — LLM justification call
    # ------------------------------------------------------------------
    def _call_justification(self, profile: dict, agent1_output: dict, survivors: list, excluded: list) -> dict:
        schema_description = (
            '{"recommended": [{"rank": 1, "id": "string", "title": "string", "justification": "string", '
            '"effort": "string", "implementation_weeks": 0, "cost_estimate_gbp": "string", '
            '"executor_mode": "string", "executable": true}], '
            '"deferred": [{"id": "string", "title": "string", "reason": "string", "unlock_condition": "string"}], '
            '"quick_win_summary": "string"}'
        )
        survivors_summary = [
            {
                "rank": i + 1,
                "id": o["id"],
                "title": o["title"],
                "requires_pillar": o["requires_pillar"],
                "actual_score": agent1_output["pillar_scores"][o["requires_pillar"]]["final_score"],
                "effort": o["effort"],
            }
            for i, o in enumerate(survivors)
        ]
        excluded_summary = [
            {
                "id": o["id"],
                "title": o["title"],
                "requires_pillar": o["requires_pillar"],
                "actual_score": o["actual_score"],
                "min_score_required": o["min_score_required"],
                "score_gap": o["score_gap"],
            }
            for o in excluded
        ]
        company_name = profile.get("company_name") or profile.get("business_name")
        user_message = (
            f"Company: {company_name}\n"
            f"Primary challenge: {profile.get('primary_challenge', 'Not specified')}\n"
            f"Readiness tier: {agent1_output['readiness_tier']}\n"
            f"RECOMMENDED (already filtered and ranked, do not re-rank):\n{json.dumps(survivors_summary, indent=2)}\n"
            f"DEFERRED (excluded by the prerequisite filter):\n{json.dumps(excluded_summary, indent=2)}\n"
            "Write the justification, reason, unlock_condition, and quick_win_summary as instructed."
        )

        try:
            result = self.llm.structured_chat(
                system=JUSTIFICATION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
                schema_description=schema_description,
                max_tokens=2000,
            )
            return result, 1
        except Exception as exc:
            fallback_recommended = [
                {
                    "rank": i + 1,
                    "id": o["id"],
                    "title": o["title"],
                    "justification": (
                        f"[LLM justification call failed ({exc})]; deterministically survived because "
                        f"{o['requires_pillar']} score "
                        f"{agent1_output['pillar_scores'][o['requires_pillar']]['final_score']} "
                        f">= required {o['min_score_required']}."
                    ),
                    "effort": o["effort"],
                    "implementation_weeks": o["implementation_weeks"],
                    "cost_estimate_gbp": o["cost_estimate_gbp"],
                    "executor_mode": o["executor_mode"],
                    "executable": o["executable"],
                }
                for i, o in enumerate(survivors)
            ]
            fallback_deferred = [
                {
                    "id": o["id"],
                    "title": o["title"],
                    "reason": f"[LLM justification call failed ({exc})]; score gap of {o['score_gap']} on {o['requires_pillar']}.",
                    "unlock_condition": f"{o['requires_pillar']} needs to reach {o['min_score_required']} (currently {o['actual_score']}).",
                }
                for o in excluded
            ]
            fallback = {
                "recommended": fallback_recommended,
                "deferred": fallback_deferred,
                "quick_win_summary": f"[LLM justification call failed ({exc})]; see top-ranked recommended item.",
            }
            return fallback, 0

    # ------------------------------------------------------------------
    # Stage 5 — executor dispatch
    # ------------------------------------------------------------------
    # This is what turns AIYYARY from a recommendation engine into a builder:
    # for the top execute_top_n executable survivors, it actually runs the
    # matching executor (full_build / market_connect / partial_build) and
    # produces real deliverables, not just a suggestion. The results here
    # become agent2_output["recommended"][i]["executor_result"], which Agent
    # 3's phase-structure builder later reads to decide what Phase 2
    # ("activate what's already built") contains.
    def _dispatch_executors(self, profile: dict, survivors: list, agent1_output: dict) -> dict:
        executor_results = {}
        executed_count = 0

        for opportunity in survivors:
            if not opportunity.get("executable"):
                continue
            if executed_count >= self.execute_top_n:
                break

            executor_class_name = opportunity.get("executor_class")
            executor_class = EXECUTOR_REGISTRY.get(executor_class_name)
            if executor_class is None:
                continue

            executor = executor_class(self.llm)
            try:
                result = executor.execute(profile, opportunity, agent1_output)
            except Exception as exc:
                result = ExecutorResult(
                    mode=opportunity.get("executor_mode", "unknown"),
                    status="failed",
                    error=str(exc),
                )

            executor_results[opportunity["id"]] = result
            executed_count += 1

        return executor_results

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
        filename = f"agent2_{safe_company}_{timestamp}.json"
        return os.path.join(output_dir, filename)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    # Main entry point for Agent 2 — takes Agent 1's output directly (never
    # re-scores anything) and runs it through retrieval, the deterministic
    # filter, LLM justification, and executor dispatch in that fixed order.
    # Writes agent2_*.json to outputs/; its "recommended"/"deferred"/
    # "executor_summary" fields are the exact contract Agent 3 consumes next
    # to build the risk-aware roadmap.
    def run(self, profile: dict, agent1_output: dict, execute_top_n: int = None) -> dict:
        if execute_top_n is not None:
            self.execute_top_n = execute_top_n

        company_name = profile.get("company_name") or profile.get("business_name")

        # Stage 2 — semantic retrieval
        retrieved = self.semantic_retrieval(profile, agent1_output, n_results=8)

        # Stage 3 — deterministic prerequisite filter
        survivors, excluded = filter_by_prerequisites(retrieved, agent1_output)

        # Stage 4 — LLM justification
        justification_data, justification_llm_calls = self._call_justification(profile, agent1_output, survivors, excluded)

        # Stage 5 — executor dispatch (top execute_top_n executable survivors only)
        executor_results = self._dispatch_executors(profile, survivors, agent1_output)

        # ------------------------------------------------------------------
        # Assemble recommended list: deterministic survivor order is the source
        # of truth for ranking/effort/cost/executor_mode; the LLM only supplies
        # the justification prose.
        # ------------------------------------------------------------------
        justification_by_id = {r["id"]: r for r in justification_data.get("recommended", []) if "id" in r}
        recommended = []
        executed_count = 0
        for i, opportunity in enumerate(survivors):
            rank = i + 1
            just = justification_by_id.get(opportunity["id"], {})
            executor_result = executor_results.get(opportunity["id"])
            is_executed = executor_result is not None
            if is_executed:
                executed_count += 1

            entry = {
                "rank": rank,
                "id": opportunity["id"],
                "title": opportunity["title"],
                "justification": just.get("justification", ""),
                "effort": opportunity["effort"],
                "implementation_weeks": opportunity["implementation_weeks"],
                "cost_estimate_gbp": opportunity["cost_estimate_gbp"],
                "executor_mode": opportunity["executor_mode"],
                "executable": opportunity["executable"],
                "executor_result": asdict(executor_result) if is_executed else None,
                "executable_on_request": bool(opportunity.get("executable")) and not is_executed,
            }
            recommended.append(entry)

        reason_by_id = {d["id"]: d for d in justification_data.get("deferred", []) if "id" in d}
        deferred = []
        for opportunity in excluded:
            reason = reason_by_id.get(opportunity["id"], {})
            deferred.append({
                "id": opportunity["id"],
                "title": opportunity["title"],
                "reason": reason.get("reason", ""),
                "unlock_condition": reason.get(
                    "unlock_condition",
                    f"{opportunity['requires_pillar']} needs to reach {opportunity['min_score_required']} "
                    f"(currently {opportunity['actual_score']}).",
                ),
            })

        # Executor summary
        full_builds_completed = sum(1 for r in executor_results.values() if r.mode == "full_build" and r.status == "complete")
        market_connections_made = sum(1 for r in executor_results.values() if r.mode == "market_connect" and r.status == "connected")
        partial_builds = sum(1 for r in executor_results.values() if r.mode == "partial_build")
        total_executor_llm_calls = sum(r.llm_calls_made for r in executor_results.values())

        output_file = self._output_filepath(company_name)

        result = {
            "agent": "Agent2_OpportunityResearch",
            "schema_version": "2.0",
            "assessment_timestamp": datetime.now(timezone.utc).isoformat(),
            "company_name": company_name,
            "readiness_tier": agent1_output["readiness_tier"],
            "overall_score": agent1_output["overall_score"],
            "retrieval_method": "chromadb_semantic + prerequisite_filter",
            "survivors_count": len(survivors),
            "excluded_count": len(excluded),
            "recommended": recommended,
            "deferred": deferred,
            "quick_win_summary": justification_data.get("quick_win_summary", ""),
            "executor_summary": {
                "total_executors_run": len(executor_results),
                "full_builds_completed": full_builds_completed,
                "market_connections_made": market_connections_made,
                "partial_builds": partial_builds,
                "total_executor_llm_calls": total_executor_llm_calls,
            },
            "traceability": {
                "knowledge_base_entries_total": len(self.knowledge_base),
                "semantically_retrieved": len(retrieved),
                "survived_filter": len(survivors),
                "excluded_by_filter": len(excluded),
                "llm_calls_made": justification_llm_calls,
                "output_file": output_file,
            },
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved output to: {output_file}")

        return result
