# AIYYARY

AIYYARY is a three-agent pipeline that assesses a UK fashion/apparel SME's digital capability, researches viable AI opportunities for it, and produces a risk-aware transformation roadmap — with concrete executor deliverables (email sequences, returns forms, market-connector briefs, partial builds) for the top-ranked opportunity.

## Pipeline

1. **Enrichment** (`agents/enrichment.py`) — live public-data gathering (website scrape, Google Business Profile, Companies House, Facebook Ad Library, Instagram), mapped to capability-pillar signals (Stream B).
2. **Agent 1 — Capability Assessment** (`agents/agent1_capability.py`) — synthesises Stream A (owner-interview instrument, 60%), Stream B (live enrichment, 30%), and Stream C (LLM reconciliation on divergence, 10%) into a scored assessment across 6 pillars (data foundations, process digitisation, technology infrastructure, staff digital literacy, governance & compliance, AI tool experience).
3. **Agent 2 — AI Opportunity Research** (`agents/agent2_opportunity.py`) — semantic retrieval over the fashion AI use-case knowledge base (ChromaDB), a deterministic prerequisite filter, LLM justification per survivor, and executor dispatch for the top-ranked opportunities (`agents/executors/`).
4. **Agent 3 — Risk-Aware Transformation** (`agents/agent3_risk.py`) — deterministic risk register against a risk taxonomy, LLM mitigations, a phased roadmap, and an ROI summary.

## Setup

```
pip install -r requirements.txt
```

Configure `.env` with:

- `ANTHROPIC_API_KEY`, `GROQ_API_KEY` — LLM calls (`LLM_MODE` selects provider)
- `GOOGLE_PLACES_API_KEY`, `COMPANIES_HOUSE_API_KEY` — enrichment data sources
- `OUTPUT_DIR`, `CACHE_DIR`, `CACHE_TTL_SECONDS`, `ENRICHMENT_TIMEOUT_SECONDS`, `CHROMA_DB_PATH`, `EXECUTE_TOP_N` — runtime tuning

Verify API keys independently of the pipeline:

```
python verify_keys.py
```

## Operations

### Run the full pipeline for any business

```
python main.py --company "Business Name" --url "https://their-website.com"
python main.py --company "Business Name" --url "https://their-website.com" --answers answers.json
python main.py --company "Business Name" --url "https://their-website.com" --interactive
python main.py --company "Business Name" --url "https://their-website.com" --challenge "Their primary business challenge"
```

Runs Agent 1 → Agent 2 → Agent 3 sequentially, prints the rank-1 opportunity's executor deliverables, and saves JSON output for each agent to `outputs/`.

- `--answers` — path to a JSON file of pre-filled Stream A scores (4 scores per pillar)
- `--interactive` — prompt for all 24 Stream A instrument answers (1–9 scale) in the terminal
- with neither flag, a flat mid-range default (all scores = 5) is used, with a warning

### Live evaluation UI (Streamlit)

```
streamlit run ui/live_eval.py
```

Runs the same Agent 1/2/3 classes in-process for a live session with a real business owner, in the browser.

### Demo runners (five pre-defined UK fashion SMEs)

```
python demo_five_companies.py                    # Agent 1 across all five companies
python demo_five_companies.py --company "Lucy & Yak"

python demo_agent2.py                             # Agent 2 across all five (reuses latest Agent 1 output per company)

python demo_agent3.py                             # Agent 3 across all five (reuses latest Agent 1 + 2 output per company)
python demo_agent3.py --company "Birdsong London"
```

The five demo companies: Lucy & Yak, Nobody's Child, Birdsong London, Baukjen, Gudrun Sjödén UK.

### View executor deliverables (read-only report)

```
python show_deliverables.py
python show_deliverables.py --company "Baukjen"
python show_deliverables.py --company "Nobody's Child" --all-ranks
python show_deliverables.py --company "Baukjen" --show-roadmap
```

Reads the most recent `agent2_*.json` (and optionally `agent3_*.json`) files from `outputs/` — never re-runs an agent.

### Cross-validation (dissertation results)

```
python cross_validate.py                # full run incl. LLM findings
python cross_validate.py --no-llm       # skip LLM findings generation
python cross_validate.py --summary-only # print only summary + Agent 2 confirmation
python cross_validate.py --export-table # write only the markdown table, no terminal report
```

Compares researcher-constructed Stream-A-only scores (`test_agent1.py`) against the live Stream A + B + C synthesis, and writes `outputs/cross_validation_report.json` and `outputs/cross_validation_table.md`.

### Tests

```
python test_agent1.py                                                          # deterministic Stream A scoring tests
python test_agent1.py --enrich --url https://example.co.uk --name "Example Boutique" --location "London"

python test_agent2.py                                                          # deterministic prerequisite-filter tests
python test_agent2.py --live                                                   # full pipeline incl. ChromaDB + LLM call

python test_agent3.py                                                          # deterministic risk-severity tests
python test_agent3.py --live                                                   # full pipeline against Lucy & Yak's latest output
```

## Project structure

```
agents/            Agent 1/2/3, enrichment, LLM client, executors (email, returns form, market connector, partial build)
knowledge_base/    Fashion AI use-case knowledge base (source for Agent 2's ChromaDB index)
taxonomy/          AI agent marketplace taxonomy and risk taxonomy (source for Agent 3)
profiles/          Company profile definitions
outputs/           Agent output JSONs, ChromaDB store, cross-validation report/table
ui/                Streamlit live-evaluation UI
```
