"""Standalone reporter: reads the most recent Agent 2 output JSONs from
outputs/ and prints the full executor deliverable content for each company
in a structured terminal report.

Read-only with respect to the rest of the project — it never re-runs any
agent, it only reads whatever agent2_*.json files already exist in outputs/.

Usage:
    python show_deliverables.py
    python show_deliverables.py --company "Baukjen"
    python show_deliverables.py --company "Nobody's Child" --all-ranks
    python show_deliverables.py --company "Baukjen" --show-roadmap
"""

import argparse
import datetime
import glob
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
LINE_WIDTH = 70
HEAVY = "═" * LINE_WIDTH
LIGHT = "─" * LINE_WIDTH


# ---------------------------------------------------------------------------
# Loading — never re-runs an agent, only reads what's already in outputs/
# ---------------------------------------------------------------------------


def find_latest_agent2_output(company_name: str):
    """Return (path, data) for the agent2_*.json in outputs/ belonging to
    company_name (matched by the company_name field inside the JSON, not by
    filename) with the most recent assessment_timestamp. (None, None) if no
    file exists or none parse."""
    pattern = os.path.join(OUTPUT_DIR, "agent2_*.json")
    candidates = []
    for path in glob.glob(pattern):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("company_name") != company_name:
            continue
        candidates.append((data.get("assessment_timestamp", ""), path, data))

    if not candidates:
        return None, None
    candidates.sort(key=lambda c: c[0])
    _, path, data = candidates[-1]
    return path, data


def find_latest_agent3_output(company_name: str):
    """Return (path, data) for the agent3_*.json in outputs/ belonging to
    company_name (matched by the company_name field inside the JSON, not by
    filename) with the most recent assessment_timestamp. (None, None) if no
    file exists or none parse."""
    pattern = os.path.join(OUTPUT_DIR, "agent3_*.json")
    candidates = []
    for path in glob.glob(pattern):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("company_name") != company_name:
            continue
        candidates.append((data.get("assessment_timestamp", ""), path, data))

    if not candidates:
        return None, None
    candidates.sort(key=lambda c: c[0])
    _, path, data = candidates[-1]
    return path, data


def discover_companies() -> list:
    """Scan outputs/ for every agent2_*.json and return the distinct
    company_name values found inside them (not by filename), sorted
    alphabetically. Reflects whatever has actually been run, not a fixed
    demo list."""
    pattern = os.path.join(OUTPUT_DIR, "agent2_*.json")
    names = set()
    for path in glob.glob(pattern):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        name = data.get("company_name")
        if name:
            names.add(name)
    return sorted(names)


def resolve_companies(company_arg: str, known_companies: list) -> list:
    """Case-insensitive partial match against the discovered companies."""
    if not company_arg:
        return list(known_companies)
    needle = company_arg.strip().lower()
    return [c for c in known_companies if needle in c.lower()]


# ---------------------------------------------------------------------------
# Small rendering helpers
# ---------------------------------------------------------------------------


def _is_empty_content(content) -> bool:
    if content is None:
        return True
    if isinstance(content, (str, list, dict)) and len(content) == 0:
        return True
    return False


def _text_from_content(content) -> str:
    """Best-effort plain-text rendering for a deliverable's content field,
    which may be a plain string or a dict the LLM returned (e.g. {"steps": [...]}
    or {"instructions": [...]})."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if isinstance(content.get("instructions"), list):
            return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(content["instructions"]))
        if isinstance(content.get("steps"), list):
            return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(content["steps"]))
        return json.dumps(content, indent=2, ensure_ascii=False)
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content)


def _section_header(label: str) -> str:
    prefix = f"── {label} "
    pad = max(0, LINE_WIDTH - len(prefix))
    return prefix + "─" * pad


# ---------------------------------------------------------------------------
# Per-deliverable-type renderers
# ---------------------------------------------------------------------------


def _render_reason_codes(content) -> None:
    codes = content.get("reason_codes", []) if isinstance(content, dict) else []
    if not codes:
        # Structured LLM output doesn't always land in the exact requested
        # schema — content is real, just not shaped as expected, so show it
        # rather than falsely claiming it's missing.
        print(_text_from_content(content))
        return
    for rc in codes:
        print(f"  [{rc.get('code', '?')}] {rc.get('label', '')}")
        subs = rc.get("sub_reasons") or []
        if subs:
            print(f"     Sub-reasons: {', '.join(str(s) for s in subs)}")


def _render_form_structure(content) -> None:
    form = content.get("form_structure", content) if isinstance(content, dict) else {}
    questions = form.get("questions", []) if isinstance(form, dict) else []
    if not questions:
        # As above — the LLM sometimes returns a differently-shaped form
        # definition (e.g. a flat "options" list instead of "questions").
        # Fall back to a raw dump instead of hiding real content.
        print(_text_from_content(content))
        return
    for idx, q in enumerate(questions, start=1):
        print(f"  Q{idx}. {q.get('prompt', '')}")
        print(f"      Type: {q.get('type', '')}")
        options = q.get("options")
        if options:
            print(f"      Options: {', '.join(str(o) for o in options)}")


def _render_email_sequence(content) -> None:
    emails = content.get("sequence", []) if isinstance(content, dict) else []
    if not emails:
        print(_text_from_content(content))
        return
    for e in emails:
        print(f"  --- Email {e.get('email_number', '?')} | Send: {e.get('send_timing', '')} ---")
        print(f"  Subject:  {e.get('subject_line', '')}")
        print(f"  Preview:  {e.get('preview_text', '')}")
        print("  Body:")
        print(f"  {e.get('body_copy', '')}")
        print(f"  CTA: {e.get('call_to_action', '')}")
        print()


def _render_measurement(content) -> None:
    if not isinstance(content, dict) or not (content.get("kpis") or content.get("thresholds")):
        print(_text_from_content(content))
        return
    kpis = content.get("kpis") or []
    thresholds = content.get("thresholds") or {}
    print(f"  KPIs: {' | '.join(str(k) for k in kpis)}")
    other = {k: v for k, v in thresholds.items() if k != "review_after_days"}
    print(f"  Target: {', '.join(f'{k}={v}' for k, v in other.items())}")
    print(f"  Review after: {thresholds.get('review_after_days', '?')} days")


def _render_csv(content) -> None:
    text = content if isinstance(content, str) else str(content)
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        print("  [Content not available — re-run pipeline to regenerate]")
        return
    for line in lines[:5]:
        cells = [c.strip() for c in line.split(",")]
        print("  " + " | ".join(cells))


def render_deliverable(name: str, dtype: str, content) -> None:
    if _is_empty_content(content):
        print("  [Content not available — re-run pipeline to regenerate]")
        return

    name_l = (name or "").lower()
    dtype_l = (dtype or "").lower()

    if dtype_l == "json" and "reason_codes" in name_l:
        _render_reason_codes(content)
    elif dtype_l == "json" and ("form_structure" in name_l or "question" in name_l):
        _render_form_structure(content)
    elif dtype_l == "json" and ("email" in name_l or "sequence" in name_l):
        _render_email_sequence(content)
    elif dtype_l == "json" and ("measurement" in name_l or "kpi" in name_l):
        _render_measurement(content)
    elif dtype_l == "csv":
        _render_csv(content)
    elif dtype_l in ("markdown", "text", "plain", "plain_text"):
        print()
        print(_text_from_content(content))
        print()
    else:
        # Unmatched json content (e.g. a chatbot conversation tree) — fall
        # back to a readable dump rather than skipping it.
        print(_text_from_content(content))


def render_market_connect_block(executor_result: dict, deliverables: list) -> None:
    """market_connect opportunities rank real third-party tools rather than
    build an artifact — this renders the ranked tools plus the (shared)
    configuration guide as one consolidated block instead of one near-
    identical DELIVERABLE section per tool."""
    tools = executor_result.get("market_tools_recommended") or []
    if not tools:
        print("  [Content not available — re-run pipeline to regenerate]")
        return
    for t in tools:
        print(f"  Rank {t.get('rank', '?')}: {t.get('tool_name', 'Unknown tool')}")
        print(f"          Why for this business: {t.get('why_this_business', '')}")
        free = "yes" if t.get("free_tier_available") else "no"
        print(f"          Free tier: {free}  |  Est. cost: {t.get('estimated_monthly_cost_gbp', 'n/a')}")
    guide = deliverables[0].get("content") if deliverables else None
    print()
    if _is_empty_content(guide):
        print("  [Configuration guide not available — re-run pipeline to regenerate]")
    else:
        print(_text_from_content(guide))


# ---------------------------------------------------------------------------
# Per-opportunity and per-company rendering
# ---------------------------------------------------------------------------


def print_opportunity(entry: dict) -> None:
    title = entry.get("title", "Unknown")
    opp_id = entry.get("id", "?")
    executor_mode = entry.get("executor_mode", "?")

    print(f"Opportunity:   {title} ({opp_id})")
    print(f"Executor mode: {executor_mode}")

    executor_result = entry.get("executor_result")
    if executor_result is None:
        print("Status:        not dispatched")
        print("Activation:    n/a")
        print(f"Justification: {entry.get('justification', '')}")
        print()
        print("  [Recommended but not executed — outside EXECUTE_TOP_N for this run]")
        print()
        return

    status = executor_result.get("status", "unknown")
    print(f"Status:        {status}")
    print(f"Activation:    {executor_result.get('estimated_activation_time') or 'n/a'}")
    print(f"Justification: {entry.get('justification', '')}")
    print()

    if status == "failed" or executor_result.get("error"):
        print(f"  [EXECUTOR FAILED] {executor_result.get('error', 'Unknown error')}")
        print()
        return

    deliverables = executor_result.get("deliverables") or []

    if executor_mode == "market_connect":
        print(_section_header("DELIVERABLES: Market Tool Recommendations (market_connect)"))
        print()
        try:
            render_market_connect_block(executor_result, deliverables)
        except Exception as exc:
            print(f"  [Error rendering deliverable: {exc}]")
        print()
    elif not deliverables:
        print("  [No deliverables were produced for this opportunity]")
        print()
    else:
        n = len(deliverables)
        for i, d in enumerate(deliverables, start=1):
            name = d.get("name", "unknown")
            dtype = d.get("type", "unknown")
            print(_section_header(f"DELIVERABLE {i} OF {n}: {name} ({dtype})"))
            try:
                render_deliverable(name, dtype, d.get("content"))
            except Exception as exc:
                print(f"  [Error rendering deliverable: {exc}]")
            print()

    next_steps = executor_result.get("next_steps") or []
    if next_steps:
        print(_section_header("NEXT STEPS"))
        for i, step in enumerate(next_steps, start=1):
            print(f"{i}. {step}")
        print()

    missing = executor_result.get("missing_prerequisites") or []
    if missing:
        print(_section_header("MISSING PREREQUISITES"))
        for item in missing:
            print(f"• {item}")
        print()


def print_company(data: dict, show_all_ranks: bool) -> None:
    print(LIGHT)
    print(f"COMPANY: {data.get('company_name', 'Unknown')}   "
          f"Score: {data.get('overall_score', 'n/a')}   "
          f"Tier: {data.get('readiness_tier', 'n/a')}")
    print(LIGHT)

    recommended = data.get("recommended") or []
    if show_all_ranks:
        opportunities = recommended
    else:
        opportunities = [o for o in recommended if o.get("rank") == 1] or recommended[:1]

    if not opportunities:
        print("  [No recommended opportunities in this Agent 2 output]")
        print()
    for entry in opportunities:
        try:
            print_opportunity(entry)
        except Exception as exc:
            print(f"  [Error rendering opportunity: {exc}]")
            print()

    deferred = data.get("deferred") or []
    if deferred:
        print(_section_header("DEFERRED OPPORTUNITIES"))
        for d in deferred:
            print(f"• {d.get('title', 'Unknown')}: {d.get('unlock_condition', '')}")
        print()

    print(_section_header("QUICK WIN"))
    print(data.get("quick_win_summary", "") or "(none)")
    print()


def print_risk_and_roadmap(data: dict) -> None:
    """Renders Agent 3's risk register, phased roadmap, and headline ROI for
    one company. Reads agent3_*.json fields directly — never re-derives a
    score or phase from Agent 1/2 output."""
    print(_section_header("RISK REGISTER"))
    risk_register = data.get("risk_register") or []
    if not risk_register:
        print("  [No risk register available]")
    for r in risk_register:
        severity = r.get("severity", "?")
        print(f"[{severity:<6s}] {r.get('label', 'Unknown')}")
    print()

    print(_section_header("PHASED ROADMAP"))
    roadmap = data.get("roadmap") or []
    if not roadmap:
        print("  [No roadmap available]")
    for phase in roadmap:
        print(f"Phase {phase.get('phase', '?')}: {phase.get('title', 'Unknown')} — {phase.get('timeframe_weeks', 0)} weeks")
        description = phase.get("description", "")
        if description:
            print(f"  {description}")
        agents_list = phase.get("recommended_ai_agents") or []
        if agents_list:
            agent_names = " · ".join(a.get("name", "Unknown") for a in agents_list)
            print(f"  Agents: {agent_names}")
        else:
            print("  Agents: (none recommended)")
        print()

    print(f"Total: {data.get('total_roadmap_weeks', 0)} weeks across {len(roadmap)} phases")
    print()

    print(_section_header("HEADLINE ROI"))
    print(data.get("headline_metric", "") or "(none)")
    print(HEAVY)


# ---------------------------------------------------------------------------
# Cross-company summary table
# ---------------------------------------------------------------------------


def build_summary_row(data: dict) -> tuple:
    company_name = data.get("company_name", "Unknown")
    recommended = data.get("recommended") or []
    top = next((o for o in recommended if o.get("rank") == 1), recommended[0] if recommended else None)
    if not top:
        return company_name, "n/a", "not dispatched", 0, "n/a"

    executor_mode = top.get("executor_mode", "n/a")
    executor_result = top.get("executor_result")
    if not executor_result:
        return company_name, executor_mode, "not dispatched", 0, "n/a"

    status = executor_result.get("status", "unknown")
    n_deliverables = len(executor_result.get("deliverables") or [])
    activation = executor_result.get("estimated_activation_time") or "n/a"
    return company_name, executor_mode, status, n_deliverables, activation


def print_summary(rows: list) -> None:
    print(HEAVY)
    print("SUMMARY ACROSS ALL COMPANIES")
    print(HEAVY)
    print(f"{'Company':<20s} {'Mode':<15s} {'Status':<11s} {'Deliverables':<13s} {'Activation':<12s}")

    total_deliverables = 0
    full_builds_completed = 0
    market_connections_made = 0

    for company_name, mode, status, n_deliverables, activation in rows:
        print(f"{company_name:<20s} {mode:<15s} {status:<11s} {n_deliverables:<13d} {activation:<12s}")
        total_deliverables += n_deliverables
        if mode == "full_build" and status == "complete":
            full_builds_completed += 1
        if mode == "market_connect" and status == "connected":
            market_connections_made += 1

    print(HEAVY)
    print(f"Total executor deliverables produced: {total_deliverables}")
    print(f"Full builds completed: {full_builds_completed}")
    print(f"Market connections made: {market_connections_made}")
    print(HEAVY)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AIYYARY — Executor Deliverables Report (read-only, does not re-run any agent)"
    )
    parser.add_argument(
        "--company",
        help='Case-insensitive partial match on company name, e.g. --company "baukjen"',
    )
    parser.add_argument(
        "--all-ranks",
        action="store_true",
        help="Show deliverables for every ranked opportunity, not just rank 1 (default)",
    )
    parser.add_argument(
        "--show-roadmap",
        action="store_true",
        help="After each company's deliverables, print its risk register, phased roadmap, "
        "and headline ROI from the most recent agent3_*.json",
    )
    args = parser.parse_args()

    known_companies = discover_companies()
    companies = resolve_companies(args.company, known_companies)
    if args.company and not companies:
        print(f"[ERROR] No company name contains '{args.company}'.")
        if known_companies:
            print(f"Known companies: {', '.join(known_companies)}")
        else:
            print("No Agent 2 outputs found in outputs/ yet.")
        sys.exit(1)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(HEAVY)
    print("AIYYARY — EXECUTOR DELIVERABLES REPORT")
    print(f"Generated: {timestamp}")
    print(HEAVY)
    print()

    summary_rows = []
    for company_name in companies:
        path, data = find_latest_agent2_output(company_name)
        if data is None:
            print(f"[WARNING] No Agent 2 output found for {company_name} in outputs/. Skipping.")
            print()
            continue
        try:
            print_company(data, args.all_ranks)
        except Exception as exc:
            print(f"[ERROR] Failed to render {company_name}: {exc}")
            print()
        summary_rows.append(build_summary_row(data))

        if args.show_roadmap:
            _, agent3_data = find_latest_agent3_output(company_name)
            if agent3_data is None:
                print(f"[WARNING] No Agent 3 output found for {company_name} in outputs/. Skipping roadmap section.")
                print()
            else:
                try:
                    print_risk_and_roadmap(agent3_data)
                except Exception as exc:
                    print(f"[ERROR] Failed to render roadmap for {company_name}: {exc}")
                    print()

    if summary_rows:
        print_summary(summary_rows)
    else:
        print("No Agent 2 outputs were found for any company.")


if __name__ == "__main__":
    main()
