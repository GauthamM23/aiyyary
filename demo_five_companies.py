"""Demo runner: the full Agent 1 pipeline against five real UK fashion SMEs.

For each company this script:
  1. Runs EnrichmentAgent.run() against the company's real website/social presence
     to gather live public-data intelligence (Stream B).
  2. Uses a hand-constructed 24-answer owner-interview instrument (Stream A) that
     represents a plausible researcher assessment for a business at that company's
     publicly observable digital maturity level. Stream A is deliberately an
     independent judgement call, not a mechanical restatement of Stream B — that
     independence is what lets the pipeline's divergence check (Stream C) do
     anything meaningful.
  3. Runs CapabilityAssessmentAgent.run() to synthesise both streams into the
     final Agent 1 output and save it to outputs/.
  4. Prints a per-company summary, then a comparison table across all five.
"""

import argparse
import time

from agents.agent1_capability import PILLARS, CapabilityAssessmentAgent
from agents.enrichment import EnrichmentAgent

DEMO_COMPANIES = [
    {
        "company_name": "Lucy & Yak",
        "website": "https://lucyandyak.com",
        "location": "Brighton, UK",
        "instagram_handle": "lucyandyak",
        "primary_challenge": "Managing rapid growth in online sales while maintaining brand authenticity and sustainable supply chain transparency",
    },
    {
        "company_name": "Nobody's Child",
        "website": "https://nobodyschild.com",
        "location": "London, UK",
        "instagram_handle": "nobodyschild",
        "primary_challenge": "Increasing repeat purchase rate and reducing returns through better size guidance online",
    },
    {
        "company_name": "Birdsong London",
        "website": "https://birdsong.london",
        "location": "London, UK",
        "instagram_handle": "birdsongldn",
        "primary_challenge": "Growing online sales beyond the existing ethical fashion community without losing brand values",
    },
    {
        "company_name": "Gudrun Sjödén UK",
        "website": "https://www.gudrunsjoden.com/en_gb/",
        "location": "London, UK",
        "instagram_handle": "gudrunsjoden",
        "primary_challenge": "Modernising digital customer experience while maintaining loyalty of an older, less digitally native customer base",
    },
    {
        "company_name": "Baukjen",
        "website": "https://www.baukjen.com",
        "location": "London, UK",
        "instagram_handle": "baukjen",
        "primary_challenge": "Leveraging sustainability credentials in AI-assisted product discovery and personalisation",
    },
]


def _instrument_from_scores(raw_scores: dict) -> dict:
    """Convert {pillar: [4 raw 1-9 scores]} into the full instrument answer format."""
    instrument = {}
    for pillar, scores in raw_scores.items():
        instrument[pillar] = [
            {
                "question": f"{pillar} question {i + 1}",
                "answer": f"researcher-assessed score {s}",
                "score": s,
            }
            for i, s in enumerate(scores)
        ]
    return instrument


def build_instrument(company_name: str) -> dict:
    """Hand-constructed Stream A instrument (24 answers) per company.

    Each score reflects the researcher's plausible live-interview judgement for
    a business at this company's publicly observable digital maturity level.
    Reasoning per pillar is commented inline, grounded in what a researcher
    would realistically expect to find during enrichment for a business of
    this type and scale.
    """

    if company_name == "Lucy & Yak":
        # DTC sustainable fashion brand: very strong, highly engaged social
        # media community, Shopify storefront, active Klaviyo email marketing,
        # strong digital team, owner visibly engaged with digital tools.
        # Expected: mid-high to high, strongest on staff digital literacy and
        # ai_tool_experience.
        raw = {
            "data_foundations": [6, 6, 7, 6],
            "process_digitisation": [6, 7, 6, 6],
            "technology_infrastructure": [6, 7, 6, 6],
            "staff_digital_literacy": [8, 7, 8, 7],  # owner + team highly engaged with digital tooling
            "governance_compliance": [6, 6, 5, 6],  # good governance for a brand of this size, not enterprise-grade
            "ai_tool_experience": [6, 7, 6, 7],  # active Klaviyo use plus owner's general digital-tool engagement
        }
    elif company_name == "Nobody's Child":
        # Fast-growing, digital-first UK womenswear brand: strong social
        # media, active paid advertising, Shopify-based, clear marketing
        # automation, younger digitally native team, sustainability focus.
        # Expected: mid-high on most pillars, strongest on technology
        # infrastructure and staff digital literacy.
        raw = {
            "data_foundations": [6, 7, 6, 6],
            "process_digitisation": [6, 7, 6, 6],
            "technology_infrastructure": [7, 8, 7, 7],  # strongest pillar: scaled Shopify setup supporting rapid growth
            "staff_digital_literacy": [7, 7, 6, 7],  # young, digitally native team
            "governance_compliance": [5, 6, 5, 5],  # growing brand, reasonable but not most mature
            "ai_tool_experience": [5, 6, 5, 5],  # clear marketing automation, not necessarily deeper AI adoption
        }
    elif company_name == "Birdsong London":
        # Small ethical fashion brand: garments hand-made with marginalised
        # women, very small team, basic Shopify store, limited social media,
        # no obvious marketing automation, founder-led with strong values but
        # limited digital resource. Expected: low-mid on most pillars,
        # slightly higher governance due to ethical brand values.
        raw = {
            "data_foundations": [3, 3, 4, 3],
            "process_digitisation": [3, 2, 3, 3],  # hand-made, low-volume production implies manual processes
            "technology_infrastructure": [4, 3, 4, 4],  # basic Shopify store
            "staff_digital_literacy": [3, 4, 3, 4],  # very small team, limited digital resource
            "governance_compliance": [5, 4, 5, 4],  # slightly higher: ethical/values-led transparency, still informal
            "ai_tool_experience": [2, 2, 3, 2],  # no obvious marketing automation or AI tooling
        }
    elif company_name == "Gudrun Sjödén UK":
        # Scandinavian heritage brand: established physical presence, older
        # customer base, traditional catalogue-style merchandising, website
        # functional but not sophisticated, limited AI/automation evidence,
        # conservative digital approach. Expected: low-mid overall, slightly
        # higher on data foundations due to established catalogue history.
        raw = {
            "data_foundations": [5, 6, 5, 5],  # decades of structured catalogue data; relative high point
            "process_digitisation": [4, 4, 5, 4],  # traditional order/fulfilment processes
            "technology_infrastructure": [4, 4, 3, 4],  # functional but dated e-commerce experience
            "staff_digital_literacy": [3, 4, 3, 4],  # older customer/staff base, less digitally native
            "governance_compliance": [5, 5, 6, 5],  # established European retailer, reasonable GDPR/compliance habits
            "ai_tool_experience": [2, 2, 3, 2],  # minimal AI tooling; modernisation is an explicit stated gap
        }
    elif company_name == "Baukjen":
        # Mid-size sustainable UK womenswear brand: strong e-commerce focus,
        # likely Shopify Plus, active email marketing, sustainability
        # messaging prominent, growing digital team, some personalisation
        # evidence. Expected: mid to mid-high, strongest on technology
        # infrastructure and process digitisation.
        raw = {
            "data_foundations": [6, 6, 7, 6],
            "process_digitisation": [7, 6, 7, 6],
            "technology_infrastructure": [7, 7, 6, 7],  # strongest pillar: Shopify Plus-class e-commerce focus
            "staff_digital_literacy": [5, 6, 5, 5],  # growing digital team, moderate depth
            "governance_compliance": [6, 5, 6, 5],  # sustainable brand, reasonable compliance posture
            "ai_tool_experience": [5, 5, 6, 5],  # some personalisation evidence, not deep AI adoption
        }
    else:
        raise ValueError(f"No instrument defined for company: {company_name}")

    return _instrument_from_scores(raw)


def run_company(company: dict) -> dict:
    profile = {
        "business_name": company["company_name"],
        "company_name": company["company_name"],
        "website": company["website"],
        "location": company["location"],
        "instagram_handle": company["instagram_handle"],
        "primary_challenge": company["primary_challenge"],
        "industry": "Fashion and Apparel",
    }

    enrichment_report = EnrichmentAgent().run(profile)
    instrument = build_instrument(company["company_name"])
    result = CapabilityAssessmentAgent().run(profile, instrument, enrichment_report=enrichment_report)
    return result


def print_company_summary(result: dict) -> None:
    print("-" * 78)
    print(result["company_name"])
    print("-" * 78)
    for pillar in PILLARS:
        p = result["pillar_scores"][pillar]
        b_score = p["stream_b_score"]
        b_str = "N/A " if b_score is None else f"{b_score:.2f}"
        c_str = "  " if p["stream_c_reconciled"] is None else f"C={p['stream_c_reconciled']:.2f}"
        print(
            f"  {pillar:28s} A={p['stream_a_score']:.2f}  B={b_str}  {c_str}  "
            f"Final={p['final_score']:.2f}  (dominant: {p['dominant_stream']})"
        )
    print(f"\n  Overall: {result['overall_score']:.2f}   Tier: {result['readiness_tier']}")
    print(f"  Verdict: {result['verdict']}")
    print(f"  Data confidence: {result['data_confidence_note']}")
    print(f"  Enrichment sources available: {result['enrichment_summary']['sources_available']}")
    print(f"  Enrichment sources failed:    {result['enrichment_summary']['sources_failed']}")
    print(f"  Stream C reconciliations:     {result['enrichment_summary']['stream_c_reconciliations']}")
    print(f"  Output saved to: {result['traceability']['output_file']}")
    print()


def print_comparison_table(results: list) -> None:
    print("=" * 118)
    print("COMPARISON ACROSS ALL FIVE COMPANIES")
    print("=" * 118)
    header = (
        f"{'Company':<22s} {'Overall':>8s} {'Tier':>15s} "
        f"{'Lowest Pillar':>32s} {'Highest Pillar':>32s} {'Sources':>8s} {'StreamC':>8s}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        pillar_scores = {p: r["pillar_scores"][p]["final_score"] for p in PILLARS}
        lowest_pillar = min(pillar_scores, key=pillar_scores.get)
        highest_pillar = max(pillar_scores, key=pillar_scores.get)
        n_sources = len(r["enrichment_summary"]["sources_available"])
        n_stream_c = r["enrichment_summary"]["stream_c_reconciliations"]
        lowest_str = f"{lowest_pillar} ({pillar_scores[lowest_pillar]:.2f})"
        highest_str = f"{highest_pillar} ({pillar_scores[highest_pillar]:.2f})"
        row = (
            f"{r['company_name']:<22s} {r['overall_score']:>8.2f} {r['readiness_tier']:>15s} "
            f"{lowest_str:>32s} {highest_str:>32s} {n_sources:>8d} {n_stream_c:>8d}"
        )
        print(row)
    print("=" * 118)


def main() -> None:
    parser = argparse.ArgumentParser(description="AIYYARY Agent 1 — Live Demo: Five Real UK Fashion SMEs")
    parser.add_argument(
        "--company",
        help="Run only the named company (case-insensitive, exact match), e.g. --company \"Lucy & Yak\"",
    )
    args = parser.parse_args()

    companies = DEMO_COMPANIES
    if args.company:
        companies = [c for c in DEMO_COMPANIES if c["company_name"].lower() == args.company.lower()]
        if not companies:
            known = ", ".join(c["company_name"] for c in DEMO_COMPANIES)
            raise SystemExit(f"No demo company named '{args.company}'. Known companies: {known}")

    results = []
    print("=" * 78)
    print("AIYYARY Agent 1 — Live Demo: Five Real UK Fashion SMEs")
    print("=" * 78)

    for company in companies:
        print(f"\nRunning full pipeline for: {company['company_name']} ({company['website']}) ...")
        start = time.time()
        try:
            result = run_company(company)
        except Exception as exc:
            print(f"[ERROR] {company['company_name']} failed unexpectedly: {exc}")
            continue
        elapsed = time.time() - start
        print(f"Completed {company['company_name']} in {elapsed:.1f}s\n")
        print_company_summary(result)
        results.append(result)

    if results:
        print_comparison_table(results)
    else:
        print("No companies completed successfully.")


if __name__ == "__main__":
    main()
