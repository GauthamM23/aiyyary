"""Live public-data enrichment for AIYYARY Agent 1.

Gathers publicly available intelligence about a fashion/apparel SME
(website scrape, Google Business Profile, Companies House, Facebook Ad
Library, Instagram) and maps every finding to a capability-pillar signal
(Stream B). Every external call is wrapped in try/except with an explicit
timeout so a failed source never prevents the enrichment run from
completing.
"""

import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode, urljoin

import requests
from bs4 import BeautifulSoup
from cachetools import TTLCache
from dotenv import load_dotenv

from agents.llm_client import LLMClient

load_dotenv()

PILLARS = [
    "data_foundations",
    "process_digitisation",
    "technology_infrastructure",
    "staff_digital_literacy",
    "governance_compliance",
    "ai_tool_experience",
]

TIMEOUT = int(os.getenv("ENRICHMENT_TIMEOUT_SECONDS", "30"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))
CACHE_DIR = os.getenv("CACHE_DIR", "./cache")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_enrichment_cache = TTLCache(maxsize=256, ttl=CACHE_TTL_SECONDS)


def _is_api_key_configured(value: str) -> bool:
    return bool(value) and not value.strip().lower().startswith("placeholder")


def _normalize_url(url: str) -> str:
    if not url:
        return ""
    url = url.strip().lower()
    url = re.sub(r"^https?://", "", url)
    url = re.sub(r"^www\.", "", url)
    return url.rstrip("/")


class EnrichmentAgent:
    """Runs all five public-data intelligence functions and synthesises Stream B."""

    def __init__(self):
        self.llm = LLMClient()

    # ------------------------------------------------------------------
    # Function 1: Website scrape
    # ------------------------------------------------------------------
    # Builds Stream B's largest single set of signals: platform, privacy
    # policy, cookie consent, analytics/personalisation/generative-AI
    # tooling, SSL, page speed. These signals feed two different places
    # downstream — the six-pillar Stream B scores in Agent 1, and the
    # enrichment_structured booleans (privacy_policy_found, platform_detected,
    # etc.) that Agent 3 later uses to deterministically set regulatory_risk,
    # vendor_risk, and hallucination_risk severity.
    def scrape_website(self, url: str) -> dict:
        result = {
            "url": url,
            "platform": {"name": "unknown", "confidence": "low", "evidence": ""},
            "privacy_policy": {
                "found": False,
                "url": None,
                "mentions_ai": False,
                "mentions_profiling": False,
            },
            "cookie_consent": {"found": False, "provider": None},
            "chat_widget": {"found": False, "provider": None},
            "analytics": {"tools_detected": [], "count": 0},
            "marketing_automation": {"tools_detected": [], "count": 0},
            "review_widgets": {"found": False, "providers": []},
            "structured_product_data": {"found": False},
            "ssl": {"enabled": False},
            "page_load_seconds": None,
            "raw_signals": [],
            "pillar_contributions": {},
            "errors": [],
        }

        contributions: dict = {}

        def add_signal(pillar: str, score: float, evidence: str, band: str = "", confidence: str = "medium") -> None:
            contributions.setdefault(pillar, []).append(
                {"score": score, "evidence": evidence, "band": band, "confidence": confidence}
            )
            result["raw_signals"].append(
                f"[{pillar}] {evidence} (score={score}, band={band}, confidence={confidence})"
            )

        if not url:
            result["errors"].append("No URL provided")
            result["pillar_contributions"] = contributions
            return result

        if not url.startswith("http"):
            url = "https://" + url
        result["url"] = url
        result["ssl"]["enabled"] = url.lower().startswith("https")

        html = ""
        headers_resp = {}
        try:
            start = time.time()
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
            elapsed = time.time() - start
            html = resp.text or ""
            headers_resp = resp.headers
            result["page_load_seconds"] = round(elapsed, 2)

            body_text_len = len(BeautifulSoup(html, "html.parser").get_text(strip=True)) if html else 0
            if body_text_len < 500:
                try:
                    rendered, render_elapsed = self._render_with_playwright(url)
                    if rendered:
                        html = rendered
                        result["page_load_seconds"] = round(render_elapsed, 2)
                except Exception as exc:
                    result["errors"].append(f"Playwright fallback failed: {exc}")
        except Exception as exc:
            result["errors"].append(f"requests.get failed: {exc}")
            try:
                rendered, render_elapsed = self._render_with_playwright(url)
                html = rendered or ""
                if rendered:
                    result["page_load_seconds"] = round(render_elapsed, 2)
            except Exception as exc2:
                result["errors"].append(f"Playwright fallback also failed: {exc2}")

        if not html:
            result["pillar_contributions"] = contributions
            return result

        # Everything below is pure string/regex analysis of already-fetched HTML.
        # Wrapped defensively so a single unexpected parsing issue never raises
        # out of scrape_website() — findings gathered so far are still returned.
        try:
            lower_html = html.lower()

            # --- E-commerce platform detection ---
            platform_checks = [
                ("Shopify", "cdn.shopify.com", (6, 8)),
                ("BigCommerce", "bigcommerce.com", (6, 8)),
                ("WooCommerce", "wp-content/plugins/woocommerce", (5, 7)),
                ("Wix", "static.wixstatic.com", (4, 6)),
                ("Squarespace", "squarespace.com", (4, 6)),
                ("Magento", "mage/", (3, 5)),
            ]
            detected = next(((n, m, b) for n, m, b in platform_checks if m in lower_html), None)
            if detected:
                name, marker, band = detected
                mid = round((band[0] + band[1]) / 2, 1)
                result["platform"] = {
                    "name": name,
                    "confidence": "high",
                    "evidence": f"Found '{marker}' in page source",
                }
                add_signal(
                    "technology_infrastructure",
                    mid,
                    f"E-commerce platform detected: {name}",
                    band=f"{band[0]}-{band[1]}",
                    confidence="high",
                )
            else:
                result["platform"] = {
                    "name": "custom_or_unknown",
                    "confidence": "low",
                    "evidence": "No known platform markers found in page source",
                }
                add_signal(
                    "technology_infrastructure",
                    4,
                    "No recognized e-commerce platform detected",
                    band="3-5",
                    confidence="low",
                )

            # --- Privacy policy ---
            privacy_found = False
            privacy_url_found = None
            privacy_text = ""
            for path in ["/privacy-policy", "/privacy", "/legal/privacy", "/pages/privacy-policy"]:
                try:
                    test_url = urljoin(url, path)
                    presp = requests.get(test_url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
                    if presp.status_code == 200:
                        privacy_found = True
                        privacy_url_found = test_url
                        privacy_text = (presp.text or "").lower()
                        break
                except Exception as exc:
                    result["errors"].append(f"privacy policy check failed for {path}: {exc}")

            mentions_ai = bool(re.search(r"\bai\b|artificial intelligence", privacy_text))
            mentions_profiling = "profiling" in privacy_text
            mentions_automated_decision = "automated decision" in privacy_text
            mentions_data_processing = "data processing" in privacy_text

            result["privacy_policy"] = {
                "found": privacy_found,
                "url": privacy_url_found,
                "mentions_ai": mentions_ai,
                "mentions_profiling": mentions_profiling,
            }
            if privacy_found:
                mention_count = sum(
                    [mentions_ai, mentions_profiling, mentions_automated_decision, mentions_data_processing]
                )
                add_signal(
                    "governance_compliance",
                    min(9, 5 + mention_count),
                    f"Privacy policy found at {privacy_url_found} ({mention_count} governance-relevant mentions)",
                    band="5-9",
                    confidence="high",
                )
            else:
                add_signal(
                    "governance_compliance",
                    2,
                    "No privacy policy found at common paths",
                    band="1-3",
                    confidence="low",
                )

            # --- Cookie consent ---
            for name, marker in [
                ("OneTrust", "onetrust"),
                ("Cookiebot", "cookiebot"),
                ("CookieYes", "cookieyes"),
                ("Generic cookie-consent", "cookie-consent"),
            ]:
                if marker in lower_html:
                    result["cookie_consent"] = {"found": True, "provider": name}
                    add_signal(
                        "governance_compliance",
                        7,
                        f"Cookie consent tool detected: {name}",
                        band="6-8",
                        confidence="high",
                    )
                    break

            # --- Chat widget ---
            # Chat widgets alone measure customer-service tooling, not AI adoption,
            # so they no longer contribute to ai_tool_experience by default. They
            # only count as an AI signal if the same <script> block that references
            # the widget also references "ai"/"gpt" — see generative AI detection
            # below.
            chat_widget_found = None
            for name, marker in [
                ("Tidio", "tidio"),
                ("Intercom", "intercom"),
                ("Zendesk", "zopim"),
                ("Zendesk", "zendesk"),
                ("LiveChat", "livechatinc"),
                ("Drift", "drift.com"),
                ("Crisp", "crisp.chat"),
            ]:
                if marker in lower_html:
                    result["chat_widget"] = {"found": True, "provider": name}
                    chat_widget_found = (name, marker)
                    break

            # --- Analytics and tracking ---
            # Generic analytics tools measure general digital tooling, not AI
            # adoption. They still feed data_foundations; ai_tool_experience is
            # only credited via the "basic analytics only" baseline further down.
            analytics_tools = []
            if "gtag(" in lower_html or re.search(r"g-[a-z0-9]{6,}", lower_html):
                analytics_tools.append("Google Analytics 4")
            if "fbq(" in lower_html or "connect.facebook.net" in lower_html:
                analytics_tools.append("Meta Pixel")
            if "hotjar" in lower_html:
                analytics_tools.append("Hotjar")
            if "clarity.ms" in lower_html:
                analytics_tools.append("Microsoft Clarity")
            result["analytics"] = {"tools_detected": analytics_tools, "count": len(analytics_tools)}
            if len(analytics_tools) >= 2:
                evidence = f"Multiple analytics tools detected: {', '.join(analytics_tools)}"
                add_signal("data_foundations", 7, evidence, band="6+", confidence="high")
            elif len(analytics_tools) == 1:
                evidence = f"Analytics tool detected: {analytics_tools[0]}"
                add_signal("data_foundations", 5, evidence, band="4-6", confidence="medium")

            # --- AI personalisation / recommendation engines ---
            personalisation_checks = [
                ("Nosto", ["nosto.com", "nosto"]),
                ("Dynamic Yield", ["dynamicyield.com", "dy-api"]),
                ("Rebuy", ["rebuyengine.com"]),
                ("Constructor.io", ["cnstrc.com"]),
                ("Klevu", ["klevu.com"]),
                ("Searchspring", ["searchspring.net"]),
                ("Algolia", ["algolia.net", "algolianet"]),
                ("LimeSpot", ["limespot.com"]),
                ("Visenze", ["visenze.com"]),
            ]
            personalisation_tools = [
                name for name, markers in personalisation_checks if any(m in lower_html for m in markers)
            ]
            has_personalisation_keyword = any(
                kw in lower_html for kw in ["recommendation-engine", "personalisation", "personalization"]
            )
            result["ai_personalisation_tools"] = {
                "tools_detected": personalisation_tools,
                "keyword_match": has_personalisation_keyword,
            }
            if personalisation_tools or has_personalisation_keyword:
                evidence = (
                    f"AI personalisation/recommendation tool(s) detected: {', '.join(personalisation_tools)}"
                    if personalisation_tools
                    else "Page source references a personalisation/recommendation engine"
                )
                add_signal("ai_tool_experience", 8, evidence, band="7-9", confidence="high")

            # --- Generative AI tool indicators ---
            genai_markers = [
                ("OpenAI", "openai.com"),
                ("Anthropic/Claude", "anthropic.com"),
                ("Cohere", "cohere.ai"),
            ]
            genai_detected = [name for name, marker in genai_markers if marker in lower_html]
            if genai_detected:
                add_signal(
                    "ai_tool_experience",
                    9,
                    f"Generative AI provider reference(s) detected: {', '.join(genai_detected)}",
                    band="8-9",
                    confidence="high",
                )
            # Chat widget + "ai"/"gpt" mentioned in the *same* script block. Uses a
            # word-boundary match (not a bare "ai" substring) because "ai" occurs
            # inside common words like "contain"/"domain"/"email" in almost every
            # script tag — a bare substring check would false-positive on nearly
            # every page.
            if chat_widget_found:
                widget_name, widget_marker = chat_widget_found
                for block in re.findall(r"<script[^>]*>(.*?)</script>", lower_html, re.DOTALL):
                    if widget_marker in block and re.search(r"\bai\b|gpt", block):
                        add_signal(
                            "ai_tool_experience",
                            9,
                            f"Chat widget ({widget_name}) script references AI/GPT",
                            band="8-9",
                            confidence="medium",
                        )
                        break

            # --- Review platform widgets ---
            review_providers = [
                name
                for name, marker in [
                    ("Judge.me", "judge.me"),
                    ("Yotpo", "yotpo"),
                    ("Trustpilot", "trustpilot"),
                    ("Google Reviews", "google reviews"),
                ]
                if marker in lower_html
            ]
            result["review_widgets"] = {"found": bool(review_providers), "providers": review_providers}
            if review_providers:
                add_signal(
                    "process_digitisation",
                    6,
                    f"Review widget(s) detected: {', '.join(review_providers)}",
                    band="5-7",
                    confidence="medium",
                )

            # --- Marketing automation ---
            # Automation tools, not AI tools — capped at the mid band (4-6), never
            # the 7-9 band reserved for personalisation/generative-AI signals.
            marketing_tools = [
                name
                for name, marker in [
                    ("Klaviyo", "klaviyo"),
                    ("Mailchimp", "mailchimp"),
                    ("Omnisend", "omnisend"),
                    ("Drip", "getdrip"),
                ]
                if marker in lower_html
            ]
            result["marketing_automation"] = {"tools_detected": marketing_tools, "count": len(marketing_tools)}
            if marketing_tools:
                marketing_score = 6 if review_providers else 5
                evidence = f"Marketing automation tool(s) detected: {', '.join(marketing_tools)}"
                if review_providers:
                    evidence += " (also has review platform tooling)"
                add_signal("ai_tool_experience", marketing_score, evidence, band="4-6", confidence="medium")

            # --- Baseline: basic analytics only, no AI-specific signal found ---
            if not contributions.get("ai_tool_experience") and analytics_tools:
                add_signal(
                    "ai_tool_experience",
                    3,
                    f"Only generic analytics tooling detected ({', '.join(analytics_tools)}); "
                    "no AI-specific tooling found",
                    band="2-3",
                    confidence="medium",
                )

            # --- Structured product data ---
            has_product_schema = bool(re.search(r'"@type"\s*:\s*"product"', lower_html))
            result["structured_product_data"] = {"found": has_product_schema}
            if has_product_schema:
                add_signal(
                    "data_foundations",
                    7,
                    "JSON-LD Product structured data found",
                    band="6-8",
                    confidence="high",
                )

            # --- SSL and security headers ---
            header_keys = {k.lower() for k in headers_resp.keys()} if headers_resp else set()
            security_headers_present = [
                h for h in ["x-frame-options", "content-security-policy"] if h in header_keys
            ]
            if result["ssl"]["enabled"]:
                score = 6 + (1 if security_headers_present else 0)
                add_signal(
                    "governance_compliance",
                    min(score, 9),
                    f"HTTPS enabled; security headers present: {security_headers_present or 'none'}",
                    band="6-9",
                    confidence="high",
                )
            else:
                add_signal(
                    "governance_compliance",
                    3,
                    "Site not served over HTTPS",
                    band="1-3",
                    confidence="high",
                )

            # --- Page load time ---
            if result["page_load_seconds"] is not None:
                if result["page_load_seconds"] < 3:
                    add_signal(
                        "technology_infrastructure",
                        8,
                        f"Fast page load: {result['page_load_seconds']}s",
                        band="7-9",
                        confidence="medium",
                    )
                elif result["page_load_seconds"] > 8:
                    add_signal(
                        "technology_infrastructure",
                        3,
                        f"Slow page load: {result['page_load_seconds']}s",
                        band="2-4",
                        confidence="medium",
                    )
        except Exception as exc:
            result["errors"].append(f"HTML analysis failed: {exc}")

        result["pillar_contributions"] = contributions
        return result

    def _render_with_playwright(self, url: str):
        from playwright.sync_api import sync_playwright

        start = time.time()
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                page = browser.new_page(user_agent=USER_AGENT)
                page.goto(url, timeout=TIMEOUT * 1000)
                content = page.content()
            finally:
                browser.close()
        return content, time.time() - start

    # ------------------------------------------------------------------
    # Function 2: Google Business Profile
    # ------------------------------------------------------------------
    # Review volume/recency is the main external proxy for how actively a
    # business manages its customer-facing digital presence — feeds the
    # process_digitisation pillar's Stream B score. Skips cleanly (returns
    # an "error" field, never raises) if GOOGLE_PLACES_API_KEY is unset.
    def get_google_profile(self, business_name: str, location: str, input_website: str = None) -> dict:
        result = {
            "total_reviews": None,
            "average_rating": None,
            "last_review_days_ago": None,
            "is_claimed": None,
            "has_complete_hours": None,
            "website_matches_input": None,
            "business_status": None,
            "pillar_contributions": {},
            "error": None,
        }

        api_key = os.getenv("GOOGLE_PLACES_API_KEY", "")
        if not _is_api_key_configured(api_key):
            result["error"] = "API key not configured"
            return result

        try:
            search_resp = requests.get(
                "https://maps.googleapis.com/maps/api/place/textsearch/json",
                params={"query": f"{business_name} {location}", "type": "clothing_store", "key": api_key},
                timeout=TIMEOUT,
            )
            candidates = (search_resp.json() or {}).get("results", [])
            if not candidates:
                result["error"] = "No matching Google Places result found"
                return result

            place_id = candidates[0].get("place_id")
            details_resp = requests.get(
                "https://maps.googleapis.com/maps/api/place/details/json",
                params={
                    "place_id": place_id,
                    "fields": "name,rating,user_ratings_total,reviews,opening_hours,website,formatted_address,business_status",
                    "key": api_key,
                },
                timeout=TIMEOUT,
            )
            details = (details_resp.json() or {}).get("result", {})

            total_reviews = details.get("user_ratings_total")
            average_rating = details.get("rating")
            reviews = details.get("reviews", []) or []
            last_review_days_ago = None
            if reviews:
                most_recent_ts = max((r.get("time", 0) for r in reviews), default=None)
                if most_recent_ts:
                    last_review_days_ago = (
                        datetime.now(timezone.utc) - datetime.fromtimestamp(most_recent_ts, tz=timezone.utc)
                    ).days

            opening_hours = details.get("opening_hours", {}) or {}
            has_complete_hours = bool(opening_hours.get("weekday_text"))
            website = details.get("website")
            is_claimed = bool(website) and has_complete_hours

            website_matches_input = None
            if input_website and website:
                website_matches_input = _normalize_url(website) == _normalize_url(input_website)

            result.update(
                {
                    "total_reviews": total_reviews,
                    "average_rating": average_rating,
                    "last_review_days_ago": last_review_days_ago,
                    "is_claimed": is_claimed,
                    "has_complete_hours": has_complete_hours,
                    "website_matches_input": website_matches_input,
                    "business_status": details.get("business_status"),
                }
            )

            contributions: dict = {}
            if total_reviews is not None:
                if total_reviews < 10:
                    score = 2
                elif total_reviews <= 50:
                    score = 5
                else:
                    score = 8
                contributions.setdefault("process_digitisation", []).append(
                    {"score": score, "evidence": f"{total_reviews} Google reviews"}
                )
            if last_review_days_ago is not None and last_review_days_ago > 180:
                contributions.setdefault("process_digitisation", []).append(
                    {"score": 3, "evidence": f"Most recent review is {last_review_days_ago} days old"}
                )
            result["pillar_contributions"] = contributions
        except Exception as exc:
            result["error"] = f"Google Places lookup failed: {exc}"

        return result

    # ------------------------------------------------------------------
    # Function 3: Companies House
    # ------------------------------------------------------------------
    # Pulls official UK filing history — this is the one enrichment source
    # with legal/regulatory weight rather than just marketing signal. Its
    # days_since_last_filing output is what later gives Agent 3's
    # regulatory_risk override real teeth: a business can self-report strong
    # governance in the owner interview, but a stale Companies House filing
    # record forces the risk severity up regardless.
    def check_companies_house(self, company_name: str) -> dict:
        result = {
            "company_status": None,
            "incorporation_date": None,
            "sic_codes": [],
            "days_since_last_filing": None,
            "filing_count_last_2_years": None,
            "has_accounts_filed": None,
            "pillar_contributions": {},
            "error": None,
        }

        api_key = os.getenv("COMPANIES_HOUSE_API_KEY", "")
        if not _is_api_key_configured(api_key):
            result["error"] = "API key not configured"
            return result

        def normalize(name: str) -> str:
            name = name.lower()
            for token in [" ltd", " limited", " & ", " and "]:
                name = name.replace(token, " ")
            return re.sub(r"[^a-z0-9]", "", name)

        try:
            search_resp = requests.get(
                "https://api.company-information.service.gov.uk/search/companies",
                params={"q": company_name, "items_per_page": 5},
                auth=(api_key, ""),
                timeout=TIMEOUT,
            )
            items = (search_resp.json() or {}).get("items", [])

            def is_active(item: dict) -> bool:
                return (item.get("company_status") or "").lower() == "active"

            # Prefer the top ACTIVE result whose name fuzzy-matches; fall back
            # progressively to a name match of any status, then the top active
            # result regardless of name, then simply the first result returned.
            target_norm = normalize(company_name)
            best_match = (
                next((i for i in items if normalize(i.get("title", "")) == target_norm and is_active(i)), None)
                or next((i for i in items if normalize(i.get("title", "")) == target_norm), None)
                or next((i for i in items if is_active(i)), None)
                or (items[0] if items else None)
            )
            if not best_match:
                result["error"] = "No matching company found"
                return result

            company_number = best_match.get("company_number")
            detail = requests.get(
                f"https://api.company-information.service.gov.uk/company/{company_number}",
                auth=(api_key, ""),
                timeout=TIMEOUT,
            ).json() or {}

            filings = (
                requests.get(
                    f"https://api.company-information.service.gov.uk/company/{company_number}/filing-history",
                    params={"items_per_page": 10},
                    auth=(api_key, ""),
                    timeout=TIMEOUT,
                ).json()
                or {}
            ).get("items", [])

            days_since_last_filing = None
            dates = [f.get("date") for f in filings if f.get("date")]
            if dates:
                filed_date = datetime.strptime(max(dates), "%Y-%m-%d").replace(tzinfo=timezone.utc)
                days_since_last_filing = (datetime.now(timezone.utc) - filed_date).days

            two_years_ago = datetime.now(timezone.utc) - timedelta(days=730)
            filing_count_last_2_years = sum(
                1
                for f in filings
                if f.get("date")
                and datetime.strptime(f["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc) >= two_years_ago
            )
            has_accounts_filed = any(f.get("category") == "accounts" for f in filings)

            result.update(
                {
                    "company_status": detail.get("company_status"),
                    "incorporation_date": detail.get("date_of_creation"),
                    "sic_codes": detail.get("sic_codes", []),
                    "days_since_last_filing": days_since_last_filing,
                    "filing_count_last_2_years": filing_count_last_2_years,
                    "has_accounts_filed": has_accounts_filed,
                }
            )

            contributions: dict = {}
            if days_since_last_filing is not None and days_since_last_filing > 365:
                contributions.setdefault("governance_compliance", []).append(
                    {"score": 3, "evidence": f"No filings in {days_since_last_filing} days"}
                )
            if filing_count_last_2_years == 0:
                contributions.setdefault("governance_compliance", []).append(
                    {"score": 2, "evidence": "No filings in the last 2 years"}
                )
            elif filing_count_last_2_years is not None and filing_count_last_2_years >= 2:
                contributions.setdefault("governance_compliance", []).append(
                    {"score": 6, "evidence": f"{filing_count_last_2_years} filings in the last 2 years"}
                )
            result["pillar_contributions"] = contributions
        except Exception as exc:
            result["error"] = f"Companies House lookup failed: {exc}"

        return result

    # ------------------------------------------------------------------
    # Function 4: Facebook Ad Library
    # ------------------------------------------------------------------
    # Detects whether the business runs paid social — a proxy for active
    # marketing investment that feeds both ai_tool_experience and
    # process_digitisation. JS-rendered page, so this is the second of two
    # places (after scrape_website) that falls back to a headless Playwright
    # render when the plain HTTP response comes back empty.
    def check_facebook_ad_library(self, business_name: str) -> dict:
        result = {"has_active_ads": None, "detection_method": "failed", "pillar_contributions": {}}
        params = {
            "active_status": "all",
            "ad_type": "all",
            "country": "GB",
            "q": business_name,
            "search_type": "keyword_unordered",
        }
        base_url = "https://www.facebook.com/ads/library/"

        html = ""
        detection_method = "failed"
        try:
            resp = requests.get(
                base_url,
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=15,
            )
            html = resp.text or ""
            detection_method = "html_parse"
        except Exception:
            html = ""

        # The Ad Library is JS-rendered — a plain request typically returns an
        # empty shell. Under 2000 chars means the real content never loaded, so
        # fall back to Playwright the same way scrape_website() does.
        if len(html) < 2000:
            try:
                from playwright.sync_api import sync_playwright

                full_url = f"{base_url}?{urlencode(params)}"
                with sync_playwright() as p:
                    browser = p.chromium.launch()
                    try:
                        page = browser.new_page(user_agent=USER_AGENT)
                        page.goto(full_url, timeout=TIMEOUT * 1000)
                        page.wait_for_load_state("networkidle", timeout=TIMEOUT * 1000)
                        html = page.content()
                    finally:
                        browser.close()
                detection_method = "playwright_render"
            except Exception:
                if not html:
                    result["has_active_ads"] = None
                    result["detection_method"] = "failed"
                    return result
                # requests returned some (sub-threshold) content and Playwright
                # errored — parse what we have rather than discarding it.

        result["detection_method"] = detection_method
        lower_html = html.lower()
        lower_business_name = business_name.lower().strip()

        has_json_markers = any(
            marker in lower_html for marker in ['"page_id"', '"ad_archive_id"', '"totalcount"', '"result_count"']
        )
        # Facebook's ad-library markup is obfuscated/hashed, so there's no stable
        # "ad card" selector to target. As a proxy: the business name appearing
        # more than once (beyond the single echo in the search box/header)
        # suggests it's repeated across rendered ad cards.
        business_name_in_card = bool(lower_business_name) and lower_html.count(lower_business_name) > 1

        has_ads = has_json_markers or business_name_in_card
        result["has_active_ads"] = has_ads
        if has_ads:
            result["pillar_contributions"] = {
                "ai_tool_experience": [{"score": 5, "evidence": "Active Facebook ads detected"}],
                "process_digitisation": [{"score": 5, "evidence": "Active Facebook ads detected"}],
            }
        return result

    # ------------------------------------------------------------------
    # Function 5: Instagram presence
    # ------------------------------------------------------------------
    # Optional (skipped cleanly if no handle is on the profile) — bio/post
    # activity feeds staff_digital_literacy and, if AI/e-commerce tooling is
    # mentioned in the bio, ai_tool_experience too.
    def check_instagram_presence(self, instagram_handle: str) -> dict:
        if not instagram_handle:
            return {"skipped": True, "reason": "No handle provided"}

        result = {
            "handle": instagram_handle,
            "is_public": None,
            "approx_post_count": None,
            "mentions_ai_or_ecommerce": False,
            "pillar_contributions": {},
            "error": None,
        }
        try:
            from playwright.sync_api import sync_playwright

            url = f"https://www.instagram.com/{instagram_handle}/"
            with sync_playwright() as p:
                browser = p.chromium.launch()
                try:
                    page = browser.new_page(user_agent=USER_AGENT)
                    page.goto(url, timeout=TIMEOUT * 1000)
                    title = page.title()
                    meta_desc = page.get_attribute('meta[name="description"]', "content") or ""
                finally:
                    browser.close()

            is_public = "page not found" not in title.lower() and "login" not in title.lower()
            result["is_public"] = is_public

            post_match = re.search(r"([\d,.]+[KkMm]?)\s+Posts", meta_desc)
            approx_post_count = post_match.group(1) if post_match else None
            result["approx_post_count"] = approx_post_count

            mentions = any(
                kw in meta_desc.lower()
                for kw in ["shopify", "ai ", " ai", "chatbot", "automation", "e-commerce", "ecommerce"]
            )
            result["mentions_ai_or_ecommerce"] = mentions

            contributions: dict = {}
            if is_public and approx_post_count:
                contributions.setdefault("staff_digital_literacy", []).append(
                    {"score": 5, "evidence": f"Active public Instagram presence (~{approx_post_count} posts)"}
                )
            if mentions:
                contributions.setdefault("ai_tool_experience", []).append(
                    {"score": 6, "evidence": "Instagram bio references AI tools or e-commerce platform"}
                )
            result["pillar_contributions"] = contributions
        except Exception as exc:
            result["error"] = f"Instagram check failed: {exc}"

        return result

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    # This is what agent1_capability.py calls to get Stream B. It runs all
    # five source functions above (each independently fault-tolerant — one
    # failing source never blocks the others), averages every pillar's
    # collected signal scores into stream_b_signals, and builds the
    # cross_validation_flags/stream_c_inferences that feed Agent 1's
    # divergence check. The report this returns, together with the owner
    # interview (Stream A), is the entire evidence base Agent 1 scores from.
    def run(self, profile: dict) -> dict:
        start_time = time.time()
        business_name = profile.get("business_name") or profile.get("company_name") or "unknown_business"
        url = profile.get("website") or profile.get("url")
        location = profile.get("location", "UK")
        instagram_handle = profile.get("instagram_handle")

        cache_key = url or business_name
        if cache_key in _enrichment_cache:
            return _enrichment_cache[cache_key]

        raw_findings = {
            "website": {},
            "google_profile": {},
            "companies_house": {},
            "facebook_ads": {},
            "instagram": {},
        }
        sources_available = []
        sources_failed = []

        try:
            raw_findings["website"] = self.scrape_website(url) if url else {"errors": ["No URL provided"]}
            if not raw_findings["website"].get("errors"):
                sources_available.append("website_scrape")
            else:
                sources_failed.append("website_scrape")
        except Exception as exc:
            raw_findings["website"] = {"errors": [str(exc)]}
            sources_failed.append("website_scrape")

        try:
            raw_findings["google_profile"] = self.get_google_profile(business_name, location, input_website=url)
            if not raw_findings["google_profile"].get("error"):
                sources_available.append("google_profile")
            else:
                sources_failed.append("google_profile")
        except Exception as exc:
            raw_findings["google_profile"] = {"error": str(exc)}
            sources_failed.append("google_profile")

        try:
            raw_findings["companies_house"] = self.check_companies_house(business_name)
            if not raw_findings["companies_house"].get("error"):
                sources_available.append("companies_house")
            else:
                sources_failed.append("companies_house")
        except Exception as exc:
            raw_findings["companies_house"] = {"error": str(exc)}
            sources_failed.append("companies_house")

        try:
            raw_findings["facebook_ads"] = self.check_facebook_ad_library(business_name)
            if raw_findings["facebook_ads"].get("has_active_ads") is not None:
                sources_available.append("facebook_ads")
            else:
                sources_failed.append("facebook_ads")
        except Exception:
            raw_findings["facebook_ads"] = {"has_active_ads": None, "detection_method": "failed"}
            sources_failed.append("facebook_ads")

        try:
            raw_findings["instagram"] = self.check_instagram_presence(instagram_handle)
            if raw_findings["instagram"].get("skipped"):
                pass
            elif not raw_findings["instagram"].get("error"):
                sources_available.append("instagram")
            else:
                sources_failed.append("instagram")
        except Exception as exc:
            raw_findings["instagram"] = {"error": str(exc)}
            sources_failed.append("instagram")

        stream_b_signals = {p: {"signals": [], "stream_b_score": None} for p in PILLARS}
        numeric_scores = {p: [] for p in PILLARS}
        for source_name, finding in raw_findings.items():
            contributions = finding.get("pillar_contributions", {}) if isinstance(finding, dict) else {}
            for pillar, entries in contributions.items():
                if pillar not in stream_b_signals:
                    continue
                for entry in entries:
                    stream_b_signals[pillar]["signals"].append(
                        f"[{source_name}] {entry['evidence']} (score={entry['score']})"
                    )
                    numeric_scores[pillar].append(entry["score"])

        for pillar in PILLARS:
            scores = numeric_scores[pillar]
            stream_b_signals[pillar]["stream_b_score"] = round(sum(scores) / len(scores), 2) if scores else None

        source_label_map = {
            "website": "website_scrape",
            "google_profile": "google_profile",
            "companies_house": "companies_house",
            "facebook_ads": "facebook_ads",
            "instagram": "instagram",
        }

        pre_population_suggestions = {}
        for pillar in PILLARS:
            score = stream_b_signals[pillar]["stream_b_score"]
            if score is None:
                continue
            if score <= 4:
                band, rng = "low", "1-4"
            elif score <= 7:
                band, rng = "mid", "5-7"
            else:
                band, rng = "high", "8-10"

            n_signals = len(stream_b_signals[pillar]["signals"])
            confidence = "high" if n_signals >= 3 else ("medium" if n_signals >= 1 else "low")

            source_counts = {}
            for source_name, finding in raw_findings.items():
                contributions = finding.get("pillar_contributions", {}) if isinstance(finding, dict) else {}
                if pillar in contributions:
                    source_counts[source_name] = len(contributions[pillar])
            dominant_source = max(source_counts, key=source_counts.get) if source_counts else "website"

            pre_population_suggestions[pillar] = {
                "suggested_band": band,
                "suggested_score_range": rng,
                "confidence": confidence,
                "reason": f"Derived from {n_signals} public data signal(s), mean score {score}",
                "source": source_label_map.get(dominant_source, dominant_source),
            }

        cross_validation_flags = []
        website_finding = raw_findings.get("website", {}) or {}
        platform_name = website_finding.get("platform", {}).get("name")
        gh = raw_findings.get("companies_house", {}) or {}
        google_profile = raw_findings.get("google_profile", {}) or {}

        if gh.get("filing_count_last_2_years") == 0:
            cross_validation_flags.append(
                {
                    "flag": "No Companies House filings in the last 24 months",
                    "pillar": "governance_compliance",
                    "severity": "medium",
                    "probe_question": (
                        "Companies House shows no recent filings — can you confirm your accounts "
                        "and confirmation statement are up to date?"
                    ),
                }
            )
        if website_finding and not website_finding.get("privacy_policy", {}).get("found", True):
            cross_validation_flags.append(
                {
                    "flag": "No privacy policy found on website",
                    "pillar": "governance_compliance",
                    "severity": "high",
                    "probe_question": (
                        "We couldn't find a privacy policy on your website — how do you currently "
                        "handle data protection and customer data governance?"
                    ),
                }
            )
        if google_profile.get("total_reviews") is not None and google_profile["total_reviews"] < 10:
            cross_validation_flags.append(
                {
                    "flag": "Very low Google review count",
                    "pillar": "process_digitisation",
                    "severity": "low",
                    "probe_question": (
                        "Your Google Business Profile has very few reviews — how do you currently "
                        "collect customer feedback?"
                    ),
                }
            )

        stream_c_inferences = []

        if platform_name == "Shopify" and profile.get("has_ecommerce") is False:
            question = (
                "Given that Shopify assets were detected on the business website but the owner states "
                "they have no online sales channel, what is the most likely explanation and which "
                "answer should carry more weight for assessing technology infrastructure maturity?"
            )
            context = json.dumps({"platform": platform_name, "owner_stated_ecommerce": False})
            try:
                answer = self.llm.inference_call(context=context, question=question)
            except Exception as exc:
                answer = f"[LLM inference call failed: {exc}]"
            stream_c_inferences.append({"trigger": "shopify_vs_no_ecommerce", "question": question, "answer": answer})

        if (
            website_finding.get("analytics", {}).get("count", 0) == 0
            and profile.get("owner_self_rated_data_foundations", 0) is not None
            and profile.get("owner_self_rated_data_foundations", 0) >= 7
        ):
            question = (
                "No web analytics tools were detected on this business's website. The owner has "
                "indicated high data maturity. What specific probing questions should the researcher "
                "ask to verify whether internal data collection exists independently of web analytics?"
            )
            context = json.dumps(
                {
                    "analytics_detected": 0,
                    "owner_self_rated_data_foundations": profile.get("owner_self_rated_data_foundations"),
                }
            )
            try:
                answer = self.llm.inference_call(context=context, question=question)
            except Exception as exc:
                answer = f"[LLM inference call failed: {exc}]"
            stream_c_inferences.append(
                {"trigger": "no_analytics_vs_high_data_maturity", "question": question, "answer": answer}
            )

        owner_self_rated_governance = profile.get("owner_self_rated_governance", 0)
        if (
            gh.get("filing_count_last_2_years") == 0
            and owner_self_rated_governance is not None
            and owner_self_rated_governance >= 7
        ):
            question = (
                "Companies House records show no filings in the last 24 months for this business. "
                "The owner has indicated strong governance and compliance practices. What is the "
                "most plausible interpretation and what should the researcher verify during the session?"
            )
            context = json.dumps(
                {
                    "filing_count_last_2_years": 0,
                    "owner_self_rated_governance": profile.get("owner_self_rated_governance"),
                }
            )
            try:
                answer = self.llm.inference_call(context=context, question=question)
            except Exception as exc:
                answer = f"[LLM inference call failed: {exc}]"
            stream_c_inferences.append(
                {"trigger": "no_filings_vs_high_governance", "question": question, "answer": answer}
            )

        report = {
            "business_name": business_name,
            "assessment_timestamp": datetime.now(timezone.utc).isoformat(),
            "stream_b_signals": stream_b_signals,
            "pre_population_suggestions": pre_population_suggestions,
            "cross_validation_flags": cross_validation_flags,
            "stream_c_inferences": stream_c_inferences,
            "raw_findings": raw_findings,
            "sources_available": sources_available,
            "sources_failed": sources_failed,
            "enrichment_duration_seconds": round(time.time() - start_time, 2),
        }

        self._save_report(business_name, report)
        _enrichment_cache[cache_key] = report
        return report

    def _save_report(self, business_name: str, report: dict) -> None:
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", business_name)
            date_str = datetime.now().strftime("%Y%m%d")
            path = os.path.join(CACHE_DIR, f"{safe_name}_{date_str}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
        except Exception:
            pass
