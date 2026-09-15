"""Verifies GOOGLE_PLACES_API_KEY and COMPANIES_HOUSE_API_KEY are configured
and returning real data, independent of the full AIYYARY enrichment pipeline.

Usage:
    python verify_keys.py
"""

import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PLACEHOLDER = "placeholder_replace_with_real_key"
TIMEOUT = 10


def _print_key_prefix(name: str, value: str) -> None:
    if value:
        print(f"{name}: {value[:8]}...")
    else:
        print(f"{name}: (not set)")


def _key_is_usable(name: str, value: str) -> bool:
    if not value:
        print(f"[ERROR] {name} is missing from .env — skipping this test.")
        return False
    if value.strip() == PLACEHOLDER:
        print(f"[ERROR] {name} is still the placeholder value — skipping this test.")
        return False
    return True


def test_google_places(api_key: str) -> bool:
    print("\n" + "-" * 60)
    print("GOOGLE PLACES API TEST")
    print("-" * 60)

    if not _key_is_usable("GOOGLE_PLACES_API_KEY", api_key):
        return False

    try:
        response = requests.get(
            "https://maps.googleapis.com/maps/api/place/textsearch/json",
            params={"query": "Lucy and Yak Brighton UK", "type": "clothing_store", "key": api_key},
            timeout=TIMEOUT,
        )
        data = response.json() or {}
        status = data.get("status")
        print(f"status: {status}")

        if status == "OK":
            results = data.get("results", [])
            if results:
                first = results[0]
                print(f"name: {first.get('name')}")
                print(f"formatted_address: {first.get('formatted_address')}")
            return True

        error_message = data.get("error_message")
        if error_message:
            print(f"error_message: {error_message}")

        explanations = {
            "REQUEST_DENIED": "The API key is invalid, restricted, or the Places API is not enabled for this key.",
            "INVALID_REQUEST": "The request is missing a required parameter or is malformed.",
            "ZERO_RESULTS": "The key works, but no matching place was found for this query.",
            "OVER_QUERY_LIMIT": "The key's quota or billing limit has been exceeded.",
        }
        print(explanations.get(status, f"Unrecognised status '{status}' — check the Google Places API response."))
        return False

    except requests.exceptions.RequestException as exc:
        print(f"[EXCEPTION] {exc}")
        return False


def test_companies_house(api_key: str) -> bool:
    print("\n" + "-" * 60)
    print("COMPANIES HOUSE API TEST")
    print("-" * 60)

    if not _key_is_usable("COMPANIES_HOUSE_API_KEY", api_key):
        return False

    try:
        response = requests.get(
            "https://api.company-information.service.gov.uk/search/companies",
            params={"q": "Lucy and Yak", "items_per_page": 3},
            auth=(api_key, ""),
            timeout=TIMEOUT,
        )
        print(f"HTTP status code: {response.status_code}")

        if response.status_code == 200:
            items = (response.json() or {}).get("items", [])
            if items:
                first = items[0]
                print(f"title: {first.get('title')}")
                print(f"company_number: {first.get('company_number')}")
            return True

        if response.status_code == 401:
            print("API key rejected — check the key is correct and the account is verified.")
            return False

        print(f"status_code: {response.status_code}")
        print(f"response text: {response.text}")
        return False

    except requests.exceptions.RequestException as exc:
        print(f"[EXCEPTION] {exc}")
        return False


def main() -> None:
    google_key = os.getenv("GOOGLE_PLACES_API_KEY", "")
    companies_house_key = os.getenv("COMPANIES_HOUSE_API_KEY", "")

    print("=" * 60)
    print("LOADED KEYS")
    print("=" * 60)
    _print_key_prefix("GOOGLE_PLACES_API_KEY", google_key)
    _print_key_prefix("COMPANIES_HOUSE_API_KEY", companies_house_key)

    google_passed = test_google_places(google_key)
    companies_house_passed = test_companies_house(companies_house_key)

    ready = google_passed and companies_house_passed

    print("\n" + "═" * 36)
    print("KEY VERIFICATION SUMMARY")
    print("═" * 36)
    print(f"Google Places API    : {'PASS' if google_passed else 'FAIL'}")
    print(f"Companies House API  : {'PASS' if companies_house_passed else 'FAIL'}")
    print()
    print(f"Ready for enrichment : {'YES' if ready else 'NO'}")
    print("═" * 36)

    if not ready:
        if not google_passed and not companies_house_passed:
            print("Fix both GOOGLE_PLACES_API_KEY and COMPANIES_HOUSE_API_KEY in .env before running live enrichment.")
        elif not google_passed:
            print("Fix GOOGLE_PLACES_API_KEY in .env — see the Google Places test output above for the reason.")
        else:
            print("Fix COMPANIES_HOUSE_API_KEY in .env — see the Companies House test output above for the reason.")


if __name__ == "__main__":
    main()
