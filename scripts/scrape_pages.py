import os
import time
import requests
from slugify import slugify
from sources import SOURCES

RAW_DIR = "data/raw_pages"
os.makedirs(RAW_DIR, exist_ok=True)

API_URL = "https://en.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": "RAG-Task2/1.0 (educational project)"
}

def fetch_page_extract(title: str) -> str | None:
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "titles": title,
        "redirects": 1
    }

    response = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
    response.raise_for_status()

    data = response.json()
    pages = data.get("query", {}).get("pages", {})

    for _, page_data in pages.items():
        extract = page_data.get("extract", "").strip()
        if extract:
            return extract

    return None

def main():
    success = 0
    failed = 0

    for title in SOURCES:
        try:
            print(f"Downloading via API: {title}")
            text = fetch_page_extract(title)

            if not text:
                print(f"[ERROR] No text returned for: {title}")
                failed += 1
                continue

            filename = slugify(title) + ".txt"
            path = os.path.join(RAW_DIR, filename)

            with open(path, "w", encoding="utf-8") as f:
                f.write(f"Title: {title}\n\n{text}")

            success += 1
            time.sleep(0.5)

        except Exception as e:
            print(f"[ERROR] {title}: {e}")
            failed += 1

    print(f"\nDone. Success: {success}, Failed: {failed}")

if __name__ == "__main__":
    main()
