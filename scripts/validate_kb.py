import os
import json
import re

KB_DIR = "knowledge_base"

with open("terms_map.json", "r", encoding="utf-8") as f:
    terms_map = json.load(f)

original_terms = list(terms_map.keys())

def term_found(text: str, term: str) -> bool:
    pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", flags=re.IGNORECASE)
    return bool(pattern.search(text))

def main():
    problems = []

    files = [f for f in os.listdir(KB_DIR) if f.endswith(".txt")]
    print(f"Documents found: {len(files)}")

    for filename in files:
        path = os.path.join(KB_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        for term in original_terms:
            if term_found(text, term):
                problems.append((filename, term))

    if len(files) < 30:
        print("ERROR: Less than 30 documents in knowledge_base/")
    else:
        print("OK: 30+ documents present.")

    if problems:
        print("\nFound original terms still present:")
        for filename, term in problems[:100]:
            print(f"- {filename}: {term}")
    else:
        print("OK: No original terms found.")

if __name__ == "__main__":
    main()
