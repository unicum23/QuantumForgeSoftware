import os
import json
import re

CLEAN_DIR = "data/cleaned_pages"
TRANSFORMED_DIR = "data/transformed_pages"
KNOWLEDGE_BASE_DIR = "knowledge_base"

os.makedirs(TRANSFORMED_DIR, exist_ok=True)
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)

with open("terms_map.json", "r", encoding="utf-8") as f:
    TERMS_MAP = json.load(f)

def replace_terms(text: str, mapping: dict) -> str:
    sorted_items = sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True)

    for source, target in sorted_items:
        pattern = re.compile(rf"(?<!\w){re.escape(source)}(?!\w)", flags=re.IGNORECASE)
        text = pattern.sub(target, text)

    return text

def sanitize_filename(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9а-яё]+", "_", text, flags=re.IGNORECASE)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")

def extract_title(text: str) -> str:
    first_line = text.splitlines()[0].strip()
    if first_line.lower().startswith("title:"):
        return first_line.split(":", 1)[1].strip()
    return "document"

def clear_folder(folder: str) -> None:
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if os.path.isfile(path):
            os.remove(path)

def main():
    clear_folder(TRANSFORMED_DIR)
    clear_folder(KNOWLEDGE_BASE_DIR)

    for filename in os.listdir(CLEAN_DIR):
        if not filename.endswith(".txt"):
            continue

        in_path = os.path.join(CLEAN_DIR, filename)
        with open(in_path, "r", encoding="utf-8") as f:
            text = f.read()

        transformed = replace_terms(text, TERMS_MAP)
        title = extract_title(transformed)
        safe_name = sanitize_filename(title) + ".txt"

        out_path_1 = os.path.join(TRANSFORMED_DIR, safe_name)
        out_path_2 = os.path.join(KNOWLEDGE_BASE_DIR, safe_name)

        with open(out_path_1, "w", encoding="utf-8") as f:
            f.write(transformed)

        with open(out_path_2, "w", encoding="utf-8") as f:
            f.write(transformed)

        print(f"Saved: {safe_name}")

    print("Done: transformed knowledge base created.")

if __name__ == "__main__":
    main()
