import os
import re

RAW_DIR = "data/raw_pages"
CLEAN_DIR = "data/cleaned_pages"
os.makedirs(CLEAN_DIR, exist_ok=True)

def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    return text.strip()

def main():
    for filename in os.listdir(RAW_DIR):
        if not filename.endswith(".txt"):
            continue

        in_path = os.path.join(RAW_DIR, filename)
        out_path = os.path.join(CLEAN_DIR, filename)

        with open(in_path, "r", encoding="utf-8") as f:
            text = f.read()

        cleaned = clean_text(text)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        print(f"Cleaned: {filename}")

    print("Done: cleaned pages saved.")

if __name__ == "__main__":
    main()
