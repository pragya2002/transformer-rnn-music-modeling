from pathlib import Path
from tqdm import tqdm

ABC_DIR = Path("data/abc")
CORPUS_PATH = Path("data/corpus/abc_corpus.txt")

CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)

# Special separator between tunes so the model can learn boundaries
SEPARATOR = "\n\nXXX_NEW_TUNE\n\n"

# Cleaning thresholds (you can mention these in your report)
MIN_CHARS = 400   # minimum characters in an ABC file to keep
MAX_CHARS = 200000  # safety cap, extremely long corrupted files will be dropped


def is_valid_abc(text: str) -> bool:
    """
    Basic ABC validation / cleaning rules:
    - Non-empty
    - Length between MIN_CHARS and MAX_CHARS
    - Contains at least X: (index) and K: (key) fields
    """
    if not text:
        return False

    n_chars = len(text)
    if n_chars < MIN_CHARS or n_chars > MAX_CHARS:
        return False

    # Check for essential ABC headers
    # We only check the first ~50 lines for performance
    lines = text.splitlines()
    first_lines = lines[:50]

    has_x = any(line.strip().startswith("X:") for line in first_lines)
    has_k = any(line.strip().startswith("K:") for line in first_lines)

    if not (has_x and has_k):
        return False

    return True


def main():
    abc_files = list(ABC_DIR.glob("**/*.abc"))
    print(f"Found {len(abc_files)} ABC files in {ABC_DIR}")

    total = len(abc_files)
    non_empty = 0
    passed_cleaning = 0

    with open(CORPUS_PATH, "w", encoding="utf-8") as out_f:
        for abc_file in tqdm(abc_files):
            try:
                raw = abc_file.read_text(encoding="utf-8", errors="ignore")
                text = raw.strip()
                if not text:
                    continue
                non_empty += 1

                if not is_valid_abc(text):
                    continue

                passed_cleaning += 1
                out_f.write(text)
                out_f.write(SEPARATOR)

            except Exception as e:
                print(f"Failed to read {abc_file}: {e}")

    print("===== Cleaning stats =====")
    print(f"Total ABC files:            {total}")
    print(f"Non-empty ABC files:        {non_empty}")
    print(f"Files used after cleaning:  {passed_cleaning}")
    print(f"Corpus written to {CORPUS_PATH}")


if __name__ == "__main__":
    main()
