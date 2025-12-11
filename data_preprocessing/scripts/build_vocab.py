from pathlib import Path
from collections import Counter
import json

CORPUS_PATH = Path("data/corpus/abc_corpus.txt")
TOKENIZED_DIR = Path("data/tokenized")
TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)

# We'll read the corpus in chunks so we don't load 3.6GB into RAM at once.
CHUNK_SIZE = 1_000_000  # 1 MB of characters per read


def main():
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Corpus file not found at {CORPUS_PATH}")

    counter = Counter()
    total_chars = 0

    print(f"Building vocabulary from {CORPUS_PATH} ...")
    with CORPUS_PATH.open("r", encoding="utf-8", errors="ignore") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            counter.update(chunk)
            total_chars += len(chunk)

    print(f"Total characters (approx tokens): {total_chars}")
    print(f"Unique characters: {len(counter)}")

    # Sort characters to have a stable order (for reproducibility)
    chars = sorted(counter.keys())
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    vocab_path = TOKENIZED_DIR / "vocab.json"
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "stoi": stoi,
                "itos": itos,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Vocabulary saved to {vocab_path}")
    print("Some stats:")
    print(f"- Vocab size: {len(stoi)}")
    # Show a few example characters
    print("Example characters:")
    for ch in list(stoi.keys())[:20]:
        printable = ch if ch not in ["\n", "\t", " "] else repr(ch)
        print(f"  {repr(printable)} -> {stoi[ch]}")


if __name__ == "__main__":
    main()
