from pathlib import Path
import json
import numpy as np
from numpy.lib.format import open_memmap

CORPUS_PATH = Path("data/corpus/abc_corpus.txt")
TOKENIZED_DIR = Path("data/tokenized")
TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)

VOCAB_PATH = TOKENIZED_DIR / "vocab.json"

# How many tokens (characters) we want to use in total.
# 150M gives us plenty for 100M train + val + test.
TOTAL_TOKENS_TO_USE = 150_000_000

CHUNK_SIZE = 1_000_000  # read 1M characters at a time


def main():
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Corpus file not found at {CORPUS_PATH}")
    if not VOCAB_PATH.exists():
        raise FileNotFoundError(f"Vocab file not found at {VOCAB_PATH}")

    # Load vocab (stoi: char -> int)
    with VOCAB_PATH.open("r", encoding="utf-8") as f:
        vocab = json.load(f)
    stoi = vocab["stoi"]

    print(f"Loaded vocab with {len(stoi)} characters")
    print(f"Target total tokens: {TOTAL_TOKENS_TO_USE}")

    # Create a memory-mapped array on disk for all ids
    all_ids_path = TOKENIZED_DIR / "all_ids.npy"
    ids = open_memmap(
        all_ids_path,
        mode="w+",
        dtype=np.int32,
        shape=(TOTAL_TOKENS_TO_USE,),
    )

    n_written = 0

    with CORPUS_PATH.open("r", encoding="utf-8", errors="ignore") as f:
        while n_written < TOTAL_TOKENS_TO_USE:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break  # end of file

            # If this chunk would overshoot, trim it
            remaining = TOTAL_TOKENS_TO_USE - n_written
            if len(chunk) > remaining:
                chunk = chunk[:remaining]

            # Convert characters in this chunk to ids
            for ch in chunk:
                # Every char should be in vocab; if not, you could map to a special token
                idx = stoi.get(ch)
                if idx is None:
                    # Fallback: skip or map to 0; here we skip, but this should almost never happen
                    continue
                ids[n_written] = idx
                n_written += 1
                if n_written >= TOTAL_TOKENS_TO_USE:
                    break

            if n_written % 10_000_000 == 0:
                print(f"Wrote {n_written} tokens so far...")

    print(f"Finished tokenization. Total tokens written: {n_written}")
    if n_written < TOTAL_TOKENS_TO_USE:
        print("Note: corpus ended before reaching TARGET; shrinking array.")
        # Optionally shrink to actual size
        # But np.memmap can't be resized in-place; we can save a smaller copy:
        final_ids = np.array(ids[:n_written], dtype=np.int32)
        np.save(all_ids_path, final_ids)
        print(f"Saved resized all_ids.npy with shape {final_ids.shape}")
    else:
        print(f"all_ids.npy saved at {all_ids_path} with shape {ids.shape}")


if __name__ == "__main__":
    main()
