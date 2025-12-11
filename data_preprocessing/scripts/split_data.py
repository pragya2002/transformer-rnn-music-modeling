from pathlib import Path
import numpy as np

TOKENIZED_DIR = Path("data/tokenized")
ALL_IDS_PATH = TOKENIZED_DIR / "all_ids.npy"

TRAIN_TOKENS = 100_000_000
VAL_TOKENS = 25_000_000
TEST_TOKENS = 25_000_000

def main():
    print("Loading token array (memory mapped)...")
    ids = np.load(ALL_IDS_PATH, mmap_mode="r")
    n = ids.shape[0]
    print(f"Total tokens available: {n}")

    assert n >= TRAIN_TOKENS + VAL_TOKENS + TEST_TOKENS, \
        "Not enough tokens to split!"

    train_ids = ids[:TRAIN_TOKENS]
    val_ids = ids[TRAIN_TOKENS:TRAIN_TOKENS + VAL_TOKENS]
    test_ids = ids[TRAIN_TOKENS + VAL_TOKENS : TRAIN_TOKENS + VAL_TOKENS + TEST_TOKENS]

    print("Saving splits...")
    np.save(TOKENIZED_DIR / "train_ids.npy", train_ids)
    np.save(TOKENIZED_DIR / "val_ids.npy", val_ids)
    np.save(TOKENIZED_DIR / "test_ids.npy", test_ids)

    print("Done!")
    print(f"Train tokens: {train_ids.shape[0]}")
    print(f"Val tokens:   {val_ids.shape[0]}")
    print(f"Test tokens:  {test_ids.shape[0]}")

if __name__ == "__main__":
    main()
