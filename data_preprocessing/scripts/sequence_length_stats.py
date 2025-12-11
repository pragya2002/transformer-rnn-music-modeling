from pathlib import Path
import numpy as np
from tqdm import tqdm

ABC_DIR = Path("data/abc")

def main():
    lengths = []

    abc_files = list(ABC_DIR.glob("**/*.abc"))
    print(f"Found {len(abc_files)} ABC files")

    for f in tqdm(abc_files):
        text = f.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        lengths.append(len(text))

    lengths = np.array(lengths)

    print("===== Sequence Length Stats =====")
    print(f"Total ABC files (non-empty): {len(lengths)}")
    print(f"Mean length:   {lengths.mean():.2f} chars")
    print(f"Median length: {np.median(lengths):.2f} chars")
    print(f"Min length:    {lengths.min()} chars")
    print(f"Max length:    {lengths.max()} chars")
    print(f"90th percentile: {np.percentile(lengths, 90):.2f} chars")
    print(f"95th percentile: {np.percentile(lengths, 95):.2f} chars")
    print(f"99th percentile: {np.percentile(lengths, 99):.2f} chars")

if __name__ == "__main__":
    main()
