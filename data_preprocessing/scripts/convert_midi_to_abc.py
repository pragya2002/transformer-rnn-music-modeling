import subprocess
from pathlib import Path
from tqdm import tqdm

RAW_MIDI_DIR = Path("data/raw_midis")
ABC_DIR = Path("data/abc")

ABC_DIR.mkdir(parents=True, exist_ok=True)

def convert_one_midi(midi_path: Path, abc_path: Path) -> bool:
    """
    Use the external `midi2abc` command to convert a single .mid file to .abc.
    """
    try:
        # midi2abc input.mid -o output.abc
        result = subprocess.run(
            ["midi2abc", str(midi_path), "-o", str(abc_path)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Failed to convert {midi_path}:")
            print(result.stderr)
            return False

        return True

    except FileNotFoundError:
        print("Error: midi2abc command not found. Make sure abcMIDI is installed.")
        return False
    except Exception as e:
        print(f"Unexpected error for {midi_path}: {e}")
        return False

def main():
    midi_files = list(RAW_MIDI_DIR.glob("**/*.mid")) + list(RAW_MIDI_DIR.glob("**/*.midi"))
    print(f"Found {len(midi_files)} MIDI files in {RAW_MIDI_DIR}")

    success = 0
    for midi_file in tqdm(midi_files):
        rel = midi_file.relative_to(RAW_MIDI_DIR)
        abc_file = ABC_DIR / rel.with_suffix(".abc")
        abc_file.parent.mkdir(parents=True, exist_ok=True)

        if convert_one_midi(midi_file, abc_file):
            success += 1

    print(f"Successfully converted {success}/{len(midi_files)} files.")

if __name__ == "__main__":
    main()
