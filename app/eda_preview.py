from pathlib import Path
import json
import re
from collections import Counter

TRANSCRIPTS_DIR = Path("data/transcripts")

def list_transcripts():
    # Find all transcript .txt files in the transcripts directory.
    return sorted(TRANSCRIPTS_DIR.glob("*.txt"))

def load_text(path: Path) -> str:
    # Read a UTF-8 text file.
    return path.read_text(encoding="utf-8")

def load_meta(path_stem: Path) -> dict:
    # Load the companion .meta.json if it exists.
    meta_path = path_stem.with_suffix(".meta.json")
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}

def basic_stats(text: str, duration_sec: float | int = 0):
    # Compute simple metrics: length, word count, and words-per-minute (if duration available).
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)
    char_count = len(text)
    wpm = round((word_count / (duration_sec / 60.0)), 1) if duration_sec else None
    return {
        "chars": char_count,
        "words": word_count,
        "wpm": wpm,
    }

def top_keywords(text: str, k: int = 10):
    # Very naive keyword frequency (stopword-less for simplicity).
    tokens = [t.lower() for t in re.findall(r"\b\w+\b", text)]
    counts = Counter(tokens)
    return counts.most_common(k)

def preview(text: str, n: int = 240):
    # First n characters for a quick glance.
    return (text[:n] + "…") if len(text) > n else text

def main():
    paths = list_transcripts()
    if not paths:
        print("No transcripts found in data/transcripts/. Run the transcription step first.")
        return

    print(f"Found {len(paths)} transcript(s).\n")

    for txt_path in paths:
        meta = load_meta(txt_path)
        text = load_text(txt_path)
        stats = basic_stats(text, duration_sec=meta.get("duration_sec", 0))

        print("=" * 80)
        print(f"File: {txt_path.name}")
        if meta:
            print(f" - Language: {meta.get('language')}")
            print(f" - Duration: {meta.get('duration_sec')} sec")
            print(f" - Model:    {meta.get('model')}")
            print(f" - Created:  {meta.get('created_at')}")
        print(f" - Chars: {stats['chars']} | Words: {stats['words']} | WPM: {stats['wpm']}")
        print("\nPreview:")
        print(preview(text))
        print("\nTop keywords:")
        for word, count in top_keywords(text, k=10):
            print(f"  {word:<15} {count}")
        print()

    print("=" * 80)
    print("EDA complete.")


if __name__ == "__main__":
    main()
