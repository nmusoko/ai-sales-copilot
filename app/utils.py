import re
import json
from datetime import datetime
from pathlib import Path
from mutagen import File as MutagenFile
import wave

# Text cleaning
def clean_transcript(raw_text: str) -> str:
    # Remove common fillers and extra whitespace.
    fillers = ["uh", "um", "you know", "like", "hmm", "ah"]
    text = raw_text
    for f in fillers:
        # \b ensures whole-word match; re.IGNORECASE = not case sensitive
        text = re.sub(rf"\b{f}\b", "", text, flags=re.IGNORECASE)

    # collapse repeated spaces/newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Metadata
def get_audio_duration(audio_path: Path) -> float:
    """
    Returns duration for .wav via wave, and other formats via mutagen.
    """
    try:
        if audio_path.suffix.lower() == ".wav":
            import wave
            with wave.open(str(audio_path), "r") as f:
                frames = f.getnframes()
                rate = f.getframerate()
                return round(frames / float(rate), 2)
        else:
            m = MutagenFile(str(audio_path))
            if m is not None and getattr(m, "info", None) and getattr(m.info, "length", None):
                return round(float(m.info.length), 2)
    except Exception:
        pass
    return 0.0

def save_metadata(audio_path: Path, language: str = "en", model_name: str = "base"):
    # Save metadata alongside transcript.
    info = {
        "file_name": audio_path.name,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "language": language,
        "model": model_name,
        "duration_sec": get_audio_duration(audio_path)
    }
    meta_path = Path("data/transcripts") / f"{audio_path.stem}.meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(info, indent=2))
    print(f"Saved metadata → {meta_path}")
