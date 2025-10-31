from __future__ import annotations
from pathlib import Path
from typing import Iterator, Optional
import json
from pydantic import BaseModel, Field, ValidationError


TRANSCRIPTS_DIR = Path("data/transcripts")


# -----------------------------------------------------------------------------
# 1) Pydantic models (these mirror the minimal fields we rely on now)
#    We can expand later as needed.
# -----------------------------------------------------------------------------

class TranscriptMeta(BaseModel):
    file_name: str = Field(..., description="Original audio filename (e.g., sample_call.wav)")
    created_at: str = Field(..., description="UTC ISO timestamp when the transcript was generated")
    language: str = Field("en", description="Language code used for transcription")
    model: str = Field(..., description="Whisper model name (e.g., base/small/medium)")
    duration_sec: float = Field(0.0, description="Audio duration in seconds (0.0 if unknown)")


class TranscriptPackage(BaseModel):
    # A normalized transcript bundle your LangChain pipeline can consume.
    stem: str = Field(..., description="Base name without extension (e.g., 'sample_call')")
    text: str = Field(..., description="Clean transcript text")
    meta: TranscriptMeta

    @property
    def words(self) -> int:
        return len(self.text.split())

    @property
    def minutes(self) -> float:
        return round(self.meta.duration_sec / 60.0, 2) if self.meta.duration_sec else 0.0

    @property
    def wpm(self) -> Optional[float]:
        if self.meta.duration_sec and self.words:
            return round(self.words / self.minutes, 1) if self.minutes else None
        return None


# -----------------------------------------------------------------------------
# 2) Low-level helpers to load files
# -----------------------------------------------------------------------------

def _txt_path_for_stem(stem: str) -> Path:
    # Build path like data/transcripts/<stem>.txt
    return TRANSCRIPTS_DIR / f"{stem}.txt"


def _meta_path_for_stem(stem: str) -> Path:
    # Build path like data/transcripts/<stem>.meta.json
    return TRANSCRIPTS_DIR / f"{stem}.meta.json"


def _load_text(path: Path) -> str:
    # Read UTF-8 text with a clear error if missing.
    if not path.exists():
        raise FileNotFoundError(f"Transcript text not found: {path}")
    return path.read_text(encoding="utf-8")


def _load_meta(path: Path) -> dict:
    # Load JSON metadata or raise if missing—metadata is important for analytics.
    if not path.exists():
        raise FileNotFoundError(f"Transcript metadata not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# -----------------------------------------------------------------------------
# 3) Public loader functions
# -----------------------------------------------------------------------------

def load_transcript_by_stem(stem: str) -> TranscriptPackage:
    """
    Given a base name like 'sample_call', load:
      - data/transcripts/sample_call.txt
      - data/transcripts/sample_call.meta.json
    Validate metadata with Pydantic, and return a TranscriptPackage.
    """
    txt_path = _txt_path_for_stem(stem)
    meta_path = _meta_path_for_stem(stem)

    text = _load_text(txt_path)
    meta_raw = _load_meta(meta_path)

    try:
        meta = TranscriptMeta(**meta_raw)
    except ValidationError as e:
        # Surfacing a clear message helps debugging bad or partial metadata
        raise ValueError(f"Invalid metadata for {stem}: {e}") from e

    package = TranscriptPackage(stem=stem, text=text, meta=meta)
    return package


def list_transcript_packages() -> Iterator[TranscriptPackage]:
    """
    Iterate all *.txt transcripts in data/transcripts/ and yield TranscriptPackage objects.
    If a file has no matching .meta.json or invalid metadata, we skip it (but warn).
    """
    for txt_path in sorted(TRANSCRIPTS_DIR.glob("*.txt")):
        stem = txt_path.stem
        meta_path = _meta_path_for_stem(stem)
        if not meta_path.exists():
            print(f"[warn] Skipping {stem}: missing metadata file {meta_path.name}")
            continue
        try:
            yield load_transcript_by_stem(stem)
        except Exception as e:
            print(f"[warn] Skipping {stem}: {e}")
            continue
