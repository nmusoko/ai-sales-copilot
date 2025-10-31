from app.utils import clean_transcript, save_metadata
import argparse
from pathlib import Path
import whisper 

def load_model(name: str = "base"):
    model= whisper.load_model(name)
    return (model)
def transcribe_file(model, audio_path: Path, language: str = "en") -> str:
    result = model.transcribe(str(audio_path), language=language, fp16=False)
    return result["text"].strip()
def save_text(text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")

def main():
    # Parse CLI args
    parser = argparse.ArgumentParser(description="Transcribe audio with Whisper")
    parser.add_argument("audio_file", type=str, help="Path to .wav/.mp3 file")
    parser.add_argument("--model", type=str, default="base", help="Whisper model name")
    parser.add_argument("--lang", type=str, default="en", help="ISO language code (e.g., en, fr)")
    args = parser.parse_args()

    # Resolve paths
    audio_path = Path(args.audio_file).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # We’ll mirror the CLI output layout: data/transcripts/<basename>.txt
    transcripts_dir = Path("data/transcripts")
    output_txt = transcripts_dir / f"{audio_path.stem}.txt"

    # Load model once (cache is handled by Whisper)
    model = load_model(args.model)

    # Transcribe
    print(f"Transcribing: {audio_path.name} with model={args.model}, lang={args.lang} ...")
    text = transcribe_file(model, audio_path, language=args.lang)

    # Clean transcript
    clean_text = clean_transcript(text)

    # Save metadata
    save_metadata(audio_path, language=args.lang, model_name=args.model)

    # Save transcript
    save_text(clean_text, output_txt)
    print(f"Saved transcript → {output_txt}\n")
    print("--- Clean Transcript Preview ---")
    print(text[:500])  # preview first 500 chars


if __name__ == "__main__":
    main()