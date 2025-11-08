from app.loader import load_transcript_by_stem

def test_load_sample_call():
    pkg = load_transcript_by_stem("sample_call")
    assert pkg.stem == "sample_call"
    assert isinstance(pkg.text, str) and pkg.text.strip()

    # file_name can be the original audio or the transcript file
    fname = pkg.meta.file_name
    assert "sample_call" in fname
    assert any(
        fname.endswith(ext) for ext in [".txt", ".mp3", ".wav", ".m4a", ".flac"]
    )

    # optional extra sanity checks
    assert isinstance(pkg.meta.language, str) and pkg.meta.language
    assert isinstance(pkg.meta.model, str) and pkg.meta.model
