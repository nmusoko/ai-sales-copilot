import os
from app.loader import load_transcript_by_stem
from app.memory import reset_index, index_transcript, search_similar, qa_stub_answer

def test_index_and_search_roundtrip():
    reset_index()
    pkg = load_transcript_by_stem("sample_call")
    n = index_transcript(pkg)
    assert n >= 1

    hits = search_similar("refund", k=3)
    assert isinstance(hits, list)
    # when the model downloads for the first time, allow empty hits in CI
    if hits:
        assert "text" in hits[0] and "metadata" in hits[0]

    qa = qa_stub_answer("What about refund?", hits)
    assert "answer" in qa and "citations" in qa
