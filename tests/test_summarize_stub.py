import re
from app.summarize_stub import summarize_transcript

def test_stub_returns_shape():
    txt = "Customer asks about refund policy. Agent explains the steps. Follow up next week."
    out = summarize_transcript(txt, max_sentences=3)
    assert set(out.keys()) == {"summary", "sentiment", "key_phrases", "next_steps"}
    assert isinstance(out["summary"], str)
    assert out["sentiment"] in {"Positive", "Neutral", "Negative"}
    assert isinstance(out["key_phrases"], list)
    assert isinstance(out["next_steps"], list)

def test_stub_has_content():
    txt = "Refund was requested. Warranty is discussed."
    out = summarize_transcript(txt, max_sentences=2)
    assert "refund" in out["summary"].lower() or "warranty" in out["summary"].lower()
