"""
app/summarize_stub.py — deterministic, LLM-free summarization for debugging.

Usage:
    python - << 'PY'
from app.loader import load_transcript_by_stem
from app.summarize_stub import summarize_transcript

pkg = load_transcript_by_stem("sample_call")
result = summarize_transcript(pkg.text)
print(result["summary"])
print("Sentiment:", result["sentiment"])
print("Key phrases:", result["key_phrases"])
print("Next steps:", result["next_steps"])
PY
"""

from __future__ import annotations
import re
from collections import Counter
from typing import List, Dict


# --------------------------
# Sentence utilities
# --------------------------
def split_sentences(text: str) -> List[str]:
    """
    Very simple sentence splitter.
    """
    # normalize whitespace
    t = re.sub(r"\s+", " ", text).strip()
    # split on ., ?, !
    parts = re.split(r"(?<=[\.\!\?])\s+", t)
    # keep non-empty
    return [p.strip() for p in parts if p.strip()]


def tokenize(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r"\b\w+\b", text)]


# --------------------------
# Naive keywording & scoring
# --------------------------
STOPWORDS = set("""
a an and are as at be by for from has have i if in is it its of on or our so that the their there they this to was we were will with you your
""".split())

def top_keywords(text: str, k: int = 12) -> List[str]:
    tokens = [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 2]
    freqs = Counter(tokens)
    return [w for (w, _) in freqs.most_common(k)]


def score_sentence(sent: str, keywords: List[str]) -> int:
    """
    Score by keyword overlap (tiny heuristic).
    """
    words = set(tokenize(sent))
    return sum(1 for kw in keywords if kw in words)


# --------------------------
# Naive sentiment
# --------------------------
POS_WORDS = {"great","good","glad","happy","love","excited","appreciate","thanks","agree","yes","pleased","awesome","excellent"}
NEG_WORDS = {"bad","angry","sad","hate","upset","refund","issue","problem","delay","late","no","concern","confused","terrible"}

def naive_sentiment(text: str) -> str:
    tokens = set(tokenize(text))
    pos = len(tokens & POS_WORDS)
    neg = len(tokens & NEG_WORDS)
    if pos > neg: return "Positive"
    if neg > pos: return "Negative"
    return "Neutral"


# --------------------------
# Extract rough "next steps"
# --------------------------
NEXT_STEP_PATTERNS = [
    # Refund / replace / return / exchange
    r"\b(refund|replace|exchange|return)\b.*?\b(headphones?|item|order|product|pair)\b",
    r"\b(send|ship|dispatch)\b.*?\b(new|replacement)\b.*?\b(headphones?|item|unit|pair)\b",
    # Escalation to manager/supervisor
    r"\b(put|get|speak|talk|transfer|escalate)\b.*?\b(manager|supervisor)\b",
    # Confirm / provide docs or policy
    r"\b(provide|show|share)\b.*?\b(guarantee|warranty|policy|receipt|proof)\b",
    # Generic sales actions
    r"\b(send|share|email|deliver|provide)\b.*?\b(proposal|quote|pricing|invoice|demo|document|deck)\b",
    r"\b(schedule|set up|arrange|book)\b.*?\b(call|meeting|demo)\b",
    r"\b(follow up|follow-up|circle back|touch base)\b",
    r"\b(confirm|finalize|review|sign)\b.*?\b(agreement|contract|scope|terms)\b",
]

def extract_next_steps(text: str, max_items: int = 5) -> List[str]:
    steps: List[str] = []

    # 1) Pattern-based extraction
    for pat in NEXT_STEP_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            raw = m.group(0)
            raw = re.sub(r"\s+", " ", raw).strip().rstrip(".")
            # Normalize a few common actions to clearer task language
            norm = raw
            norm = re.sub(r"(?i)\brefund\b.*\bheadphones?\b", "Process refund for customer's headphones", norm)
            norm = re.sub(r"(?i)\b(send|ship|dispatch)\b.*\b(replacement|new)\b.*\b(headphones?|pair|item|unit)\b", "Ship replacement headphones to customer", norm)
            norm = re.sub(r"(?i)\b(put|get|speak|talk|transfer|escalate)\b.*\b(manager|supervisor)\b", "Escalate call to a manager/supervisor", norm)
            norm = re.sub(r"(?i)\bprovide\b.*\b(guarantee|warranty|policy|receipt|proof)\b", "Provide warranty/guarantee policy details", norm)

            if norm not in steps:
                steps.append(norm)
            if len(steps) >= max_items:
                return steps

    # 2) Imperative fallback if nothing matched
    if not steps:
        for raw in _imperative_snippets(text, max_items=max_items):
            cleaned = raw
            cleaned = re.sub(r"(?i)^refund .*headphones?.*", "Process refund for customer's headphones", cleaned)
            cleaned = re.sub(r"(?i)^(send|ship|dispatch) .* (replacement|new) .* (headphones?|pair|item|unit).*", "Ship replacement headphones to customer", cleaned)
            cleaned = re.sub(r"(?i)^(put|get|speak|talk|transfer|escalate) .* (manager|supervisor).*", "Escalate call to a manager/supervisor", cleaned)
            if cleaned not in steps:
                steps.append(cleaned)
            if len(steps) >= max_items:
                break

    return steps

def _imperative_snippets(text: str, max_items: int = 5):
    # Look for sentences that start with a verb-like token (very naive)
    sents = split_sentences(text)
    picks = []
    for s in sents:
        # first word (lowercased, letters only)
        m = re.match(r"^\s*([A-Za-z']+)", s)
        if not m:
            continue
        first = m.group(1).lower()
        if first in {"refund","replace","exchange","return","send","ship","dispatch","put","get","speak","talk","transfer","escalate","provide","show","share","confirm"}:
            snippet = re.sub(r"\s+", " ", s).strip().rstrip(".")
            if snippet not in picks:
                picks.append(snippet)
            if len(picks) >= max_items:
                break
    return picks


# --------------------------
# Main stub
# --------------------------
def summarize_transcript(text: str, max_sentences: int = 4) -> Dict:
    sents = split_sentences(text)
    if not sents:
        return {
            "summary": "",
            "sentiment": "Neutral",
            "key_phrases": [],
            "next_steps": []
        }

    keywords = top_keywords(text)
    # score each sentence by keyword overlap
    scored = sorted(sents, key=lambda s: score_sentence(s, keywords), reverse=True)

    # choose top-N sentences (stable order by original index for readability)
    top = set(scored[:max_sentences])
    ordered_top = [s for s in sents if s in top][:max_sentences]

    # naive sentiment over whole transcript
    sentiment = naive_sentiment(text)
    steps = extract_next_steps(text)

    # build a compact summary with bullets
    bullets = "\n".join([f"- {s}" for s in ordered_top])
    summary = f"Key points:\n{bullets}" if bullets else ""

    return {
        "summary": summary,
        "sentiment": sentiment,
        "key_phrases": keywords,
        "next_steps": steps
    }
