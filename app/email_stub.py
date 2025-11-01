# app/email_stub.py
from datetime import date

TONE_PREFIX = {
    "friendly": "Hope you're doing well!",
    "formal": "I trust this message finds you well.",
    "concise": "Quick follow-up from our call.",
    "persuasive": "Following up with next steps to keep momentum.",
}

def email_stub(summary: str, sentiment: str, next_steps: list[str], tone: str = "friendly") -> dict:
    """Deterministic email generator for offline use / fallback."""
    prefix = TONE_PREFIX.get(tone, TONE_PREFIX["friendly"])
    subject = "Follow-up and next steps"
    if "Negative" in sentiment:
        subject = "Following up on concerns and next steps"
    elif "Positive" in sentiment:
        subject = "Great speaking today — next steps"

    bullet_lines = "\n".join(f"- {s}" for s in (next_steps or [])) or "- (no action items captured)"
    body = f"""Hi there,

{prefix}

**Recap**
{summary or "(summary unavailable)"}

**Next Steps**
{bullet_lines}

If anything above needs adjusting, just reply here. Looking forward to moving ahead.

Best regards,
{date.today().isoformat()}
"""
    return {"subject": subject, "body_markdown": body}
