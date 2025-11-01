# main.py
# Streamlit viewer for transcripts → summary/sentiment/next-steps + follow-up email
# Works with Stub or LLM, with graceful fallback.

import json
import streamlit as st

from app.loader import list_transcript_packages, load_transcript_by_stem
from app.summarize_stub import summarize_transcript
from app.pipeline import summarize_with_llm, generate_email_with_llm
from app.email_stub import email_stub


# -----------------------------
# Small helpers for downloads
# -----------------------------
def to_markdown(summary: str, sentiment: str, key_phrases: list[str], next_steps: list[str]) -> str:
    """Build a simple Markdown export."""
    lines = [
        "# AI Sales Copilot — Summary",
        "",
        "## Summary",
        summary or "_(empty)_",
        "",
        "## Sentiment",
        f"- {sentiment}",
        "",
        "## Key Phrases",
    ]
    if key_phrases:
        lines.extend([f"- {k}" for k in key_phrases])
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append("## Next Steps")
    if next_steps:
        lines.extend([f"- {s}" for s in next_steps])
    else:
        lines.append("- (none)")

    lines.append("")
    return "\n".join(lines)


def to_json(summary: str, sentiment: str, key_phrases: list[str], next_steps: list[str], meta: dict) -> str:
    """Build a JSON export (string)."""
    payload = {
        "meeting_summary": summary,
        "sentiment": sentiment,
        "key_phrases": key_phrases,
        "next_steps": next_steps,
        "meta": meta,
    }
    return json.dumps(payload, indent=2)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Sales Copilot", layout="wide")
st.title("🤖 AI Sales Copilot — Transcript Viewer")
st.caption("Upload or select a transcript to preview summary, sentiment, next steps, and a follow-up email. Toggle LLM when available.")

tab_select, tab_upload = st.tabs(["Select Existing", "Upload .txt Transcript"])

# ============================
# Tab: Select Existing
# ============================
with tab_select:
    pkgs = list(list_transcript_packages())
    if not pkgs:
        st.info("No transcripts found in `data/transcripts/`. Use the Upload tab or run the Whisper step first.")
    else:
        options = [p.stem for p in pkgs]
        choice = st.selectbox("Choose a transcript:", options, key="choice_existing")
        if choice:
            pkg = load_transcript_by_stem(choice)

            left, right = st.columns([1, 2], gap="large")
            with left:
                st.subheader("Metadata")
                st.markdown(f"**File:** {pkg.meta.file_name}")
                st.markdown(f"**Language:** {pkg.meta.language}")
                st.markdown(f"**Model:** {pkg.meta.model}")
                st.markdown(f"**Duration (sec):** {pkg.meta.duration_sec}")
                st.markdown(f"**Words:** {pkg.words}")
                st.markdown(f"**WPM:** {pkg.wpm if pkg.wpm is not None else 'N/A'}")
                st.markdown(f"**Created:** {pkg.meta.created_at}")
                with st.expander("Raw Text Preview"):
                    st.write(pkg.text[:2000] + ("…" if len(pkg.text) > 2000 else ""))

            with right:
                st.subheader("Analysis")

                # Toggle for LLM summary
                use_llm = st.toggle(
                    "Use LLM (LangChain)",
                    value=False,
                    help="Turn on to use the real LLM summarizer. If it fails (e.g., quota), the app will fall back to the stub.",
                    key="analysis_llm_existing"
                )

                # Run analysis (LLM or Stub) with fallback
                try:
                    if use_llm:
                        out = summarize_with_llm(pkg.text, model_name="gpt-4o-mini")
                        st.caption("Powered by LangChain + OpenAI")
                        summary_text = out.get("meeting_summary", "")
                    else:
                        out = summarize_transcript(pkg.text, max_sentences=4)
                        st.caption("Stub (deterministic)")
                        summary_text = out.get("summary", "")
                except RuntimeError:
                    st.warning("LLM unavailable (likely quota/rate-limit). Falling back to stub automatically.")
                    out = summarize_transcript(pkg.text, max_sentences=4)
                    st.caption("Stub (fallback after LLM error)")
                    summary_text = out.get("summary", "")

                # Render analysis
                st.markdown("### Summary")
                st.code(summary_text or "(empty)", language="markdown")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Sentiment")
                    st.info(out.get("sentiment", "Neutral"))
                with col2:
                    st.markdown("### Key Phrases")
                    kp = out.get("key_phrases", [])
                    st.write(", ".join(kp) if kp else "(none)")

                st.markdown("### Next Steps")
                steps = out.get("next_steps", [])
                if steps:
                    for step in steps:
                        st.markdown(f"- {step}")
                else:
                    st.write("(none)")

                # Downloads for analysis
                st.divider()
                md = to_markdown(summary_text, out.get("sentiment", "Neutral"), kp, steps)
                js = to_json(summary_text, out.get("sentiment", "Neutral"), kp, steps, meta=pkg.meta.model_dump())
                colA, colB = st.columns(2)
                with colA:
                    st.download_button("⬇️ Download as Markdown", data=md, file_name=f"{pkg.stem}_summary.md", mime="text/markdown")
                with colB:
                    st.download_button("⬇️ Download as JSON", data=js, file_name=f"{pkg.stem}_summary.json", mime="application/json")

                # -------------------------
                # Email section
                # -------------------------
                st.markdown("## 📧 Follow-up Email")

                email_tone = st.selectbox(
                    "Tone",
                    options=["friendly", "formal", "concise", "persuasive"],
                    index=0,
                    help="Choose how the email should sound.",
                    key="tone_existing"
                )
                use_llm_email = st.toggle(
                    "Use LLM for Email",
                    value=False,
                    help="Turn on to generate the email with the LLM; otherwise use the stub.",
                    key="llm_email_existing"
                )

                try:
                    if use_llm_email:
                        email_out = generate_email_with_llm(
                            summary=summary_text,
                            sentiment=out.get("sentiment", "Neutral"),
                            next_steps=steps,
                            tone=email_tone,
                            model_name="gpt-4o-mini"
                        )
                        st.caption("Email powered by LangChain + OpenAI")
                    else:
                        email_out = email_stub(
                            summary=summary_text,
                            sentiment=out.get("sentiment", "Neutral"),
                            next_steps=steps,
                            tone=email_tone
                        )
                        st.caption("Email (stub)")
                except RuntimeError:
                    st.warning("LLM unavailable for email (quota/rate-limit). Using stub automatically.")
                    email_out = email_stub(
                        summary=summary_text,
                        sentiment=out.get("sentiment", "Neutral"),
                        next_steps=steps,
                        tone=email_tone
                    )
                    st.caption("Email (stub fallback)")

                st.markdown("**Subject**")
                st.code(email_out.get("subject") or "(empty)", language="markdown")
                st.markdown("**Body (Markdown)**")
                st.code(email_out.get("body_markdown") or "(empty)", language="markdown")

                # Download buttons for the email
                st.divider()
                colE1, colE2 = st.columns(2)
                colE1.download_button(
                    "⬇️ Download Email (.md)",
                    data=email_out.get("body_markdown", ""),
                    file_name=f"{pkg.stem}_email.md",
                    mime="text/markdown"
                )
                colE2.download_button(
                    "⬇️ Download Email (.json)",
                    data=json.dumps(email_out, indent=2),
                    file_name=f"{pkg.stem}_email.json",
                    mime="application/json"
                )

# ============================
# Tab: Upload
# ============================
with tab_upload:
    st.write("Upload a plain text transcript (.txt).")
    up = st.file_uploader("Choose .txt file", type=["txt"], key="uploader_txt")
    if up is not None:
        text = up.read().decode("utf-8", errors="ignore")
        st.success("Transcript loaded from upload.")

        # Toggle for LLM summary (upload)
        use_llm_upload = st.toggle("Use LLM (for uploaded text)", value=False, key="analysis_llm_upload")

        try:
            if use_llm_upload:
                out = summarize_with_llm(text, model_name="gpt-4o-mini")
                st.caption("Powered by LangChain + OpenAI")
                summary_text = out.get("meeting_summary", "")
            else:
                out = summarize_transcript(text, max_sentences=4)
                st.caption("Stub (deterministic)")
                summary_text = out.get("summary", "")
        except RuntimeError:
            st.warning("LLM unavailable (likely quota/rate-limit). Falling back to stub automatically.")
            out = summarize_transcript(text, max_sentences=4)
            st.caption("Stub (fallback after LLM error)")
            summary_text = out.get("summary", "")

        st.markdown("### Summary")
        st.code(summary_text or "(empty)", language="markdown")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Sentiment")
            st.info(out.get("sentiment", "Neutral"))
        with col2:
            st.markdown("### Key Phrases")
            kp = out.get("key_phrases", [])
            st.write(", ".join(kp) if kp else "(none)")

        st.markdown("### Next Steps")
        steps = out.get("next_steps", [])
        if steps:
            for step in steps:
                st.markdown(f"- {step}")
        else:
            st.write("(none)")

        # Downloads for analysis (upload)
        st.divider()
        md = to_markdown(summary_text, out.get("sentiment", "Neutral"), kp, steps)
        js = to_json(summary_text, out.get("sentiment", "Neutral"), kp, steps, meta={"source": "upload"})
        colA, colB = st.columns(2)
        with colA:
            st.download_button("⬇️ Download as Markdown", data=md, file_name="upload_summary.md", mime="text/markdown")
        with colB:
            st.download_button("⬇️ Download as JSON", data=js, file_name="upload_summary.json", mime="application/json")

        # -------------------------
        # Email section (upload)
        # -------------------------
        st.markdown("## 📧 Follow-up Email")
        email_tone_up = st.selectbox(
            "Tone",
            options=["friendly", "formal", "concise", "persuasive"],
            index=0,
            help="Choose how the email should sound.",
            key="tone_upload"
        )
        use_llm_email_up = st.toggle(
            "Use LLM for Email",
            value=False,
            help="Turn on to generate the email with the LLM; otherwise use the stub.",
            key="llm_email_upload"
        )

        try:
            if use_llm_email_up:
                email_out = generate_email_with_llm(
                    summary=summary_text,
                    sentiment=out.get("sentiment", "Neutral"),
                    next_steps=steps,
                    tone=email_tone_up,
                    model_name="gpt-4o-mini"
                )
                st.caption("Email powered by LangChain + OpenAI")
            else:
                email_out = email_stub(
                    summary=summary_text,
                    sentiment=out.get("sentiment", "Neutral"),
                    next_steps=steps,
                    tone=email_tone_up
                )
                st.caption("Email (stub)")
        except RuntimeError:
            st.warning("LLM unavailable for email (quota/rate-limit). Using stub automatically.")
            email_out = email_stub(
                summary=summary_text,
                sentiment=out.get("sentiment", "Neutral"),
                next_steps=steps,
                tone=email_tone_up
            )
            st.caption("Email (stub fallback)")

        st.markdown("**Subject**")
        st.code(email_out.get("subject") or "(empty)", language="markdown")
        st.markdown("**Body (Markdown)**")
        st.code(email_out.get("body_markdown") or "(empty)", language="markdown")

        # Download buttons for the email (upload)
        st.divider()
        colE1, colE2 = st.columns(2)
        colE1.download_button(
            "⬇️ Download Email (.md)",
            data=email_out.get("body_markdown", ""),
            file_name="upload_email.md",
            mime="text/markdown"
        )
        colE2.download_button(
            "⬇️ Download Email (.json)",
            data=json.dumps(email_out, indent=2),
            file_name="upload_email.json",
            mime="application/json"
        )
