# main.py
# Streamlit app — AI Sales Copilot
# Transcript viewer → summary / sentiment / next steps / follow-up email / knowledge search
# Supports Stub + LLM modes, plus FAISS vector memory controls and PDF export

import json
import streamlit as st

from app.loader import list_transcript_packages, load_transcript_by_stem
from app.summarize_stub import summarize_transcript
from app.pipeline import (
    summarize_with_llm,
    generate_email_with_llm,
    generate_qa_with_llm,  # QA chain for Knowledge Search
)
from app.email_stub import email_stub
from app.export_pdf import create_summary_pdf


# -----------------------------
# Helpers
# -----------------------------
def to_markdown(summary, sentiment, key_phrases, next_steps):
    """Build a Markdown export."""
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
        lines.extend(f"- {k}" for k in key_phrases)
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("## Next Steps")
    if next_steps:
        lines.extend(f"- {s}" for s in next_steps)
    else:
        lines.append("- (none)")
    lines.append("")
    return "\n".join(lines)


def to_json(summary, sentiment, key_phrases, next_steps, meta):
    payload = {
        "meeting_summary": summary,
        "sentiment": sentiment,
        "key_phrases": key_phrases,
        "next_steps": next_steps,
        "meta": meta,
    }
    return json.dumps(payload, indent=2)


# -----------------------------
# Streamlit Layout
# -----------------------------
st.set_page_config(page_title="AI Sales Copilot", layout="wide")
st.title("🤖 AI Sales Copilot")
st.caption("Summarize calls, draft follow-up emails, search across conversations, and export polished reports.")

tab_select, tab_upload, tab_search = st.tabs(
    ["Select Existing", "Upload Transcript", "Knowledge Search"]
)


# ============================================================
# TAB 1 — SELECT EXISTING
# ============================================================
with tab_select:
    pkgs = list(list_transcript_packages())
    if not pkgs:
        st.info("No transcripts found in `data/transcripts/`.")
    else:
        choice = st.selectbox("Choose a transcript:", [p.stem for p in pkgs], key="sel_choice")
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
                st.markdown(f"**WPM:** {pkg.wpm if pkg.wpm else 'N/A'}")
                st.markdown(f"**Created:** {pkg.meta.created_at}")
                with st.expander("Raw Text Preview"):
                    st.write(pkg.text[:2000] + ("…" if len(pkg.text) > 2000 else ""))

            with right:
                st.subheader("Analysis")
                use_llm = st.toggle("Use LLM (LangChain)", value=False, key="sel_llm")

                try:
                    if use_llm:
                        out = summarize_with_llm(pkg.text, model_name="gpt-4o-mini")
                        summary_text = out.get("meeting_summary", "")
                        st.caption("Powered by LangChain + OpenAI")
                    else:
                        out = summarize_transcript(pkg.text, max_sentences=4)
                        summary_text = out.get("summary", "")
                        st.caption("Stub (deterministic)")
                except RuntimeError:
                    st.warning("LLM unavailable. Falling back to stub.")
                    out = summarize_transcript(pkg.text, max_sentences=4)
                    summary_text = out.get("summary", "")

                st.markdown("### Summary")
                st.code(summary_text or "(empty)", language="markdown")

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("### Sentiment")
                    st.info(out.get("sentiment", "Neutral"))
                with c2:
                    st.markdown("### Key Phrases")
                    kp = out.get("key_phrases", [])
                    st.write(", ".join(kp) if kp else "(none)")

                st.markdown("### Next Steps")
                steps = out.get("next_steps", [])
                if steps:
                    for s in steps:
                        st.markdown(f"- {s}")
                else:
                    st.write("(none)")

                st.divider()
                md = to_markdown(summary_text, out.get("sentiment", "Neutral"), kp, steps)
                js = to_json(
                    summary_text,
                    out.get("sentiment", "Neutral"),
                    kp,
                    steps,
                    pkg.meta.model_dump(),
                )
                cA, cB = st.columns(2)
                cA.download_button("⬇️ Markdown", md, f"{pkg.stem}_summary.md")
                cB.download_button("⬇️ JSON", js, f"{pkg.stem}_summary.json")

                # ----------------- Email
                st.markdown("## 📧 Follow-up Email")
                tone = st.selectbox("Tone", ["friendly", "formal", "concise", "persuasive"], key="sel_tone")
                use_llm_email = st.toggle("Use LLM for Email", value=False, key="sel_llm_email")
                try:
                    if use_llm_email:
                        email_out = generate_email_with_llm(
                            summary=summary_text,
                            sentiment=out.get("sentiment", "Neutral"),
                            next_steps=steps,
                            tone=tone,
                            model_name="gpt-4o-mini",
                        )
                        st.caption("Email powered by LangChain + OpenAI")
                    else:
                        email_out = email_stub(
                            summary=summary_text,
                            sentiment=out.get("sentiment", "Neutral"),
                            next_steps=steps,
                            tone=tone,
                        )
                        st.caption("Email (stub)")
                except RuntimeError:
                    st.warning("LLM unavailable for email; using stub.")
                    email_out = email_stub(
                        summary=summary_text,
                        sentiment=out.get("sentiment", "Neutral"),
                        next_steps=steps,
                        tone=tone,
                    )

                st.markdown("**Subject**")
                st.code(email_out.get("subject") or "(empty)")
                st.markdown("**Body**")
                st.code(email_out.get("body_markdown") or "(empty)")

                cE1, cE2 = st.columns(2)
                cE1.download_button(
                    "⬇️ Email (.md)",
                    email_out.get("body_markdown", ""),
                    f"{pkg.stem}_email.md",
                )
                cE2.download_button(
                    "⬇️ Email (.json)",
                    json.dumps(email_out, indent=2),
                    f"{pkg.stem}_email.json",
                )

                # ----------------- PDF Export (analysis + email)
                pdf_bytes = create_summary_pdf(
                    title=f"{pkg.stem} — Meeting Report",
                    meta=pkg.meta.model_dump(),
                    summary=summary_text,
                    sentiment=out.get("sentiment", "Neutral"),
                    key_phrases=kp,
                    next_steps=steps,
                    email_subject=email_out.get("subject"),
                    email_body_md=email_out.get("body_markdown"),
                )
                st.download_button(
                    "⬇️ Download Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"{pkg.stem}_report.pdf",
                    mime="application/pdf",
                )


# ============================================================
# TAB 2 — UPLOAD
# ============================================================
with tab_upload:
    st.write("Upload a plain text transcript (.txt).")
    up = st.file_uploader("Choose file", type=["txt"], key="up_file")
    if up:
        text = up.read().decode("utf-8", errors="ignore")
        st.success("Transcript loaded.")
        use_llm_upload = st.toggle("Use LLM", False, key="up_llm")
        try:
            if use_llm_upload:
                out = summarize_with_llm(text, model_name="gpt-4o-mini")
                summary_text = out.get("meeting_summary", "")
                st.caption("Powered by LangChain + OpenAI")
            else:
                out = summarize_transcript(text, max_sentences=4)
                summary_text = out.get("summary", "")
                st.caption("Stub (deterministic)")
        except RuntimeError:
            st.warning("LLM unavailable. Fallback stub.")
            out = summarize_transcript(text, max_sentences=4)
            summary_text = out.get("summary", "")

        st.markdown("### Summary")
        st.code(summary_text or "(empty)", language="markdown")
        st.markdown("### Sentiment")
        st.info(out.get("sentiment", "Neutral"))

        # We will still collect key phrases and steps for exports/email
        kp = out.get("key_phrases", [])
        steps = out.get("next_steps", [])

        st.divider()
        md = to_markdown(summary_text, out.get("sentiment", "Neutral"), kp, steps)
        js = to_json(summary_text, out.get("sentiment", "Neutral"), kp, steps, {"source": "upload"})
        st.download_button("⬇️ Markdown", md, "upload_summary.md")
        st.download_button("⬇️ JSON", js, "upload_summary.json")

        # -------- Email for upload
        st.markdown("## 📧 Follow-up Email")
        tone_up = st.selectbox("Tone", ["friendly", "formal", "concise", "persuasive"], key="up_tone")
        use_llm_email_up = st.toggle("Use LLM for Email", False, key="up_llm_email")
        try:
            if use_llm_email_up:
                email_out = generate_email_with_llm(
                    summary=summary_text,
                    sentiment=out.get("sentiment", "Neutral"),
                    next_steps=steps,
                    tone=tone_up,
                    model_name="gpt-4o-mini",
                )
                st.caption("Email powered by LangChain + OpenAI")
            else:
                email_out = email_stub(
                    summary=summary_text,
                    sentiment=out.get("sentiment", "Neutral"),
                    next_steps=steps,
                    tone=tone_up,
                )
                st.caption("Email (stub)")
        except RuntimeError:
            st.warning("LLM unavailable for email; using stub.")
            email_out = email_stub(
                summary=summary_text,
                sentiment=out.get("sentiment", "Neutral"),
                next_steps=steps,
                tone=tone_up,
            )

        st.markdown("**Subject**")
        st.code(email_out.get("subject") or "(empty)")
        st.markdown("**Body**")
        st.code(email_out.get("body_markdown") or "(empty)")

        st.download_button("⬇️ Email (.md)", email_out.get("body_markdown", ""), "upload_email.md")
        st.download_button("⬇️ Email (.json)", json.dumps(email_out, indent=2), "upload_email.json")

        # -------- PDF for upload
        pdf_bytes = create_summary_pdf(
            title="Upload — Meeting Report",
            meta={"source": "upload"},
            summary=summary_text,
            sentiment=out.get("sentiment", "Neutral"),
            key_phrases=kp,
            next_steps=steps,
            email_subject=email_out.get("subject"),
            email_body_md=email_out.get("body_markdown"),
        )
        st.download_button(
            "⬇️ Download Report (PDF)",
            data=pdf_bytes,
            file_name="upload_report.pdf",
            mime="application/pdf",
        )


# ============================================================
# TAB 3 — KNOWLEDGE SEARCH
# ============================================================
with tab_search:
    st.subheader("Knowledge Search (FAISS Memory)")
    st.write("Search across indexed transcripts. Use the controls below to build or reset the index.")

    from app.memory import search_similar, qa_stub_answer, reset_index, index_all

    # ---- Index controls
    colx1, colx2, colx3 = st.columns([1, 1, 2], gap="small")
    with colx1:
        if st.button("Index All Transcripts", key="btn_index_all"):
            pkgs_all = list(list_transcript_packages())
            if not pkgs_all:
                st.info("No transcripts found in data/transcripts.")
            else:
                with st.spinner("Indexing..."):
                    added = index_all(pkgs_all)
                st.success(f"Indexed {added} chunks from {len(pkgs_all)} transcript(s).")

    with colx2:
        if st.button("Reset Index", key="btn_reset_index"):
            reset_index()
            st.success("Index reset. Re-index to search again.")

    # ---- Quick topics
    st.caption("Quick topics:")
    qcols = st.columns(4)
    topics = ["refunds", "pricing", "delivery", "warranty"]
    for i, t in enumerate(topics):
        if qcols[i].button(t.capitalize(), key=f"topic_{t}"):
            st.session_state["qa_question"] = f"What do the transcripts say about {t}?"
            st.rerun()

    # ---- Search UI
    colq1, colq2 = st.columns([3, 1], gap="small")
    with colq1:
        question = st.text_input(
            "Ask a question:",
            key="qa_question",
            placeholder="e.g., What did the client say about pricing?",
        )
    with colq2:
        topk = st.number_input("Top K", min_value=1, max_value=10, value=5, step=1, key="qa_topk")

    use_llm_qa = st.toggle("Use LLM to synthesize answer", value=False, key="qa_use_llm")

    if st.button("Search", key="qa_search_btn"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            hits = search_similar(question, k=int(topk))
            if not hits:
                st.info("No matches or index missing.")
            else:
                st.subheader("Top Matches")
                for i, h in enumerate(hits, 1):
                    meta = h["metadata"]
                    with st.expander(f"[{i}] {meta.get('stem','?')} • {meta.get('chunk_id','?')} • score={h.get('score',0):.4f}"):
                        st.write(h["text"])

                # --- Compare by meeting
                from collections import Counter
                by_stem = Counter(h["metadata"].get("stem", "?") for h in hits)
                if by_stem:
                    st.markdown("### Meetings mentioning this topic")
                    for stem, cnt in by_stem.most_common():
                        st.markdown(f"- **{stem}** — {cnt} match(es)")

                # --- Answer synthesis
                st.subheader("Answer")
                try:
                    if use_llm_qa:
                        qa = generate_qa_with_llm(question, hits, model_name="gpt-4o-mini")
                        st.caption("Answer powered by LangChain + OpenAI")
                    else:
                        qa = qa_stub_answer(question, hits)
                        st.caption("Stub (deterministic)")
                except Exception:
                    st.warning("LLM unavailable. Fallback stub.")
                    qa = qa_stub_answer(question, hits)
                    st.caption("Stub (fallback)")

                st.markdown("**Answer**")
                st.code(qa.get("answer") or "(empty)", language="markdown")

                if qa.get("citations"):
                    st.markdown("**Citations**")
                    for c in qa["citations"]:
                        st.markdown(f"- **{c.get('stem','?')} / {c.get('chunk_id','?')}** — “{c.get('quote','')}”")
