# app/export_pdf.py
from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.units import inch
from reportlab.lib import colors

def _h(text):  # small helper
    return Paragraph(text.replace("\n", "<br/>"))

def create_summary_pdf(
    title: str,
    meta: dict | None,
    summary: str,
    sentiment: str,
    key_phrases: list[str],
    next_steps: list[str],
    email_subject: str | None,
    email_body_md: str | None,
) -> bytes:
    """
    Returns PDF bytes containing a compact one-pager of the analysis + email.
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=LETTER, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54)

    styles = getSampleStyleSheet()
    H1 = styles["Heading1"]
    H2 = styles["Heading2"]
    Body = styles["BodyText"]

    # Slightly denser body
    Body.leading = 14
    H1.spaceAfter = 8
    H2.spaceBefore = 10
    H2.spaceAfter = 6

    # Light gray rule style
    rule = Paragraph('<para backColor="#EAEAEA" spaceb="6"> </para>')

    story = []

    # Header
    story.append(Paragraph("AI Sales Copilot — Meeting Report", H1))
    story.append(Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M"), Body))
    story.append(Spacer(1, 8))

    # Metadata
    if meta:
        meta_lines = []
        for k in ["file_name", "language", "model", "duration_sec", "created_at"]:
            if k in meta and meta[k] not in (None, ""):
                meta_lines.append(f"<b>{k.replace('_',' ').title()}:</b> {meta[k]}")
        if meta_lines:
            story.append(Paragraph("<br/>".join(meta_lines), Body))
            story.append(Spacer(1, 10))

    story.append(rule)
    story.append(Spacer(1, 6))

    # Summary
    story.append(Paragraph("Summary", H2))
    story.append(Paragraph((summary or "(empty)").replace("\n", "<br/>"), Body))

    # Sentiment / Key Phrases
    story.append(Spacer(1, 6))
    story.append(Paragraph("Sentiment", H2))
    story.append(Paragraph(sentiment or "Neutral", Body))

    story.append(Spacer(1, 6))
    story.append(Paragraph("Key Phrases", H2))
    if key_phrases:
        lf = ListFlowable(
            [ListItem(Paragraph(k, Body), leftIndent=12) for k in key_phrases],
            bulletType="bullet",
            start="circle",
            bulletColor=colors.black,
        )
        story.append(lf)
    else:
        story.append(Paragraph("(none)", Body))

    # Next Steps
    story.append(Spacer(1, 6))
    story.append(Paragraph("Next Steps", H2))
    if next_steps:
        lf = ListFlowable(
            [ListItem(Paragraph(s, Body), leftIndent=12) for s in next_steps],
            bulletType="bullet",
            start="bullet",
            bulletColor=colors.black,
        )
        story.append(lf)
    else:
        story.append(Paragraph("(none)", Body))

    # Email
    story.append(Spacer(1, 6))
    story.append(Paragraph("Follow-up Email", H2))
    story.append(Paragraph(f"<b>Subject:</b> {email_subject or '(empty)'}", Body))
    if email_body_md:
        # naive markdown-ish to paragraphs; keeps lines readable
        for line in email_body_md.splitlines():
            story.append(Paragraph(line if line.strip() else "&nbsp;", Body))
    else:
        story.append(Paragraph("(no body)", Body))

    doc.build(story)
    return buf.getvalue()
