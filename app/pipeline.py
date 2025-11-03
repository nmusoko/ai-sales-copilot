# app/pipeline.py
"""
LLM-backed summarization with strict JSON output.
- Uses LangChain + OpenAI (easily swappable to watsonx).
- Validates output with Pydantic and auto-repairs once if needed.
"""

from __future__ import annotations
import os
import json
from typing import List, Literal, Optional, Dict

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

# LangChain core
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable
from langchain_core.exceptions import OutputParserException

# OpenAI LLM (LangChain wrapper)
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    # fallback older import if needed
    from langchain.chat_models import ChatOpenAI  # type: ignore


# ---------------------------
# 1) Load environment
# ---------------------------
load_dotenv()  # allows OPENAI_API_KEY from .env

# ---------------------------
# 2) Define the structured output (Pydantic)
# ---------------------------
SentimentType = Literal["Positive", "Neutral", "Negative"]

class LLMCallSummary(BaseModel):
    meeting_summary: str = Field(..., description="Concise bullet-style summary of the call")
    sentiment: SentimentType = Field(..., description="Overall sentiment of the client")
    key_phrases: List[str] = Field(default_factory=list, description="Top phrases/keywords")
    next_steps: List[str] = Field(default_factory=list, description="Actionable next steps captured from the call")


# Parser that will generate format instructions for the LLM
parser = PydanticOutputParser(pydantic_object=LLMCallSummary)


# ---------------------------
# 3) Build the prompt
# ---------------------------
SYSTEM = """You are a meticulous sales meeting analyst.
Return ONLY valid JSON that matches the provided schema.
Do NOT include explanations, code fences, or extra keys.
If a field is unknown, use an empty string or an empty list accordingly.
Keep summaries factual; do not invent information not present in the transcript.
"""

# The {format_instructions} placeholder will be auto-filled by parser.get_format_instructions()
USER_TMPL = """Transcript:
\"\"\"
{transcript}
\"\"\"

{format_instructions}
"""

prompt: PromptTemplate = PromptTemplate(
    template=USER_TMPL,
    input_variables=["transcript"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


# ---------------------------
# 4) Create the LLM
# ---------------------------
def make_llm(model_name: str = "gpt-4o-mini", temperature: float = 0.2):
    """
    Create an OpenAI chat model via LangChain.
    - model_name can be any available OpenAI chat model.
    - temperature low for consistency.
    """
    return ChatOpenAI(model=model_name, temperature=temperature)


# ---------------------------
# 5) Chain assembly
# ---------------------------
def make_chain(llm: Optional[ChatOpenAI] = None) -> Runnable:
    """
    Returns a Runnable pipeline: prompt -> llm -> pydantic parser
    """
    llm = llm or make_llm()
    chain = prompt | llm | parser
    return chain


# ---------------------------
# 6) Public function: summarize text with JSON+repair
# ---------------------------
def summarize_with_llm(transcript: str, model_name: str = "gpt-4o-mini") -> Dict:
    """
    Run the chain with a single repair attempt if parsing fails once.
    Returns a dict matching LLMCallSummary schema.
    """
    chain = make_chain(make_llm(model_name=model_name))

    # First attempt
    try:
        result: LLMCallSummary = chain.invoke({"transcript": transcript})
        return json.loads(result.model_dump_json())
    except OutputParserException as e:
        # Try one repair: ask the model to fix JSON only
        fix_prompt = f"""The previous output could not be parsed as the target JSON schema.
Reformat the following content into EXACTLY the required JSON schema.
Do not add explanations or code fences.

SCHEMA:
{parser.get_format_instructions()}

CONTENT TO FIX:
{str(e)}
"""
        llm = make_llm(model_name=model_name)
        fixed_text = llm.invoke(fix_prompt).content
        try:
            # try parsing fixed result
            fixed = parser.parse(fixed_text)
            return json.loads(fixed.model_dump_json())
        except Exception as e2:
            raise ValueError(f"Failed to parse LLM output after one repair attempt: {e2}") from e
    except ValidationError as ve:
        raise ValueError(f"Validation error: {ve}") from ve
# --- Email Draft schema & chain ---

class EmailDraft(BaseModel):
    subject: str = Field(..., description="Professional subject line for the follow-up")
    body_markdown: str = Field(..., description="Complete email body in Markdown with greeting, body, bullets if useful, and a courteous sign-off")

email_parser = PydanticOutputParser(pydantic_object=EmailDraft)

EMAIL_SYSTEM = """You are an expert B2B sales assistant.
Return ONLY valid JSON matching the provided schema.
Write concise, professional emails in the requested tone.
Do not invent facts. Base content on the provided context."""

EMAIL_TMPL = """Context for the email:
- Summary of call:
{summary}

- Overall sentiment: {sentiment}

- Next steps we captured: {next_steps}

- Desired tone: {tone}  (one of: friendly, formal, concise, persuasive)

Write a follow-up email to the client. The email should:
- Recap key outcomes accurately
- Acknowledge any objections/concerns gently
- Propose the next steps clearly (dates if available)
- Keep it short (120–220 words)
- Use the provided tone

{format_instructions}
"""

email_prompt = PromptTemplate(
    template=EMAIL_TMPL,
    input_variables=["summary", "sentiment", "next_steps", "tone"],
    partial_variables={"format_instructions": email_parser.get_format_instructions()},
)

def make_email_chain(llm: Optional[ChatOpenAI] = None) -> Runnable:
    llm = llm or make_llm()
    return email_prompt | llm | email_parser

def generate_email_with_llm(
    summary: str,
    sentiment: str,
    next_steps: list[str],
    tone: str = "friendly",
    model_name: str = "gpt-4o-mini",
) -> dict:
    """
    Produce an EmailDraft dict {subject, body_markdown} with one repair attempt.
    """
    chain = make_email_chain(make_llm(model_name=model_name))
    payload = {
        "summary": summary,
        "sentiment": sentiment,
        "next_steps": "\n".join(f"- {s}" for s in (next_steps or [])),
        "tone": tone,
    }

    # try normal path with small backoff like before
    import time
    delays = [0, 2]
    last_err = None
    for d in delays:
        try:
            if d: time.sleep(d)
            result: EmailDraft = chain.invoke(payload)
            return json.loads(result.model_dump_json())
        except Exception as e:
            last_err = e
            if "insufficient_quota" in str(e).lower() or "rate limit" in str(e).lower() or "429" in str(e):
                continue
            break

    # repair once
    try:
        fix_prompt = f"""Reformat the content below into EXACTLY the required JSON schema for EmailDraft.

SCHEMA:
{email_parser.get_format_instructions()}

CONTENT TO FIX:
{str(last_err)}
"""
        llm = make_llm(model_name=model_name)
        fixed_text = llm.invoke(fix_prompt).content
        fixed = email_parser.parse(fixed_text)
        return json.loads(fixed.model_dump_json())
    except Exception as e2:
        raise RuntimeError("LLM email generation failed. Use stub or fix quota.") from e2

# --- QA over retrieved chunks ---

from pydantic import BaseModel, Field
from typing import List, Dict

class QACitation(BaseModel):
    stem: str = Field(..., description="Transcript base name")
    chunk_id: str = Field(..., description="Chunk identifier like 'sample_call:0'")
    quote: str = Field(..., description="Short supporting quote (≤300 chars)")

class QAAnswer(BaseModel):
    answer: str = Field(..., description="Concise answer grounded in the provided context")
    citations: List[QACitation] = Field(default_factory=list, description="Supporting citations (stem/chunk/quote)")

qa_parser = PydanticOutputParser(pydantic_object=QAAnswer)

QA_TMPL = """You are a precise analyst. Answer the user's question ONLY using the CONTEXT below.
If the answer is not contained in the context, say "I don't see that in the transcripts."

QUESTION:
{question}

CONTEXT (top {k} chunks):
{context}

{format_instructions}
"""

qa_prompt = PromptTemplate(
    template=QA_TMPL,
    input_variables=["question", "context", "k"],
    partial_variables={"format_instructions": qa_parser.get_format_instructions()},
)

def make_qa_chain(llm: Optional[ChatOpenAI] = None) -> Runnable:
    llm = llm or make_llm()
    return qa_prompt | llm | qa_parser

def generate_qa_with_llm(question: str, hits: List[Dict], model_name: str = "gpt-4o-mini") -> dict:
    """
    hits: list of { 'text': str, 'score': float, 'metadata': {...} } from memory.search_similar
    Returns: { answer: str, citations: [{stem, chunk_id, quote}] }
    """
    if not hits:
        return {"answer": "I don't see that in the transcripts.", "citations": []}

    # Build a compact context with metadata headers
    blocks = []
    for h in hits:
        meta = h["metadata"]
        header = f"[{meta.get('stem','?')} | {meta.get('chunk_id','?')} | score={h.get('score',0):.4f}]"
        text = h["text"].replace("\n", " ")
        if len(text) > 800:
            text = text[:800] + " …"
        blocks.append(header + "\n" + text)

    payload = {
        "question": question.strip(),
        "context": "\n\n".join(blocks),
        "k": str(len(hits)),
    }

    # Light retry + one repair attempt (same pattern as before)
    import time
    delays = [0, 2]
    last_err = None
    chain = make_qa_chain(make_llm(model_name=model_name))
    for d in delays:
        try:
            if d: time.sleep(d)
            result: QAAnswer = chain.invoke(payload)
            return json.loads(result.model_dump_json())
        except Exception as e:
            last_err = e
            if "insufficient_quota" in str(e).lower() or "rate limit" in str(e).lower() or "429" in str(e):
                continue
            break

    # repair attempt
    try:
        fix_prompt = f"""Reformat the content below into EXACTLY the required JSON schema for QAAnswer.

SCHEMA:
{qa_parser.get_format_instructions()}

CONTENT TO FIX:
{str(last_err)}
"""
        llm = make_llm(model_name=model_name)
        fixed_text = llm.invoke(fix_prompt).content
        fixed = qa_parser.parse(fixed_text)
        return json.loads(fixed.model_dump_json())
    except Exception:
        # Fallback minimal answer
        return {"answer": "I don't see that in the transcripts.", "citations": []}
