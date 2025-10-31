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
# If you use IBM watsonx later, this is the place to load those creds.


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
