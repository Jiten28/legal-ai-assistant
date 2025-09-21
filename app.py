# app.py
"""
Legal-AI-Assistant (single-file Streamlit prototype)
Features:
- Upload PDF/DOCX/TXT
- Extract text
- Generate plain-language summary (OpenAI)
- Simple risk-flagging (keyword+sentence extraction)
- Interactive Q&A chat grounded on the uploaded document

Run:
  pip install -r requirements.txt
  export OPENAI_API_KEY="your_api_key"   # or enter in the sidebar
  streamlit run app.py

Requirements (example):
  streamlit
  openai
  pdfplumber
  python-docx
  tiktoken (optional)
"""

import os
import re
import textwrap
from typing import List, Tuple

import streamlit as st

# PDF and DOCX parsing libs
import pdfplumber
import docx

# OpenAI
import openai

# ---------- Configuration ----------
DEFAULT_MODEL = "gpt-3.5-turbo"  # change if you have access to gpt-4
MAX_SUMMARY_TOKENS = 700
# risk keywords for naive detection (extend as needed)
RISK_KEYWORDS = [
    "termination", "penalty", "late fee", "liability", "indemnify", "indemnification",
    "breach", "governing law", "jurisdiction", "waive", "exclusive", "non-compete",
    "auto-renew", "renewal", "fine", "charge", "refund", "security deposit",
    "hidden fee", "chargeback", "force majeure", "limitation of liability"
]

# ---------- Helpers ----------
def set_api_key(api_key: str):
    """Set openai api key for current session"""
    openai.api_key = api_key

def extract_text_from_pdf(file) -> str:
    text_pages = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            text_pages.append(t)
    return "\n\n".join(text_pages)

def extract_text_from_docx(file_path_or_buffer) -> str:
    # python-docx expects a path-like object or a file-like object. It works with byte streams.
    doc = docx.Document(file_path_or_buffer)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)

def extract_text_from_txt(uploaded) -> str:
    stringio = uploaded.read().decode("utf-8", errors="ignore")
    return stringio

def extract_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    fname = uploaded_file.name.lower()
    if fname.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif fname.endswith(".docx") or fname.endswith(".doc"):
        # streamlit's uploaded file has .read() and .seek; python-docx can take a file-like
        return extract_text_from_docx(uploaded_file)
    elif fname.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)
    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")

def chunk_text(text: str, max_chars=2500) -> List[str]:
    """Simple splitter to keep prompts within token limits: chunk by paragraphs"""
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) + 2 > max_chars:
            if current:
                chunks.append(current.strip())
            current = p
        else:
            current += "\n\n" + p
    if current:
        chunks.append(current.strip())
    return chunks

# ---------- OpenAI interactions ----------
def call_openai_chat(system_prompt: str, user_prompt: str, model=DEFAULT_MODEL, temperature=0.2, max_tokens=MAX_SUMMARY_TOKENS):
    """
    Simple wrapper for ChatCompletion
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    resp = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp["choices"][0]["message"]["content"].strip()

def summarize_text(text: str, model=DEFAULT_MODEL) -> str:
    """
    Summarize text into plain language. For long docs, chunk and summarize + combine.
    """
    # If text is short, summarize directly
    if len(text) < 3000:
        system_prompt = ("You are an assistant that converts legal documents into simple, plain English. "
                         "Produce a concise summary (5-8 bullet points or short paragraphs) capturing key parties, obligations, timelines, fees, and risks.")
        user_prompt = f"Document:\n\n{text}\n\nProduce a plain-language summary of the above legal text."
        return call_openai_chat(system_prompt, user_prompt, model=model)
    # For long text: chunk -> summarize each -> combine
    chunks = chunk_text(text, max_chars=3000)
    partial_summaries = []
    for i, c in enumerate(chunks):
        system_prompt = ("You are an assistant that converts legal language into plain English. "
                         "Summarize the following extract into short bullet points focusing on key obligations, dates, fees, liabilities and risks.")
        user_prompt = f"Extract {i+1}/{len(chunks)}:\n\n{c}\n\nProvide a short summary (3-6 bullets)."
        s = call_openai_chat(system_prompt, user_prompt, model=model, max_tokens=400)
        partial_summaries.append(s)
    # combine partial summaries
    combined = "\n\n".join(partial_summaries)
    system_prompt = ("You are an assistant that condenses multiple summaries into a final concise plain-language summary. "
                     "Combine and deduplicate the points, and present a final 6-10 bullet point plain English summary.")
    user_prompt = f"Combine the following partial summaries:\n\n{combined}"
    return call_openai_chat(system_prompt, user_prompt, model=model, max_tokens=MAX_SUMMARY_TOKENS)

def naive_risk_flagging(text: str, keywords: List[str]=RISK_KEYWORDS) -> List[Tuple[str, str]]:
    """
    Return list of (keyword, sentence) where keyword found. Very naive: splits into sentences and checks keywords.
    """
    # Simple sentence splitter
    sentences = re.split(r'(?<=[\.\?\!;])\s+', text.replace("\n", " "))
    found = []
    for s in sentences:
        s_lower = s.lower()
        for kw in keywords:
            if kw in s_lower:
                found.append((kw, s.strip()))
    # deduplicate by sentence
    dedup = []
    seen = set()
    for kw, s in found:
        if s not in seen:
            dedup.append((kw, s))
            seen.add(s)
    return dedup

def answer_question(context_text: str, question: str, model=DEFAULT_MODEL) -> str:
    """
    Answer a question grounded on context_text using the model.
    We'll provide context (may be chunked).
    """
    # Take first 4000 chars of context for safety (or chunk smarter)
    context_excerpt = context_text[:4000] + ("\n\n[truncated]" if len(context_text) > 4000 else "")
    system_prompt = ("You are a helpful legal assistant. Use the provided document excerpt to answer the user's question in plain English. "
                     "If the document does not provide enough information, say you don't have enough detail and suggest what to check.")
    user_prompt = f"Document excerpt:\n\n{context_excerpt}\n\nQuestion: {question}\n\nAnswer concisely in plain English."
    return call_openai_chat(system_prompt, user_prompt, model=model, max_tokens=400)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Legal AI Assistant", layout="wide")
st.title("🧾 Generative AI — Demystifying Legal Documents")
st.write("Upload a legal document (PDF/DOCX/TXT). Get a plain-language summary, risk highlights, and ask questions about the document.")

# Sidebar: API key and options
st.sidebar.header("Settings & API Key")
api_key_input = st.sidebar.text_input("Enter OpenAI API key (or set OPENAI_API_KEY env var)", type="password")
if api_key_input:
    set_api_key(api_key_input)
else:
    # try env var
    env_key = os.getenv("OPENAI_API_KEY", "")
    if env_key:
        set_api_key(env_key)

model_choice = st.sidebar.selectbox("Model", options=[DEFAULT_MODEL, "gpt-4"], index=0)
if model_choice:
    DEFAULT_MODEL = model_choice

st.sidebar.markdown("**Risk keyword list** (editable)")
keywords_text = st.sidebar.text_area("Comma-separated keywords", value=", ".join(RISK_KEYWORDS), height=120)
if keywords_text:
    RISK_KEYWORDS = [k.strip().lower() for k in keywords_text.split(",") if k.strip()]

# File upload
uploaded_file = st.file_uploader("Upload a legal document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Document")
    if uploaded_file:
        try:
            doc_text = extract_text(uploaded_file)
        except Exception as e:
            st.error(f"Failed to extract text: {e}")
            doc_text = ""
        if not doc_text.strip():
            st.warning("No text extracted or file empty.")
        else:
            st.success("Text extracted from document.")
            # Show collapsed preview
            with st.expander("Preview: first 2000 characters"):
                st.write(doc_text[:2000] + ("..." if len(doc_text) > 2000 else ""))
    else:
        doc_text = ""
        st.info("Upload a file to begin.")

with col2:
    st.header("Actions")
    if not openai.api_key:
        st.error("OpenAI API key not found. Enter it in the sidebar or set OPENAI_API_KEY environment variable.")
    else:
        if doc_text:
            if st.button("Generate Plain-Language Summary"):
                with st.spinner("Generating summary..."):
                    try:
                        summary = summarize_text(doc_text, model=DEFAULT_MODEL)
                        st.success("Summary generated.")
                        st.markdown("### Plain-language summary")
                        st.write(summary)
                        # store in session
                        st.session_state["summary"] = summary
                    except Exception as e:
                        st.error(f"Error generating summary: {e}")

            if st.button("Run Risk Flagging"):
                with st.spinner("Detecting risk clauses..."):
                    try:
                        flags = naive_risk_flagging(doc_text, RISK_KEYWORDS)
                        if flags:
                            st.warning(f"Found {len(flags)} potential risk clauses.")
                            for kw, sent in flags[:50]:
                                st.markdown(f"- **{kw}** — {sent}")
                        else:
                            st.success("No obvious risk keywords found (naive scan).")
                        st.session_state["flags"] = flags
                    except Exception as e:
                        st.error(f"Error in risk detection: {e}")

            st.markdown("---")
            st.markdown("### Quick actions")
            if st.button("Generate Summary + Risk Flags (All-in-one)"):
                with st.spinner("Working..."):
                    try:
                        summary = summarize_text(doc_text, model=DEFAULT_MODEL)
                        flags = naive_risk_flagging(doc_text, RISK_KEYWORDS)
                        st.markdown("### Summary")
                        st.write(summary)
                        st.markdown("### Risk flags")
                        if flags:
                            for kw, sent in flags:
                                st.markdown(f"- **{kw}** — {sent}")
                        else:
                            st.write("No obvious risk keywords found (naive scan).")
                        st.session_state["summary"] = summary
                        st.session_state["flags"] = flags
                    except Exception as e:
                        st.error(f"Error: {e}")

# Chat / Q&A
st.markdown("---")
st.header("Ask questions about the document")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

question = st.text_input("Enter your question about the uploaded document (e.g., 'Can the landlord increase rent anytime?')")

if st.button("Ask"):
    if not openai.api_key:
        st.error("OpenAI API key missing.")
    elif not doc_text:
        st.error("Upload a document first.")
    elif not question.strip():
        st.error("Enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                answer = answer_question(doc_text, question, model=DEFAULT_MODEL)
                st.session_state["chat_history"].append(("You: " + question, "Assistant: " + answer))
            except Exception as e:
                st.error(f"Error answering: {e}")

# Display chat history
if st.session_state["chat_history"]:
    st.markdown("### Chat history")
    for q, a in reversed(st.session_state["chat_history"]):
        st.markdown(f"**{q}**")
        st.write(a)

# Footer / tips
st.markdown("---")
st.markdown("**Notes & Limitations**")
st.markdown(textwrap.dedent("""
- This is a prototype for demo purposes. Risk detection here is *naive* (keyword/sentence-based). For production, integrate NER, clause classification, and legal rules.
- Always consult a qualified lawyer for legal advice. This tool aids understanding but is not a replacement for professional legal advice.
- For long documents, the Q&A uses an excerpt; consider building document retrieval (RAG) for more accurate grounding over long docs.
- You can change the model in the sidebar. If you have access to GPT-4, select it for higher-quality outputs.
"""))

st.markdown("**If you want, I can also**: generate a GitHub-ready README, produce a deploy script, or convert this to a small Flask app or WhatsApp bot. Tell me which and I will prepare it.")
