import os
import re
import textwrap
from typing import List, Tuple

import streamlit as st
import pdfplumber
import docx
import openai

# ---------- Configuration ----------
DEFAULT_MODEL = "gpt-3.5-turbo"
MAX_SUMMARY_TOKENS = 700
RISK_KEYWORDS = [
    "termination", "penalty", "late fee", "liability", "indemnify", "indemnification",
    "breach", "governing law", "jurisdiction", "waive", "exclusive", "non-compete",
    "auto-renew", "renewal", "fine", "charge", "refund", "security deposit",
    "hidden fee", "chargeback", "force majeure", "limitation of liability"
]

# ---------- Helpers ----------
def set_api_key(api_key: str):
    openai.api_key = api_key

def extract_text_from_pdf(file) -> str:
    text_pages = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            text_pages.append(t)
    return "\n\n".join(text_pages)

def extract_text_from_docx(file_path_or_buffer) -> str:
    doc = docx.Document(file_path_or_buffer)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(uploaded) -> str:
    return uploaded.read().decode("utf-8", errors="ignore")

def extract_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    fname = uploaded_file.name.lower()
    if fname.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif fname.endswith(".docx") or fname.endswith(".doc"):
        return extract_text_from_docx(uploaded_file)
    elif fname.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)
    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")

def chunk_text(text: str, max_chars=2500) -> List[str]:
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

# ---------- Demo Mode ----------
def demo_response(mode: str, question: str = "", context_text: str = "", error: str = "") -> str:
    snippet = context_text[:500].replace("\n", " ") if context_text else "Sample contract text between two parties."
    if mode == "summary":
        return (
            "🛡️ Demo mode active (API not available/ Limit Exceeded).\n\n"
            "Sample summary based on uploaded document:\n"
            f"- Document begins with: '{snippet[:120]}...'\n"
            "- This is an agreement between at least two parties.\n"
            "- Duties and responsibilities are outlined.\n"
            "- Termination/penalty clauses may apply.\n"
            f"{'[Error: ' + error + ']' if error else ''}"
        )
    elif mode == "qa":
        return (
            f"🛡️ Demo mode active (API not available/ Limit Exceeded).\n\n"
            f"**Your Question:** {question}\n\n"
            "- The document defines obligations and restrictions.\n"
            "- Some risks may apply.\n"
            "- Please review the document for specific details.\n"
            f"{'[Error: ' + error + ']' if error else ''}"
        )
    return "🛡️ Demo mode: Example output."

# ---------- OpenAI interactions ----------
def call_openai_chat(system_prompt: str, user_prompt: str, model=DEFAULT_MODEL,
                    temperature=0.2, max_tokens=MAX_SUMMARY_TOKENS, context_text=""):
    if not openai.api_key:
        return demo_response("summary", context_text=context_text)
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return demo_response("summary", context_text=context_text, error=str(e))

def summarize_text(text: str, model=DEFAULT_MODEL) -> str:
    if len(text) < 3000:
        system_prompt = "You are an assistant that converts legal documents into simple, plain English."
        user_prompt = f"Document:\n\n{text}\n\nSummarize in plain language."
        return call_openai_chat(system_prompt, user_prompt, model=model, context_text=text)
    chunks = chunk_text(text, max_chars=3000)
    combined = "\n\n".join([c[:300] for c in chunks])
    return demo_response("summary", context_text=combined)

def naive_risk_flagging(text: str, keywords: List[str]=RISK_KEYWORDS) -> List[Tuple[str, str]]:
    sentences = re.split(r'(?<=[\.\?\!;])\s+', text.replace("\n", " "))
    found = []
    for s in sentences:
        for kw in keywords:
            if kw in s.lower():
                found.append((kw, s.strip()))
    return list({s: (kw, s) for kw, s in found}.values())

def answer_question(context_text: str, question: str, model=DEFAULT_MODEL) -> str:
    if not openai.api_key:
        return demo_response("qa", question=question, context_text=context_text)
    context_excerpt = context_text[:4000] + ("\n\n[truncated]" if len(context_text) > 4000 else "")
    system_prompt = "You are a helpful legal assistant. Use the document excerpt to answer questions in plain English."
    user_prompt = f"Document excerpt:\n\n{context_excerpt}\n\nQuestion: {question}"
    try:
        return call_openai_chat(system_prompt, user_prompt, model=model, context_text=context_text)
    except Exception as e:
        return demo_response("qa", question=question, context_text=context_text, error=str(e))

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Legal AI Assistant", layout="wide")
st.title("🧾 Generative AI — Demystifying Legal Documents")
st.write("Upload a legal document (PDF/DOCX/TXT). Get a plain-language summary, risk highlights, and ask questions about the document.")

# Sidebar: API key and options
st.sidebar.header("⚙️ Settings & API Key")

env_key = os.getenv("OPENAI_API_KEY", "")
if env_key:
    set_api_key(env_key)
    st.sidebar.success("✅ API key loaded from environment")
else:
    st.sidebar.warning("⚠️ No API key found — running in Demo Mode (sample outputs only)")

model_choice = st.sidebar.selectbox("🤖 Model", options=[DEFAULT_MODEL, "gpt-4"], index=0)
if model_choice:
    DEFAULT_MODEL = model_choice

st.sidebar.markdown("**⚠️ Risk keyword list** (editable)")
keywords_text = st.sidebar.text_area("Comma-separated keywords", value=", ".join(RISK_KEYWORDS), height=120)
if keywords_text:
    RISK_KEYWORDS = [k.strip().lower() for k in keywords_text.split(",") if k.strip()]

# File upload
uploaded_file = st.file_uploader("📂 Upload a legal document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
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
            with st.expander("Preview: first 2000 characters"):
                st.write(doc_text[:2000] + ("..." if len(doc_text) > 2000 else ""))
    else:
        doc_text = ""
        st.info("Upload a file to begin.")

with col2:
    st.header("Actions")
    if doc_text:
        if st.button("📝 Generate Plain-Language Summary"):
            with st.spinner("Generating summary..."):
                summary = summarize_text(doc_text, model=DEFAULT_MODEL)
                st.success("Summary generated.")
                st.markdown("### Plain-language summary")
                st.write(summary)
                st.session_state["summary"] = summary

        if st.button("⚠️ Run Risk Flagging"):
            with st.spinner("Detecting risk clauses..."):
                flags = naive_risk_flagging(doc_text, RISK_KEYWORDS)
                if flags:
                    st.warning(f"Found {len(flags)} potential risk clauses.")
                    for kw, sent in flags[:50]:
                        st.markdown(f"- **{kw}** — {sent}")
                else:
                    st.success("No obvious risk keywords found (naive scan).")
                st.session_state["flags"] = flags

        st.markdown("---")
        st.markdown("### Quick actions")
        if st.button("📝 Generate Summary + ⚠️ Risk Flags (All-in-one)"):
            with st.spinner("Working..."):
                summary = summarize_text(doc_text, model=DEFAULT_MODEL)
                flags = naive_risk_flagging(doc_text, RISK_KEYWORDS)
                st.markdown("### Summary")
                st.write(summary)
                st.markdown("### Risk flags")
                if flags:
                    for kw, sent in flags:
                        st.markdown(f"- **{kw}** — {sent}")
                else:
                    st.write("No obvious risk keywords found.")
                st.session_state["summary"] = summary
                st.session_state["flags"] = flags

# Chat / Q&A
st.markdown("---")
st.header("❓Ask questions about the document")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

question = st.text_input("Enter your question (e.g., 'Can the landlord increase rent anytime?')")

if st.button("Ask"):
    if not doc_text:
        st.error("Upload a document first.")
    elif not question.strip():
        st.error("Enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer = answer_question(doc_text, question, model=DEFAULT_MODEL)
            st.session_state["chat_history"].append(("You: " + question, "Assistant: " + answer))

if st.session_state["chat_history"]:
    st.markdown("### Chat history")
    for q, a in reversed(st.session_state["chat_history"]):
        st.markdown(f"**{q}**")
        st.write(a)

# Footer / tips
st.markdown("---")
st.markdown("**Notes & Limitations**")
st.markdown(textwrap.dedent("""
- This is a prototype for demo purposes. Risk detection is *naive* (keyword-based).
- Always consult a qualified lawyer for real advice. This tool aids understanding only.
- For long documents, consider retrieval-based QA for more accuracy.
- Change the model in the sidebar (GPT-4 if available).
"""))
