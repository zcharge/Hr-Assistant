import html
import json
import os
import shutil
import urllib.request
import secrets
import sqlite3
import hashlib
import hmac
import re
import time
from contextlib import closing
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder

PDF_PATH = "Policy.pdf"
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"

# ===== ROUTING / RETRIEVAL CONSTANTS =====
HR_EMAIL = "hr@bennett.in"
RETRIEVAL_K = 4
# Minimum cosine relevance (0-1) a chunk needs before it is used. TUNE THIS using the
# "Last retrieval scores" line in the Admin panel: ask a few on-topic and off-topic
# questions and pick a value between the two groups.
RELEVANCE_THRESHOLD = 0.25
HISTORY_TURNS = 3  # how many previous answered turns the LLM may see

ESCALATION_KEYWORDS = (
    "harass", "harras", "harrass",  # includes common misspellings
    "abuse", "assault", "molest", "stalk", "threat", "bully", "bullied",
    "discriminat", "unsafe", "sexual", "intimidate", "violence",
)

ESCALATION_REPLY = (
    "I'm sorry you're dealing with this. This is a serious matter that needs a person, "
    "not a chatbot, so please contact HR directly. HR can explain your options and how "
    "to make a formal complaint.\n\n"
    "If you feel unsafe right now, call 112 or contact campus security.\n\n"
    f"HR Department: [{HR_EMAIL}](mailto:{HR_EMAIL})"
)
NO_ANSWER_REPLY = (
    "I couldn't find anything about that in the company policy. "
    f"Please contact HR for help: [{HR_EMAIL}](mailto:{HR_EMAIL})"
)
UNAVAILABLE_REPLY = "Policy database unavailable. Ensure `Policy.pdf` and Ollama are configured."
ERROR_REPLY = "I'm having trouble querying company policies right now."

FOLLOWUP_HINTS = ("it", "that", "this", "those", "they", "them", "also", "what about", "and for", "how about")

# ===== AUTH / DB CONSTANTS =====
DB_DIR = Path("./employee_dbs")
AUTH_DB = Path("./auth.db")
ROLE_LABELS = {"employee": "Employee", "manager": "Manager", "hr_admin": "HR Admin"}
DB_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="HR Policy Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_file(filename: str) -> str:
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# AUTHENTICATION & DATABASE

def _hash_password(password: str, salt: bytes = None):
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
    return salt.hex(), digest.hex()

def init_auth_db():
    with closing(sqlite3.connect(AUTH_DB)) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS employees (
                employee_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('employee','manager','hr_admin')),
                manager_id TEXT,
                salt TEXT NOT NULL,
                pw_hash TEXT NOT NULL,
                failed_attempts INTEGER DEFAULT 0,
                locked_until REAL DEFAULT 0
            );
            """
        )
        conn.commit()

def create_employee(employee_id, name, password, role="employee", manager_id=None):
    salt, pw_hash = _hash_password(password)
    with closing(sqlite3.connect(AUTH_DB)) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO employees (employee_id, name, role, manager_id, salt, pw_hash) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (employee_id, name, role, manager_id, salt, pw_hash),
        )
        conn.commit()

def authenticate(employee_id: str, password: str):
    with closing(sqlite3.connect(AUTH_DB)) as conn:
        row = conn.execute(
            "SELECT employee_id, name, role, manager_id, salt, pw_hash FROM employees WHERE employee_id = ?",
            (employee_id,),
        ).fetchone()
        if not row:
            return None, "Invalid Employee ID or Password."
        eid, name, role, manager_id, salt, pw_hash = row
        _, candidate = _hash_password(password, bytes.fromhex(salt))
        if not hmac.compare_digest(candidate, pw_hash):
            return None, "Invalid Employee ID or Password."
    return {"employee_id": eid, "name": name, "role": role, "manager_id": manager_id}, None


# RAG PIPELINE

# Cosine distance gives clean 0-1 relevance scores, which the threshold below relies on.
# NOTE: an index built earlier with the default metric must be rebuilt (Admin > Rebuild).
CHROMA_COLLECTION_META = {"hnsw:space": "cosine"}


@st.cache_resource(show_spinner=False)
def setup_rag_pipeline():
    """Returns {"vectorstore": ..., "qa_chain": ...} or None."""
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
            vectorstore = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=embeddings,
                collection_metadata=CHROMA_COLLECTION_META,
            )
        else:
            if not os.path.exists(PDF_PATH):
                return None
            loader = PyPDFLoader(PDF_PATH)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
            splits = text_splitter.split_documents(docs)
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=CHROMA_DIR,
                collection_metadata=CHROMA_COLLECTION_META,
            )

        llm = ChatOllama(model=LLM_MODEL, temperature=0.3)

        system_prompt = (
            "You are a helpful HR assistant for GlobalCorp.\n"
            "Answer the CURRENT question using ONLY the policy context below.\n"
            "If the context does not answer the question, say you could not find it in the policy "
            f"and refer the user to HR ({HR_EMAIL}).\n"
            "Never repeat or restate answers to earlier questions. Earlier messages are only for "
            "resolving references like 'it' or 'that'.\n\n"
            "Policy context:\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        # Retrieval is done manually (see generate_reply) so we can filter by relevance
        # and route escalations / off-topic questions before any context reaches the LLM.
        qa_chain = create_stuff_documents_chain(llm, prompt)
        return {"vectorstore": vectorstore, "qa_chain": qa_chain}
    except Exception:
        return None

rag = setup_rag_pipeline()
rag_chain = rag  # truthy when the knowledge base is ready (used by the admin status line)


# ROUTING + ANSWER GENERATION

def is_escalation(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in ESCALATION_KEYWORDS)


def _is_followup(text: str) -> bool:
    t = text.lower().strip()
    words = re.findall(r"[a-z']+", t)
    if len(words) <= 4:
        return True
    return any(re.search(rf"\b{re.escape(h)}\b", t) for h in FOLLOWUP_HINTS) and len(words) <= 10


def _answered_turns():
    """Previous (user, assistant) pairs that were real policy answers (not escalations / no-answer)."""
    msgs = st.session_state.messages
    turns = []
    for i in range(len(msgs) - 1):
        u, a = msgs[i], msgs[i + 1]
        if u["role"] == "user" and a["role"] == "assistant" and a.get("kind") == "answer":
            turns.append((u["content"], a.get("llm", a["content"])))
    return turns


def build_retrieval_query(prompt: str) -> str:
    """Use the raw question; only borrow the previous question when this clearly is a follow-up."""
    turns = _answered_turns()
    if turns and _is_followup(prompt):
        return f"{turns[-1][0]} {prompt}"
    return prompt


def build_chat_history():
    history = []
    for u, a in _answered_turns()[-HISTORY_TURNS:]:
        history.append(("human", u))
        history.append(("ai", a))
    return history


def _sources_line(docs) -> str:
    pages = sorted({d.metadata["page"] + 1 for d in docs if isinstance(d.metadata.get("page"), int)})
    return f"\n\nSources: PDF pages {', '.join(str(p) for p in pages)}" if pages else ""


def generate_reply(prompt: str):
    """Decide the route from the CURRENT message. Returns (display_text, llm_text, kind).

    kind: "escalation" | "no_answer" | "unavailable" | "error" | "answer"
    Only "answer" turns are ever fed back to the LLM as chat history.
    """
    # 1) Escalation: no retrieval, no sources, no policy text.
    if is_escalation(prompt):
        return ESCALATION_REPLY, ESCALATION_REPLY, "escalation"

    if not rag:
        return UNAVAILABLE_REPLY, UNAVAILABLE_REPLY, "unavailable"

    try:
        # 2) Retrieve, then keep only chunks that are actually relevant.
        query = build_retrieval_query(prompt)
        scored = rag["vectorstore"].similarity_search_with_relevance_scores(query, k=RETRIEVAL_K)
        st.session_state.last_scores = [round(float(s), 2) for _, s in scored]
        docs = [d for d, s in scored if s >= RELEVANCE_THRESHOLD]

        # 3) Off-topic: nothing relevant -> contact HR, skip the LLM entirely.
        if not docs:
            return NO_ANSWER_REPLY, NO_ANSWER_REPLY, "no_answer"

        # 4) Normal policy answer from the filtered chunks only.
        result = rag["qa_chain"].invoke({
            "input": prompt,
            "chat_history": build_chat_history(),
            "context": docs,
        })
        text = result if isinstance(result, str) else result.get("answer", "")
        text = text.strip() or NO_ANSWER_REPLY
        return text + _sources_line(docs), text, "answer"
    except Exception:
        return ERROR_REPLY, ERROR_REPLY, "error"


# =====================================================================
# ADMIN: POLICY MANAGEMENT
# =====================================================================
def _is_admin() -> bool:
    user = st.session_state.get("user")
    return bool(user) and user.get("role") == "hr_admin"


def rebuild_knowledge_base():
    """Re-index Policy.pdf from scratch. Returns (ok, message). Admin only."""
    if not _is_admin():
        return False, "Only HR admins can rebuild the knowledge base."
    if not os.path.exists(PDF_PATH):
        return False, f"{PDF_PATH} was not found."
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        embeddings.embed_query("connection check")  # fail early, before touching the old index

        docs = PyPDFLoader(PDF_PATH).load()
        splits = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150).split_documents(docs)
        if not splits:
            return False, "The policy has no readable text, so nothing was indexed."

        if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
            try:
                Chroma(
                    persist_directory=CHROMA_DIR,
                    embedding_function=embeddings,
                    collection_metadata=CHROMA_COLLECTION_META,
                ).delete_collection()
            except Exception:
                pass
        Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
            collection_metadata=CHROMA_COLLECTION_META,
        )
        setup_rag_pipeline.clear()
        return True, f"Knowledge base rebuilt from {len(docs)} pages ({len(splits)} sections)."
    except Exception as e:
        return False, f"Rebuild failed: {e}"


def replace_policy(file_bytes: bytes):
    """Validate an uploaded PDF, swap it in as Policy.pdf (old copy kept), then rebuild. Admin only."""
    if not _is_admin():
        return False, "Only HR admins can change the policy."
    tmp_path = Path("Policy.upload.pdf")
    try:
        tmp_path.write_bytes(file_bytes)
        pages = PyPDFLoader(str(tmp_path)).load()
        if not any(pg.page_content.strip() for pg in pages):
            tmp_path.unlink(missing_ok=True)
            return False, "That PDF has no readable text (it may be a scanned image)."
        if os.path.exists(PDF_PATH):
            shutil.copy2(PDF_PATH, "Policy.previous.pdf")
        os.replace(tmp_path, PDF_PATH)
    except Exception as e:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False, f"Could not read the uploaded file: {e}"
    return rebuild_knowledge_base()


# SESSION STATE MANAGEMENT

if "user" not in st.session_state:
    st.session_state.user = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "history" not in st.session_state:
    st.session_state.history = []

if "admin_upload_n" not in st.session_state:
    st.session_state.admin_upload_n = 0

if "last_scores" not in st.session_state:
    st.session_state.last_scores = []

# =====================================================================
# UI THEME  (styled after login.html / code.html)
# =====================================================================
BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, .stApp, .stApp p, .stApp label, .stApp button, .stApp input,
.stApp textarea, .stApp li, .stApp div, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
    -webkit-font-smoothing: antialiased;
}
html, body { background: #ffffff !important; color-scheme: light only; }
header[data-testid="stHeader"]::before, header[data-testid="stHeader"]::after { display: none !important; }
header[data-testid="stHeader"] { border: 0 !important; box-shadow: none !important; background-image: none !important; }
.stApp { color: #1e293b; color-scheme: light only; }
#MainMenu, [data-testid="stMainMenu"] { display: none !important; }
footer, [data-testid="stDecoration"], #stDecoration { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] button, [data-testid="stToolbar"] a,
[data-testid="stAppDeployButton"] button { color: #475569 !important; font-size: 14px; font-weight: 500; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #e2e8f0; border-radius: 9999px; }
::-webkit-scrollbar-thumb:hover { background: #cbd5e1; }

@keyframes fadeInUp { 0% { opacity: 0; transform: translateY(12px); } 100% { opacity: 1; transform: translateY(0); } }
@keyframes fadeInSidebar { 0% { opacity: 0; transform: translateX(-8px); } 100% { opacity: 1; transform: translateX(0); } }
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration: 0.01ms !important; animation-iteration-count: 1 !important;
                              transition-duration: 0.01ms !important; scroll-behavior: auto !important; }
}

/* shared small text styles */
.hr-eyebrow { font-size: 11px; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; color: #64748b; line-height: 16.5px; }
.hr-eyebrow.light { color: #94a3b8; }
.hr-muted { font-size: 14px; color: #64748b; line-height: 20px; }
.hr-mark { display: inline-flex; align-items: center; justify-content: center; flex: 0 0 auto; width: 44px; height: 44px; border-radius: 10px;
           background: #2d5282; color: #ffffff; font-size: 18px; font-weight: 700; letter-spacing: -0.02em; }
.hr-mark.sm { width: 32px; height: 32px; border-radius: 8px; font-size: 15px; }
"""

LOGIN_CSS = """
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: #ffffff !important; }
.block-container, [data-testid="stMainBlockContainer"] { max-width: none !important; width: 100% !important; margin: 0 !important; padding: 0 !important; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] { gap: 0 !important; }

/* right-hand brand panel (fixed, rounded, inset like the reference) */
.st-key-login_panel, .st-key-login_panel > [data-testid="stVerticalBlock"] { height: 0; min-height: 0; overflow: visible; }
.hr-lg-panel { position: fixed; top: 16px; right: 16px; bottom: 16px; width: calc(50vw - 24px); z-index: 1; border-radius: 16px; overflow: hidden; background-color: #1d3c66;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='640' height='640' viewBox='0 0 640 640' fill='none' stroke='%23ffffff' stroke-opacity='0.08'%3E%3Ccircle cx='560' cy='80' r='120'/%3E%3Ccircle cx='560' cy='80' r='200'/%3E%3Ccircle cx='560' cy='80' r='280'/%3E%3Ccircle cx='560' cy='80' r='360'/%3E%3Ccircle cx='560' cy='80' r='440'/%3E%3C/svg%3E");
    background-repeat: no-repeat; background-position: top right; background-size: 640px; }
.hr-lg-panel-inner { box-sizing: border-box; height: 100%; width: 100%; padding: clamp(28px, 4vw, 56px); display: flex; flex-direction: column; justify-content: space-between; }
.hr-lg-chip { align-self: flex-start; padding: 6px 12px; border: 1px solid rgba(255,255,255,0.25); border-radius: 8px; font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #e2e8f0; }
.hr-lg-h { max-width: 30rem; font-size: clamp(28px, 2.6vw, 40px); line-height: 1.2; font-weight: 700; letter-spacing: -0.025em; color: #ffffff; }
.hr-lg-p { max-width: 27rem; margin-top: 16px; font-size: 16px; line-height: 1.6; color: rgba(255,255,255,0.78); }
.hr-lg-topics { display: flex; flex-wrap: wrap; gap: 8px; max-width: 30rem; margin-top: 28px; }
.hr-lg-topics span { padding: 6px 12px; border: 1px solid rgba(255,255,255,0.22); border-radius: 8px; background: rgba(255,255,255,0.06); font-size: 13px; color: #e2e8f0; }
.hr-lg-card { display: flex; align-items: center; gap: 14px; padding: 14px 16px; max-width: 30rem; border: 1px solid rgba(255,255,255,0.18); border-radius: 12px; background: rgba(255,255,255,0.1); }
.hr-lg-card-dot { flex: 0 0 auto; width: 36px; height: 36px; border-radius: 50%; background: rgba(255,255,255,0.16); position: relative; }
.hr-lg-card-dot::after { content: ""; position: absolute; inset: 12px; border-radius: 50%; background: #ffffff; }
.hr-lg-card-title { font-size: 13px; font-weight: 600; line-height: 1.4; color: #ffffff; }
.hr-lg-card-sub { margin-top: 2px; font-size: 12px; line-height: 1.5; color: rgba(255,255,255,0.72); }

/* left-hand sign-in column */
.st-key-login_left, .st-key-login_left > [data-testid="stVerticalBlock"] { display: flex; flex-direction: column; justify-content: space-between; min-height: 100vh; gap: 0 !important; }
.st-key-login_left { width: 50vw; max-width: 50vw; margin: 0; padding: 0 max(24px, calc((50vw - 28rem) / 2)); box-sizing: border-box; }
.hr-lg-brand { display: flex; align-items: center; gap: 10px; height: 88px; font-size: 16px; font-weight: 700; letter-spacing: -0.01em; color: #0f172a; }
.hr-lg-foot { display: flex; align-items: center; height: 88px; font-size: 12px; color: #94a3b8; }
.st-key-login_zone, .st-key-login_zone > [data-testid="stVerticalBlock"] { gap: 0 !important; }
.hr-lg-eyebrow { font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #64748b; }
.hr-lg-title { margin-top: 8px; font-size: 30px; line-height: 38px; font-weight: 700; letter-spacing: -0.025em; color: #0f172a; }
.hr-lg-sub { margin: 8px 0 28px; font-size: 14px; line-height: 20px; color: #64748b; }

@media (min-width: 900px) {
    [data-testid="stToolbar"] button, [data-testid="stToolbar"] a, [data-testid="stAppDeployButton"] button { color: rgba(255,255,255,0.88) !important; }
}
@media (max-width: 899px) {
    .hr-lg-panel { display: none; }
    .st-key-login_left { width: 100%; max-width: 100%; padding: 0 24px; }
}

[data-testid="stForm"] { max-width: none; border: 0 !important; padding: 0 !important; background: transparent !important; }
[data-testid="stForm"] [data-testid="stVerticalBlock"] { gap: 20px; }
[data-testid="stTextInput"] { gap: 6px !important; }
[data-testid="stWidgetLabel"] { min-height: 0; margin: 0; }
[data-testid="stWidgetLabel"] p { font-size: 14px !important; font-weight: 500 !important; line-height: 20px; color: #334155 !important; margin: 0; }

[data-testid="stTextInputRootElement"], div[data-baseweb="input"] { background: #ffffff !important; border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important; box-shadow: 0 1px 2px rgba(0,0,0,0.05); min-height: 0; transition: all .15s ease-in-out; }
div[data-baseweb="input"]:not([data-testid="stTextInputRootElement"]) { border: 0 !important; box-shadow: none !important; background: transparent !important; }
[data-testid="stTextInputRootElement"]:focus-within { border-color: #1e293b !important; box-shadow: 0 0 0 1px #1e293b !important; }
div[data-baseweb="base-input"], div[data-baseweb="base-input"] > div { background: transparent !important; border: 0 !important; }
[data-testid="stTextInputRootElement"] input { padding: 8px 12px !important; font-size: 14px !important; line-height: 20px; color: #0f172a !important;
    background: transparent !important; -webkit-text-fill-color: #0f172a; caret-color: #0f172a; }
[data-testid="stTextInputRootElement"] button { color: #64748b !important; background: transparent !important; border: 0 !important; }
[data-testid="stTextInputRootElement"] button:hover { color: #334155 !important; }
[data-testid="stTextInputRootElement"] button svg { color: #64748b !important; fill: currentColor; }

[data-testid="stFormSubmitButton"] { margin-top: 4px; width: 100% !important; }
[data-testid="stFormSubmitButton"] button { width: 100% !important; justify-content: center; padding: 10px 16px; min-height: 0; border: 1px solid transparent !important;
    border-radius: 6px; background: #355c8c !important; box-shadow: 0 1px 2px rgba(0,0,0,0.05); transition: background-color .15s; }
[data-testid="stFormSubmitButton"] button:hover { background: #2b4c74 !important; }
[data-testid="stFormSubmitButton"] button:active { background: #243f61 !important; }
[data-testid="stFormSubmitButton"] button:focus-visible { outline: 2px solid #355c8c; outline-offset: 2px; }
[data-testid="stFormSubmitButton"] button, [data-testid="stFormSubmitButton"] button * { color: #ffffff !important; font-size: 14px; font-weight: 500; line-height: 20px; }

/* larger, calmer controls for the split layout */
[data-testid="stTextInputRootElement"], div[data-baseweb="input"] { border-radius: 8px !important; }
[data-testid="stTextInputRootElement"] input { padding: 11px 14px !important; }
[data-testid="stFormSubmitButton"] { margin-top: 8px; }
[data-testid="stFormSubmitButton"] button { padding: 12px 16px; border-radius: 8px; background: #2d5282 !important; }
[data-testid="stFormSubmitButton"] button:hover { background: #23416b !important; }
[data-testid="stFormSubmitButton"] button:active { background: #1a3253 !important; }
"""

MAIN_CSS = """
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: #ffffff !important; }

/* ---------- Sidebar (always open) ---------- */
section[data-testid="stSidebar"] { background: #f9fafb !important; border-right: 1px solid #e2e8f0;
    width: 20rem !important; min-width: 20rem !important; max-width: 20rem !important;
    transform: none !important; margin-left: 0 !important; visibility: visible !important; }
section[data-testid="stSidebar"] > div:first-child { background: #f9fafb !important; width: 100% !important; }
[data-testid="stSidebarHeader"], [data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"], [data-testid="stExpandSidebarButton"], [data-testid="stSidebarResizeHandle"] { display: none !important; }
[data-testid="stSidebarContent"] { padding: 0 !important; }
[data-testid="stSidebarUserContent"] { padding: 28px 28px 96px !important; animation: fadeInSidebar .5s cubic-bezier(.16,1,.3,1) 50ms backwards; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { margin: 0 !important; }

.st-key-side_root, .st-key-side_root > [data-testid="stVerticalBlock"] { display: flex; flex-direction: column; gap: 28px !important; }
.st-key-side_root > [data-testid="stVerticalBlock"] { flex: 1 1 auto; }
.st-key-side_root > *, .st-key-side_root > [data-testid="stVerticalBlock"] > * { animation: fadeInUp .5s cubic-bezier(.16,1,.3,1) backwards; }

.hr-brand-title { font-size: 18px; line-height: 1.375; font-weight: 700; letter-spacing: -0.025em; color: #0f172a; }
.hr-brand-sub { font-size: 14px; line-height: 20px; font-weight: 500; color: #64748b; }
.hr-user-name { font-size: 14px; line-height: 1.25; font-weight: 600; color: #0f172a; }
.hr-user-role { margin-top: 2px; font-size: 12px; line-height: 16px; color: #64748b; }

.st-key-side_recent, .st-key-side_recent > [data-testid="stVerticalBlock"] { gap: 8px !important; }
.st-key-side_cats, .st-key-side_cats > [data-testid="stVerticalBlock"] { gap: 10px !important; }
.st-key-side_cats .hr-eyebrow { margin-bottom: 2px; }
.st-key-side_footer { position: fixed !important; left: 0; bottom: 0; z-index: 20; box-sizing: border-box; width: 20rem; margin: 0 !important;
    padding: 16px 28px 20px; background: #f9fafb; border-top: 1px solid rgba(226,232,240,0.8); }
.st-key-side_footer [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; align-items: center; gap: 8px !important; }
.st-key-side_footer [data-testid="stColumn"], .st-key-side_footer [data-testid="column"] { min-width: 0 !important; width: auto !important; }
.st-key-side_footer [data-testid="stColumn"]:last-child, .st-key-side_footer [data-testid="column"]:last-child { flex: 0 0 auto !important; display: flex; justify-content: flex-end; }
.st-key-side_footer [data-testid="stColumn"]:first-child, .st-key-side_footer [data-testid="column"]:first-child { flex: 1 1 auto !important; overflow: hidden; }
.hr-user-name, .hr-user-role { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* shared button reset */
.st-key-new_conv button, [class*="st-key-cat_"] button, [class*="st-key-sq_"] button, [class*="st-key-recent_"] button, .st-key-signout button {
    min-height: 0; line-height: 20px; transition: all .15s ease-out; }
.st-key-new_conv, [class*="st-key-cat_"], [class*="st-key-sq_"], [class*="st-key-recent_"] { width: 100% !important; }
.st-key-new_conv button, [class*="st-key-cat_"] button, [class*="st-key-sq_"] button, [class*="st-key-recent_"] button { width: 100%; }
.st-key-new_conv button p, [class*="st-key-cat_"] button p, [class*="st-key-sq_"] button p, [class*="st-key-recent_"] button p, .st-key-signout button p { margin: 0 !important; line-height: inherit; }

/* New conversation */
.st-key-new_conv button { padding: 10px 20px; border: 0 !important; border-radius: 8px; background: #2d5282 !important; box-shadow: 0 1px 2px rgba(0,0,0,0.05); transition-duration: .2s; }
.st-key-new_conv button:hover { background: #23416b !important; transform: translateY(-2px); box-shadow: 0 1px 3px rgba(0,0,0,0.1), 0 1px 2px rgba(0,0,0,0.06); }
.st-key-new_conv button:active { background: #1a3253 !important; transform: translateY(0) scale(0.98); }
.st-key-new_conv button, .st-key-new_conv button * { color: #ffffff !important; font-size: 14px; font-weight: 500; }

/* Category buttons, recent items, sign out */
[class*="st-key-cat_"] button, .st-key-signout button {
    padding: 8px 16px; border: 1px solid #e2e8f0 !important; border-radius: 8px; background: #ffffff !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05); justify-content: flex-start; text-align: left; }
[class*="st-key-cat_"] button:hover, .st-key-signout button:hover { background: #f8fafc !important; border-color: #cbd5e1 !important; }
[class*="st-key-cat_"] button:active, .st-key-signout button:active { background: #f1f5f9 !important; }
[class*="st-key-cat_"] button, [class*="st-key-cat_"] button * { color: #334155 !important; font-size: 14px; font-weight: 500; }
[class*="st-key-cat_"] button:hover, [class*="st-key-cat_"] button:hover * { color: #0f172a !important; }
.st-key-signout button { padding: 6px 16px; width: auto; }
.st-key-signout button, .st-key-signout button * { color: #334155 !important; font-size: 12px; font-weight: 500; }
.st-key-signout button:active { transform: scale(0.95); }

[class*="st-key-recent_"] button { padding: 6px 8px; border: 0 !important; border-radius: 6px; background: transparent !important; justify-content: flex-start; text-align: left; }
[class*="st-key-recent_"] button:hover { background: #f1f5f9 !important; }
[class*="st-key-recent_"] button, [class*="st-key-recent_"] button * { color: #475569 !important; font-size: 14px; font-weight: 400; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

[class*="st-key-cat_"] button > div, [class*="st-key-sq_"] button > div, [class*="st-key-recent_"] button > div,
[class*="st-key-cat_"] [data-testid="stMarkdownContainer"], [class*="st-key-sq_"] [data-testid="stMarkdownContainer"], [class*="st-key-recent_"] [data-testid="stMarkdownContainer"] {
    width: 100%; justify-content: flex-start !important; text-align: left !important; }
[class*="st-key-cat_"] button p, [class*="st-key-sq_"] button p, [class*="st-key-recent_"] button p { text-align: left !important; width: 100%; }
[class*="st-key-cat_"] button, [class*="st-key-sq_"] button, [class*="st-key-recent_"] button { justify-content: flex-start !important; }
.st-key-new_conv button, .st-key-new_conv button > div { justify-content: center !important; text-align: center; }

/* Admin panel */
[data-testid="stSidebar"] [data-testid="stExpander"] { border: 1px solid #e2e8f0 !important; border-radius: 8px; background: #ffffff; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
[data-testid="stSidebar"] [data-testid="stExpander"] details { border: 0 !important; background: transparent; }
[data-testid="stSidebar"] [data-testid="stExpander"] summary { padding: 8px 16px; }
[data-testid="stSidebar"] [data-testid="stExpander"] summary p, [data-testid="stSidebar"] [data-testid="stExpander"] summary span { color: #334155 !important; font-size: 14px; font-weight: 500; }
[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"], [data-testid="stSidebar"] [data-testid="stExpander"] details > div:last-child { padding: 4px 16px 16px; }
[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stVerticalBlock"] { gap: 12px; }
.hr-admin-note { font-size: 12px; line-height: 1.5; color: #64748b; }
.hr-admin-meta { font-size: 12px; line-height: 1.5; color: #475569; }
.hr-admin-meta b { color: #0f172a; font-weight: 600; }
[data-testid="stFileUploader"] label p { font-size: 12px !important; font-weight: 500; color: #334155 !important; }
[data-testid="stFileUploaderDropzone"] { background: #f9fafb !important; border: 1px dashed #cbd5e1 !important; border-radius: 8px; padding: 10px 12px; }
[data-testid="stFileUploaderDropzone"] *, [data-testid="stFileUploader"] small { color: #475569 !important; font-size: 12px; }
[data-testid="stFileUploaderDropzone"] button { background: #ffffff !important; border: 1px solid #e2e8f0 !important; border-radius: 8px; padding: 4px 12px; min-height: 0; }
[class*="st-key-admin_"] { width: 100% !important; }
[class*="st-key-admin_"] button { width: 100%; min-height: 0; padding: 8px 16px; line-height: 20px; border-radius: 8px; justify-content: center !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05); transition: all .15s ease-out; }
[class*="st-key-admin_"] button p { margin: 0 !important; font-size: 14px; font-weight: 500; }
.st-key-admin_download button, .st-key-admin_rebuild button { background: #ffffff !important; border: 1px solid #e2e8f0 !important; }
.st-key-admin_download button:hover, .st-key-admin_rebuild button:hover { background: #f8fafc !important; border-color: #cbd5e1 !important; }
.st-key-admin_download button, .st-key-admin_download button *, .st-key-admin_rebuild button, .st-key-admin_rebuild button * { color: #334155 !important; }
.st-key-admin_replace button { background: #2d5282 !important; border: 0 !important; }
.st-key-admin_replace button:hover:not(:disabled) { background: #23416b !important; }
.st-key-admin_replace button, .st-key-admin_replace button * { color: #ffffff !important; }
[class*="st-key-admin_"] button:disabled { opacity: .5; cursor: not-allowed; }

button:focus:not(:active) { border-color: inherit; }

/* ---------- Main area ---------- */
.block-container, [data-testid="stMainBlockContainer"] { max-width: none !important; padding: 0 40px 7rem !important; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] { gap: 0 !important; }

.hr-main-head { display: flex; align-items: center; justify-content: space-between; gap: 24px; margin: 0 -40px; padding: 26px 120px 26px 40px;
                background: #ffffff; border-bottom: 1px solid #e2e8f0; animation: fadeInUp .5s cubic-bezier(.16,1,.3,1) 150ms both; }
.hr-main-brand { display: flex; align-items: center; gap: 16px; }
.hr-main-title { font-size: 22px; line-height: 30px; font-weight: 700; letter-spacing: -0.025em; color: #0f172a; }
.hr-main-sub { margin-top: 2px; font-size: 14px; line-height: 20px; color: #64748b; }
.hr-main-note { max-width: 22rem; text-align: right; }
.hr-main-note-label { font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #94a3b8; }
.hr-main-note-text { margin-top: 2px; font-size: 13px; line-height: 1.5; color: #64748b; }
@media (max-width: 1023px) { .hr-main-note { display: none; } }

.st-key-body { max-width: 944px; padding-top: 40px; }
.st-key-body, .st-key-body > [data-testid="stVerticalBlock"] { gap: 0 !important; }
.hr-welcome { margin-bottom: 40px; animation: fadeInUp .5s cubic-bezier(.16,1,.3,1) 220ms both; }
.hr-welcome-title { font-size: 16px; line-height: 24px; font-weight: 600; color: #1e293b; }
.hr-welcome-text { margin-top: 8px !important; font-size: 14px; line-height: 1.625; color: #475569; }
.hr-sq-label { margin-bottom: 16px; animation: fadeInUp .5s cubic-bezier(.16,1,.3,1) 280ms both; }

.st-key-sq_wrap { max-width: 768px; }
.st-key-sq_wrap, .st-key-sq_wrap > [data-testid="stVerticalBlock"] { gap: 14px !important; }
.st-key-sq_wrap [data-testid="stHorizontalBlock"] { gap: 14px !important; }
[class*="st-key-sq_"]:not(.st-key-sq_wrap) { animation: fadeInUp .5s cubic-bezier(.16,1,.3,1) both; }
.st-key-sq_1 { animation-delay: 340ms !important; } .st-key-sq_2 { animation-delay: 400ms !important; }
.st-key-sq_3 { animation-delay: 460ms !important; } .st-key-sq_4 { animation-delay: 520ms !important; }
[class*="st-key-sq_"] button { padding: 14px; border: 1px solid #e2e8f0 !important; border-radius: 12px; background: #ffffff !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03); justify-content: flex-start; text-align: left; transition-duration: .2s; }
[class*="st-key-sq_"] button:hover { background: rgba(248,250,252,0.8) !important; border-color: #cbd5e1 !important; transform: translateY(-2px);
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -2px rgba(0,0,0,0.1); }
[class*="st-key-sq_"] button:active { background: #f1f5f9 !important; transform: translateY(0); }
[class*="st-key-sq_"] button, [class*="st-key-sq_"] button * { color: #334155 !important; font-size: 14px; font-weight: 400; }
[class*="st-key-sq_"] button:hover, [class*="st-key-sq_"] button:hover * { color: #0f172a !important; }

/* ---------- Chat messages ---------- */
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"], [data-testid="stChatMessageAvatarCustom"] { display: none !important; }
[data-testid="stChatMessage"] { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px 18px; margin-bottom: 16px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03); gap: 0; }
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { font-size: 14px; line-height: 1.625; }
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li, [data-testid="stChatMessage"] strong, [data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2, [data-testid="stChatMessage"] h3 { color: #334155; }
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
div[data-testid="stChatMessage"]:has(div[aria-label="Chat message from user"]) { background: #2d5282; border-color: #2d5282; }
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) :is(p, li, strong, span),
div[data-testid="stChatMessage"]:has(div[aria-label="Chat message from user"]) :is(p, li, strong, span) { color: #ffffff !important; }
[data-testid="stSpinner"], [data-testid="stSpinner"] * { color: #64748b !important; font-size: 14px; }

/* ---------- Chat input bar ---------- */
[data-testid="stBottom"], [data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background: #0b0f17 !important; }
[data-testid="stBottomBlockContainer"] { max-width: none !important; padding: 20px 32px !important; }
[data-testid="stChatInput"] { max-width: 64rem; margin: 0 auto; background: transparent !important; animation: fadeInUp .5s cubic-bezier(.16,1,.3,1) 450ms both; }
[data-testid="stChatInput"] > div { background: #1e2430 !important; border: 1px solid rgba(51,65,85,0.6) !important; border-radius: 12px !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.1); transition: all .2s; }
[data-testid="stChatInput"] > div:hover { border-color: #475569 !important; }
[data-testid="stChatInput"] > div:focus-within { border-color: #94a3b8 !important; box-shadow: 0 0 0 2px rgba(100,116,139,0.5) !important; }
[data-testid="stChatInput"] [data-baseweb="textarea"], [data-testid="stChatInput"] [data-baseweb="base-input"] { background: transparent !important; border: 0 !important; }
[data-testid="stChatInput"] textarea { background: transparent !important; color: #e2e8f0 !important; -webkit-text-fill-color: #e2e8f0;
    font-size: 14px !important; line-height: 20px; padding: 14px 56px 14px 20px !important; }
[data-testid="stChatInput"] textarea::placeholder { color: #94a3b8 !important; -webkit-text-fill-color: #94a3b8; opacity: 1; }
[data-testid="stChatInputSubmitButton"] { background: #334155 !important; color: #cbd5e1 !important; border-radius: 8px !important; transition: all .15s; }
[data-testid="stChatInputSubmitButton"]:hover { background: #475569 !important; color: #ffffff !important; }
[data-testid="stChatInputSubmitButton"]:disabled { background: #334155 !important; color: #94a3b8 !important; opacity: .7; }
"""


def inject_css(css: str):
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _md_escape(text: str) -> str:
    """Escape markdown characters so recent-conversation titles render as plain text."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|<>~$:&])", r"\\\1", text)


MAX_RECENT = 8


def archive_current_conversation():
    msgs = st.session_state.messages
    if msgs:
        st.session_state.history.insert(0, list(msgs))
        st.session_state.history = st.session_state.history[:MAX_RECENT]


# =====================================================================
# LOGIN VIEW
# =====================================================================
LOGIN_PANEL_HTML = (
    '<div class="hr-lg-panel"><div class="hr-lg-panel-inner">'
    '<div class="hr-lg-chip">HR Policy Assistant</div>'
    '<div><div class="hr-lg-h">Answers from your company policy, in seconds.</div>'
    '<div class="hr-lg-p">Ask about leave, benefits, reimbursement, working hours and workplace policies. '
    'Answers are drawn from your company\'s policy document.</div>'
    '<div class="hr-lg-topics"><span>Leave &amp; Attendance</span><span>Benefits</span><span>Reimbursement</span>'
    '<span>Working Hours</span><span>Workplace Policies</span></div></div>'
    '<div class="hr-lg-card"><div class="hr-lg-card-dot"></div><div>'
    '<div class="hr-lg-card-title">Company policy, benefits and workplace information</div>'
    '<div class="hr-lg-card-sub">Company data stays within your organization\'s infrastructure.</div></div></div>'
    '</div></div>'
)


def render_login_view():
    inject_css(BASE_CSS + LOGIN_CSS)

    with st.container(key="login_panel"):
        st.markdown(LOGIN_PANEL_HTML, unsafe_allow_html=True)

    with st.container(key="login_left"):
        st.markdown(
            '<div class="hr-lg-brand"><span class="hr-mark sm">G</span><span>GlobalCorp</span></div>',
            unsafe_allow_html=True,
        )

        with st.container(key="login_zone"):
            st.markdown(
                '<div><div class="hr-lg-eyebrow">HR Policy Assistant</div>'
                '<div class="hr-lg-title">Sign in to your account</div>'
                '<div class="hr-lg-sub">Use your employee ID and password to continue.</div></div>',
                unsafe_allow_html=True,
            )
            with st.form("login_form", clear_on_submit=False):
                emp_id = st.text_input("Employee ID")
                pw = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign in")

                if submitted:
                    if not emp_id.strip() or not pw.strip():
                        st.error("Please enter both Employee ID and Password.")
                    else:
                        user, error = authenticate(emp_id.strip(), pw)
                        if user:
                            st.session_state.user = user
                            st.rerun()
                        else:
                            st.error(error)

        st.markdown(
            '<div class="hr-lg-foot">Need help signing in? Contact your HR administrator.</div>',
            unsafe_allow_html=True,
        )


# =====================================================================
# MAIN DASHBOARD VIEW
# =====================================================================
SUGGESTED_QUESTIONS = [
    "How many annual leaves can I take?",
    "What is the reimbursement process?",
    "What is the appraisal process?",
    "What is the work-from-home policy?",
]

CATEGORIES = [
    "Leave & Attendance",
    "Benefits",
    "Reimbursement",
    "Working Hours",
    "Workplace Policies",
]


def render_admin_panel():
    """Sidebar admin tools. Only called for HR admins."""
    with st.expander("Admin", expanded=False):
        st.markdown(
            '<div class="hr-admin-note">The assistant answers from the policy document below. '
            'Upload a new version to replace it and rebuild the index.</div>',
            unsafe_allow_html=True,
        )

        policy_file = Path(PDF_PATH)
        if policy_file.exists():
            info = policy_file.stat()
            updated = datetime.fromtimestamp(info.st_mtime).strftime("%d %b %Y, %H:%M")
            st.markdown(
                f'<div class="hr-admin-meta"><b>{html.escape(PDF_PATH)}</b> · {info.st_size / 1024:.0f} KB<br>Updated {updated}</div>',
                unsafe_allow_html=True,
            )
            st.download_button(
                "Download current policy",
                data=policy_file.read_bytes(),
                file_name=PDF_PATH,
                mime="application/pdf",
                key="admin_download",
            )
        else:
            st.markdown(f'<div class="hr-admin-meta"><b>{html.escape(PDF_PATH)}</b> not found.</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Replace policy (PDF)",
            type=["pdf"],
            key=f"policy_upload_{st.session_state.admin_upload_n}",
        )
        if st.button("Replace policy and rebuild", key="admin_replace", disabled=uploaded is None):
            with st.spinner("Updating policy and rebuilding the knowledge base..."):
                ok, msg = replace_policy(uploaded.getvalue())
            if ok:
                st.session_state.admin_upload_n += 1
                st.session_state.admin_notice = msg
                st.rerun()
            st.error(msg)

        st.markdown(
            '<div class="hr-admin-note">If Policy.pdf was changed outside the app, rebuild the index so answers reflect the update.</div>',
            unsafe_allow_html=True,
        )
        if st.button("Rebuild knowledge base", key="admin_rebuild"):
            with st.spinner("Rebuilding the knowledge base..."):
                ok, msg = rebuild_knowledge_base()
            if ok:
                st.session_state.admin_notice = msg
                st.rerun()
            st.error(msg)

        status = "Ready" if rag_chain else "Unavailable. Check Policy.pdf and that Ollama is running."
        st.markdown(f'<div class="hr-admin-meta">Knowledge base: {status}</div>', unsafe_allow_html=True)

        scores = st.session_state.get("last_scores") or []
        scores_txt = ", ".join(f"{s:.2f}" for s in scores) if scores else "none yet"
        st.markdown(
            f'<div class="hr-admin-meta">Relevance threshold: <b>{RELEVANCE_THRESHOLD:.2f}</b><br>'
            f'Last retrieval scores: {scores_txt}</div>',
            unsafe_allow_html=True,
        )


def render_main_view():
    user = st.session_state.user
    inject_css(BASE_CSS + MAIN_CSS)

    notice = st.session_state.pop("admin_notice", None)
    if notice:
        st.toast(notice)

    # --- SIDEBAR ---
    with st.sidebar:
        with st.container(key="side_root"):
            st.markdown(
                '<div><div class="hr-brand-title">HR Assistant</div><div class="hr-brand-sub">GlobalCorp</div></div>',
                unsafe_allow_html=True,
            )

            if st.button("New conversation", key="new_conv"):
                archive_current_conversation()
                st.session_state.messages = []
                st.rerun()

            with st.container(key="side_recent"):
                empty_note = "" if st.session_state.history else '<div class="hr-muted">No earlier conversations yet.</div>'
                st.markdown(
                    f'<div><div class="hr-eyebrow">Recent Conversations</div>{empty_note}</div>',
                    unsafe_allow_html=True,
                )
                for i, convo in enumerate(st.session_state.history):
                    first_user = next((m["content"] for m in convo if m["role"] == "user"), "Conversation")
                    title = first_user.strip().replace("\n", " ")
                    title = (title[:34] + "…") if len(title) > 34 else title
                    if st.button(_md_escape(title), key=f"recent_{i}"):
                        current = st.session_state.messages
                        chosen = st.session_state.history.pop(i)
                        if current:
                            st.session_state.history.insert(0, list(current))
                        st.session_state.messages = chosen
                        st.rerun()

            with st.container(key="side_cats"):
                st.markdown('<div class="hr-eyebrow">FAQs</div>', unsafe_allow_html=True)
                for i, cat in enumerate(CATEGORIES):
                    if st.button(cat, key=f"cat_{i}"):
                        st.session_state.pending_prompt = f"Tell me about {cat.lower()} policies."
                        st.rerun()

            if user["role"] == "hr_admin":
                render_admin_panel()

            with st.container(key="side_footer"):
                info_col, btn_col = st.columns([5, 3], vertical_alignment="center")
                with info_col:
                    st.markdown(
                        f'<div><div class="hr-user-name">{html.escape(user["name"])}</div>'
                        f'<div class="hr-user-role">{html.escape(user["employee_id"])} · '
                        f'{html.escape(ROLE_LABELS.get(user["role"], user["role"]))}</div></div>',
                        unsafe_allow_html=True,
                    )
                with btn_col:
                    if st.button("Sign out", key="signout"):
                        st.session_state.user = None
                        st.session_state.messages = []
                        st.session_state.history = []
                        st.session_state.pending_prompt = None
                        st.session_state.last_scores = []
                        st.session_state.pop("admin_notice", None)
                        st.rerun()

    # --- HEADER ---
    st.markdown(
        '<div class="hr-main-head"><div class="hr-main-brand"><span class="hr-mark">G</span><div>'
        '<div class="hr-main-title">HR Policy Assistant</div>'
        '<div class="hr-main-sub">Company policy, benefits and workplace information</div></div></div>'
        '<div class="hr-main-note"><div class="hr-main-note-label">Data privacy</div>'
        '<div class="hr-main-note-text">Company data stays within your organization\'s infrastructure.</div></div></div>',
        unsafe_allow_html=True,
    )

    # --- CHAT INPUT & TRIGGER (resolved first so the welcome view can step aside) ---
    prompt = st.chat_input("Ask about company policies, leave, benefits, reimbursement, or working hours...")

    if not prompt and st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    # --- BODY ---
    with st.container(key="body"):
        # Welcome + suggested questions (only on an empty conversation)
        if not st.session_state.messages and not prompt:
            st.markdown(
                '<div class="hr-welcome"><div class="hr-welcome-title">How can we help?</div>'
                '<p class="hr-welcome-text">Ask about company policies, leave, benefits, reimbursement, or working hours.</p></div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="hr-eyebrow light hr-sq-label">Suggested Questions</div>', unsafe_allow_html=True)

            with st.container(key="sq_wrap"):
                for row in range(2):
                    cols = st.columns(2)
                    for c in range(2):
                        idx = row * 2 + c
                        with cols[c]:
                            q = SUGGESTED_QUESTIONS[idx]
                            if st.button(q, key=f"sq_{idx + 1}"):
                                st.session_state.pending_prompt = q
                                st.rerun()

        # Chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # New turn
        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Searching policies..."):
                    # NOTE: history is read inside generate_reply BEFORE this turn is appended
                    display_text, llm_text, kind = generate_reply(prompt)
                st.markdown(display_text)

            st.session_state.messages.append({"role": "user", "content": prompt, "kind": kind})
            st.session_state.messages.append(
                {"role": "assistant", "content": display_text, "llm": llm_text, "kind": kind}
            )


# =====================================================================
# APPLICATION ENTRYPOINT
# =====================================================================
if not st.session_state.user:
    render_login_view()
else:
    render_main_view()