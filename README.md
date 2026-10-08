# 📘 Corporate HR Policy Assistant

A privacy-focused, AI-powered HR assistant built with **Streamlit**, **LangChain** and **Ollama**. Employees ask questions about company policy (leave, reimbursement, working hours, benefits, workplace rules) and get answers grounded strictly in the official company handbook, using Retrieval-Augmented Generation (RAG).

Because the language model and the embeddings both run locally through Ollama, **employee questions and policy text are never sent to a cloud AI service.**

---

## 📸 Screenshots

**Sign-in page**

![Sign-in page]

**Main page (chat view with suggested questions and policy categories)**

![Main page]

**Admin panel (HR Admin only), open in the sidebar**

![Admin panel]

---

## ✨ Features

**AI & search**
- **100% local AI** – `llama3.2:3b` for answers and `nomic-embed-text` for embeddings, both served by Ollama on your own machine.
- **Grounded answers** – the model is instructed to answer *only* from the retrieved policy text.
- **Smart document search** – `Policy.pdf` is split into chunks, embedded and stored in a local ChromaDB index (`chroma_db/`). The top 3 matching passages are retrieved for each question.
- **Conversation memory** – earlier turns of the current chat are passed to the model so follow-up questions work.

**Accounts & roles**
- **Secure sign-in** – passwords are never stored in plain text; they are salted and hashed with PBKDF2-SHA256 (200,000 iterations) and compared in constant time.
- **Three roles** – `employee`, `manager` and `hr_admin`, stored in a local SQLite database (`auth.db`). The signed-in user's name, ID and role are shown in the sidebar.
- **Admin-only policy management** – only `hr_admin` users see the Admin panel (see below). The server-side functions re-check the role, so hiding the panel is not the only protection.

**Admin panel (HR Admin only)**
- **Download** the current `Policy.pdf`.
- **Replace** the policy by uploading a new PDF. The upload is validated (it must contain readable text), the previous file is kept as `Policy.previous.pdf`, and the index is rebuilt automatically.
- **Rebuild knowledge base** – re-index `Policy.pdf` after editing it outside the app.
- A status line shows whether the knowledge base is ready.

**User interface**
- Split-screen sign-in page with a branded panel, plus a light-only theme (no dark mode switch).
- Sidebar with **New conversation**, **Recent conversations** (kept for the current browser session only), one-click **Policy categories** and a profile footer with **Sign out**.
- Welcome screen with **suggested questions**, a floating chat bar, and friendly error messages when the policy database or Ollama is unavailable.

---

## 🛠️ Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.10+** | Recent LangChain packages need 3.10 or newer |
| **[Ollama](https://ollama.com/download)** | Must be installed and running in the background |
| **Streamlit 1.39+** | Needed for the keyed containers the UI styling relies on |
| ~4 GB free disk space | For the two Ollama models |

---

## 🚀 Installation & Setup

### 1. Get the project
Download or clone the project and open a terminal in its folder.

### 2. (Recommended) Create a virtual environment
```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Install the Python dependencies
```bash
pip install "streamlit>=1.39" langchain-core langchain-community langchain-text-splitters langchain-ollama langchain-classic chromadb pypdf
```

### 4. Download the local AI models
```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### 5. Add your policy document
Put your company's HR policy PDF in the project folder and name it exactly **`Policy.pdf`**. The PDF must contain selectable text (scanned images are not supported).

### 6. Keep the light theme and hide the developer toolbar
Make sure this file exists next to `app4.py` (it is included in the project):

`.streamlit/config.toml`
```toml
[theme]
base = "light"
primaryColor = "#2d5282"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f9fafb"
textColor = "#1e293b"
font = "sans serif"

[client]
toolbarMode = "minimal"
```

---

## 🔐 User accounts

`app4.py` creates three demo accounts automatically every time it starts, so you can sign in straight away:

| Role | Employee ID | Password |
|---|---|---|
| HR Admin | `E001` | `ChangeMe!1` |
| Manager | `E002` | `ChangeMe!2` |
| Employee | `E003` | `ChangeMe!3` |

> ⚠️ **Change this before real use.** These three `create_employee(...)` lines (in the *AUTHENTICATION & DATABASE* section, just under `init_auth_db()`) use `INSERT OR REPLACE`, so they **reset those passwords on every run**. For a real deployment, replace the demo passwords with your own, then remove or comment out the lines once your real accounts exist.

To add another account, call the helper once (for example from a Python shell, or temporarily in the script):

```python
create_employee("E004", "Dana Employee", "a-strong-password", role="employee", manager_id="E002")
```

Valid roles: `employee`, `manager`, `hr_admin`.

---

## 💻 Running the app

Make sure Ollama is running, then start the app:

```bash
streamlit run app4.py
```

It opens at **http://localhost:8501**. On the first run the app reads `Policy.pdf` and builds the search index in `chroma_db/`; this takes a little while, and later starts reuse it.

---

## 📖 How to use it

**As an employee or manager**
1. Sign in with your Employee ID and password.
2. Pick a **suggested question** or a **Policy category** in the sidebar, or type your own question in the chat bar.
3. Ask follow-up questions in the same chat; the assistant remembers the conversation.
4. Click **New conversation** to start fresh. Previous chats appear under **Recent conversations** and can be reopened until you sign out or close the session.
5. Click **Sign out** (bottom of the sidebar) when you are done.

**As an HR Admin** (extra step)
1. Open **Admin** in the sidebar.
2. To change the policy: download the current PDF if you need it, edit it in your own tool, upload the new version under **Replace policy (PDF)** and press **Replace policy and rebuild**. Answers use the new policy immediately.
3. If you changed `Policy.pdf` directly on disk, press **Rebuild knowledge base** instead.

---

## 📂 Project structure

```
.
├── app4.py                  # UI, authentication and the RAG pipeline (single file)
├── screenshots/             # images used in this README
├── Policy.pdf               # (you provide) the source-of-truth policy document
├── Policy.previous.pdf      # (auto) backup made when an admin replaces the policy
├── chroma_db/               # (auto) saved vector index of Policy.pdf
├── auth.db                  # (auto) SQLite database of accounts and salted password hashes
├── employee_dbs/            # (auto) folder reserved for per-employee data; unused in this version
└── .streamlit/
    └── config.toml          # light theme and toolbar settings
```

**Settings you may want to change** (top of `app4.py`):

| Constant | Default | Purpose |
|---|---|---|
| `PDF_PATH` | `Policy.pdf` | Policy document location |
| `LLM_MODEL` | `llama3.2:3b` | Ollama chat model |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `CHROMA_DIR` | `./chroma_db` | Where the index is stored |

---

## 🧰 Troubleshooting

| Problem | Fix |
|---|---|
| *"Policy database unavailable…"* | Check that `Policy.pdf` is in the project folder, Ollama is running and both models are pulled, then (as admin) press **Rebuild knowledge base**. |
| Answers don't reflect an edited PDF | The index is cached. Use **Rebuild knowledge base**, or delete the `chroma_db/` folder and restart. |
| Rebuild says it can't reach the embedding model | Start Ollama and run `ollama pull nomic-embed-text`. |
| Uploaded PDF is rejected | It has no readable text (probably a scan). Use a text-based PDF or run OCR first. |
| `ModuleNotFoundError: langchain_classic` | Re-run the `pip install` command in step 3. |
| Page looks dark or unstyled | Confirm `.streamlit/config.toml` is next to `app4.py` and your Streamlit version is 1.39 or newer. |

---

## 🔒 Privacy notes

- All AI processing (answers and embeddings) happens locally through Ollama.
- The interface loads the **Inter** font from Google Fonts, which is a browser request to Google. For a fully offline setup, delete the `@import url('https://fonts.googleapis.com/...')` line at the top of `BASE_CSS` in `app4.py`; the app then uses your system font.
- Chat history lives only in the browser session and is cleared on sign out.

---

## ⚠️ Current limitations

- Employee roles are stored and used to gate the Admin panel, but per-employee data views (for example leave balances) are not part of this version.
- Answers do not yet display source page numbers or citations.
- Failed-login lockout is not enforced yet, even though the database has columns for it.
- Demo passwords are reset at every start (see **User accounts** above).
