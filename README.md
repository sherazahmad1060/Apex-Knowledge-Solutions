# 🏢 Apex Global - RAG-Based Company Knowledge Assistant

A professional, enterprise-grade **Retrieval-Augmented Generation (RAG)** application built with **Python**, **Streamlit**, **LangChain**, **OpenAI API**, and **FAISS**.

The system enables company employees and stakeholders to query company information, internal policies, operational guidelines, and multi-sheet product catalogs stored in local `.txt` and `.xlsx` files with strictly grounded, verified source citations.

---

## 🌟 Key Features

- **Strict Grounding & Anti-Hallucination**:
  - Adheres to a strict corporate system prompt that instructs the AI to rely exclusively on retrieved documents.
  - Automatically returns the exact fallback `I don't know.` whenever the context does not provide sufficient, reliable evidence.
- **Multi-Format Document Ingestion**:
  - **TXT Processing**: Robust UTF-8, UTF-8-BOM (`utf-8-sig`), and fallback encoding handlers.
  - **Excel Workbook Processing**: Reads multiple worksheets (`.xlsx`) via `pandas` and `openpyxl`, preserves column headers, converts rows into structured key-value pairs, and records sheet name and row number metadata.
  - **Upload Validation**: File extension validation (`.txt`, `.xlsx`), file size caps (10MB default), safe filename sanitization, and SHA-256 content hashing to detect duplicates and modifications.
- **Vector Storage & Similarity Retrieval**:
  - High-performance vector similarity search using **FAISS CPU**.
  - Powered by OpenAI's `text-embedding-3-small` (or configured embedding model).
  - Atomic index building and state manifest (`manifest.json`) tracking indexed document hashes and chunk counts.
  - Configurable Top-K retrieval count, chunk size, and chunk overlap.
- **Executive UI/UX**:
  - Modern corporate design with custom CSS, dark-gradient banners, status badges, and responsive layouts.
  - Session state chat history with user/assistant avatars.
  - Interactive source citation cards beneath answers detailing source document, worksheet, row number, and text excerpts.
  - One-click suggested question chips.
  - Real-time document management (upload, process/re-index, delete with confirmation, and clear conversation).

---

## 🏗️ Project Architecture & Structure

```
company-chatbot/
├── app.py                      # Streamlit web application & user interface
├── rag.py                      # RAG retrieval pipeline & FAISS vector store manager
├── document_processor.py       # TXT & XLSX document parsing, validation, chunking
├── config.py                   # Centralized configuration, constants & key validation
├── requirements.txt            # Pinned, compatible Python dependencies
├── .env.example                # Configuration template for OpenAI API credentials
├── .gitignore                  # Git ignore rules for secrets, cache, and vector store
├── README.md                   # Comprehensive project documentation
├── data/                       # Company documents directory
│   ├── company_info.txt        # Realistic company overview & policies sample data
│   ├── products.xlsx           # Multi-sheet product catalog sample data
│   └── uploads/                # Local directory for user-uploaded documents
├── vector_store/               # Local FAISS index & manifest (generated)
├── tests/                      # Automated test suite
│   ├── test_document_processor.py # Document loading, chunking & validation tests
│   └── test_rag.py             # RAG indexing, retrieval & fallback tests
├── scripts/                    # Helper generation scripts
│   ├── generate_logo.py        # Generates corporate logo asset
│   └── generate_sample_data.py # Generates multi-sheet products Excel file
└── assets/
    └── logo.png                # Corporate brand logo
```

---

## 🛠️ Technology Stack

- **Python**: 3.10+ (tested on Python 3.14 on Windows)
- **Frontend / Web UI**: Streamlit
- **RAG Orchestration**: LangChain Core & LangChain Community
- **LLM & Embeddings**: LangChain OpenAI (`ChatOpenAI`, `OpenAIEmbeddings`)
- **Vector Database**: FAISS (CPU)
- **Data Manipulation**: Pandas, OpenPyXL
- **Environment Management**: Python-Dotenv
- **Testing**: Pytest

---

## 🚀 Getting Started

Follow these step-by-step instructions to set up and run the application locally on Windows using PowerShell.

### 1. Prerequisites
Ensure you have Python 3.10 or newer installed:
```powershell
py --version
# or
python --version
```

### 2. Create and Activate Virtual Environment
Open PowerShell in the project directory:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> **PowerShell Execution Policy Note**: If you encounter an error like `cannot be loaded because running scripts is disabled on this system`, run PowerShell as Administrator and execute:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 3. Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### 4. Configure Your OpenAI API Key
Copy the example environment file and add your OpenAI API key:
```powershell
Copy-Item .env.example .env
```
Edit `.env` using your favorite text editor:
```ini
OPENAI_API_KEY=sk-your-actual-openai-api-key
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```
*(Alternatively, you can launch the app and input your API key directly into the secure sidebar input field!)*

### 5. Launch the Application
```powershell
python -m streamlit run app.py
```
Streamlit will launch and automatically open `http://localhost:8501` in your browser.

---

## 💡 How to Use the Chatbot

1. **Verify API Status**: Check the green `● Connected` indicator in the sidebar. If not connected, enter your OpenAI API key.
2. **Process Knowledge Base**: Click the **⚡ Process / Re-index Documents** button in the sidebar. The system chunks all documents in `data/` and builds the FAISS vector index.
3. **Ask Questions**:
   - Use one-click suggested questions:
     - *"What products does our company offer?"*
     - *"What is our company remote work policy?"*
     - *"What are the prices of our products?"*
     - *"Summarize the information in our company documents."*
   - Or type any custom question into the chat box.
4. **Inspect Source Citations**: Click on the **📑 Verified Sources** expander below the assistant's reply to see exact filenames, sheets, row numbers, and document excerpts.
5. **Manage Documents**:
   - Upload new `.txt` or `.xlsx` files via the drag-and-drop uploader.
   - Delete specific documents with confirmation.
   - Clear conversation history anytime with **🧹 Clear Conversation**.

---

## 🔒 Security & Privacy

- **No Public API Keys**: API keys are loaded via `.env` or session memory. Keys are masked in the UI (`sk-...xxxx`) and never logged.
- **Git Security**: `.env`, `vector_store/`, and `.venv/` are strictly ignored by `.gitignore`.
- **Untrusted Context Isolation**: Document texts are injected into the prompt as raw context data and cannot override the system assistant prompt.
- **Offline / Cloud Transparency**: Document embeddings and chat responses call OpenAI's secure endpoints. No unauthorized third-party telemetry is dispatched.

---
