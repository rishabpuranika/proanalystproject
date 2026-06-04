# Upwork API Technical Support Assistant

## Project Overview

A production-quality **AI Technical Support Bot** that answers developer questions about the **Upwork API** using **Retrieval-Augmented Generation (RAG)**. Built with LangChain, ChromaDB, and Streamlit, powered by Meta-Llama-3.1-8B via DeepInfra.

The system ingests Upwork's official API documentation (PDF), creates semantic embeddings, stores them in a persistent vector database, and uses retrieval-augmented generation to deliver accurate, source-cited answers to developer queries — all while actively preventing hallucination.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                          │
│                                                                    │
│   ┌─────────┐    ┌───────────┐    ┌────────────┐    ┌──────────┐  │
│   │  PDF    │───▶│ Chunking  │───▶│ Embeddings │───▶│ ChromaDB │  │
│   │ (Data/) │    │ (500 char │    │ (MiniLM-   │    │ (Vector  │  │
│   │         │    │  +50 over)│    │   L6-v2)   │    │   Store) │  │
│   └─────────┘    └───────────┘    └────────────┘    └──────────┘  │
│                                                          │         │
└──────────────────────────────────────────────────────────┼─────────┘
                                                           │
┌──────────────────────────────────────────────────────────┼─────────┐
│                       QUERY PIPELINE                     │         │
│                                                          ▼         │
│   ┌─────────┐    ┌───────────┐    ┌────────────┐    ┌──────────┐  │
│   │  User   │───▶│ Streamlit │───▶│   RAG      │◀──▶│ ChromaDB │  │
│   │  Query  │    │    UI     │    │  Pipeline   │    │ Retrieval│  │
│   └─────────┘    └───────────┘    └─────┬──────┘    └──────────┘  │
│                                         │                          │
│                                         ▼                          │
│                                   ┌────────────┐                   │
│                                   │  DeepInfra  │                  │
│                                   │  LLM API   │                  │
│                                   │ (Llama 3.1)│                  │
│                                   └─────┬──────┘                   │
│                                         │                          │
│                                         ▼                          │
│                                   ┌────────────┐                   │
│                                   │  Response   │                  │
│                                   │ + Sources  │                  │
│                                   │ + Latency  │                  │
│                                   └────────────┘                   │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Features

| Feature | Description |
|---|---|
| **RAG-Based Q&A** | Answers grounded in actual Upwork API documentation, not parametric memory |
| **Hallucination Prevention** | Multi-layer defense with similarity thresholds (0.6) and strict system prompts |
| **Source Attribution** | Every answer cites exact source snippets with page numbers for verification |
| **Response Latency Tracking** | Real-time measurement and display of API response times |
| **Persistent Vector Database** | ChromaDB persists embeddings to disk — no re-ingestion needed between sessions |
| **Professional UI** | Clean Streamlit interface with loading spinners and formatted output |

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.11+ | Core runtime |
| **Orchestration** | LangChain | RAG pipeline orchestration and prompt management |
| **Vector Database** | ChromaDB | Persistent vector storage and similarity search |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 | Document and query embedding generation |
| **Frontend** | Streamlit | Interactive web-based user interface |
| **LLM Provider** | DeepInfra | Cloud-hosted Meta-Llama-3.1-8B-Instruct-Turbo |
| **PDF Processing** | PyPDF | PDF document parsing and text extraction |

---

## Setup Instructions

### 1. Clone & Navigate

```bash
cd proanalystproject
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
copy .env.example .env
# Edit .env and add your DeepInfra API key
```

> **Note:** Get your API key from [https://deepinfra.com/](https://deepinfra.com/). The free tier provides sufficient credits for testing.

### 5. Prepare Data

Place the Upwork API documentation PDF in the `data/` folder:

```
data/
└── upwork_api_reference.pdf
```

### 6. Run Ingestion

```bash
python ingest.py
```

This will:
- Load and parse the PDF document
- Split text into optimized chunks (500 characters, 50-character overlap)
- Generate embeddings using sentence-transformers/all-MiniLM-L6-v2
- Store embeddings and metadata in ChromaDB (persisted to `chroma_db/`)

### 7. Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## Testing Questions

Use these evaluation questions to test the system's accuracy and reliability:

### Question 1: Rate Limits
> **"What are the rate limits for the Upwork API?"**
>
> Expected: The bot should return specific rate limit numbers, endpoint-level limits, and guidance on handling 429 errors — all cited from the documentation.

### Question 2: OAuth Token Validity
> **"How long is an OAuth token valid?"**
>
> Expected: The bot should provide the exact token expiration time, refresh token details, and re-authentication flow — with page number citations.

### Question 3: Client Credentials Grant
> **"Can I use client_credentials grant type with the Upwork API?"**
>
> Expected: The bot should clearly state whether this grant type is supported, reference the supported OAuth flows, and provide alternatives if applicable.

---

## Project Structure

```
proanalystproject/
│
├── app.py                  # Streamlit application (main entry point)
├── ingest.py               # PDF ingestion and vector database creation
├── rag.py                  # Core RAG pipeline (retrieval + generation)
│
├── data/                   # Source documents
│   └── upwork_api_reference.pdf
│
├── chroma_db/              # Persistent vector database (auto-generated)
│
├── utils/                  # Utility modules
│   ├── __init__.py
│   ├── pdf_loader.py       # PDF loading and diagnostics
│   ├── vector_store.py     # ChromaDB vector store management
│   ├── llm_client.py       # DeepInfra API client
│   └── prompts.py          # System prompts and templates
│
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .env                    # Local environment variables (git-ignored)
│
├── README.md               # This file
└── technical_summary.md    # Detailed technical summary and decisions
```

---

## How It Works

### 1. Document Ingestion & Chunking

The ingestion pipeline (`ingest.py`) loads the Upwork API PDF using PyPDF, extracts text with page-level metadata, and splits the content into manageable chunks using LangChain's `RecursiveCharacterTextSplitter`. Each chunk is 500 characters with a 50-character overlap to preserve context across boundaries.

### 2. Embedding Generation & Vector Storage

Each text chunk is transformed into a 384-dimensional dense vector using the `sentence-transformers/all-MiniLM-L6-v2` model. These embeddings, along with the original text and metadata (page numbers, source file), are stored in ChromaDB — a persistent, local vector database.

### 3. Semantic Retrieval

When a user submits a query, the system embeds the question using the same model and performs a similarity search against the vector store. The top-k (k=3) most semantically similar chunks are retrieved, along with their similarity scores. Chunks below the confidence threshold (0.6) are filtered out.

### 4. LLM-Powered Answer Generation

The retrieved context chunks are injected into a carefully engineered prompt template and sent to Meta-Llama-3.1-8B-Instruct-Turbo via the DeepInfra API. The system prompt strictly constrains the LLM to answer **only** based on the provided context.

### 5. Hallucination Prevention

Multiple safeguards prevent hallucinated responses:
- **Similarity threshold:** Queries with no relevant matches (score < 0.6) receive a transparent "I don't have enough information" response
- **Strict system prompt:** The LLM is instructed to never fabricate information and to admit uncertainty
- **Source attribution:** Every answer includes the exact source passages and page numbers, enabling user verification

---

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
