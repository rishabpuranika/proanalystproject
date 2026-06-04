# Force pysqlite3 override on Linux systems (like Streamlit Cloud) that run old SQLite3 versions
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

"""
Upwork API Technical Support Assistant
========================================

A Streamlit web application that provides AI-powered technical support
for Upwork API developers using Retrieval-Augmented Generation (RAG).

Features:
    - Natural language question answering grounded in official documentation.
    - Source attribution with page numbers and exact snippets.
    - Hallucination prevention via confidence thresholds.
    - Response latency tracking.
    - Persistent chat history within the session.

Usage:
    streamlit run streamlit_app.py
"""

import os
# Force pure-Python implementation for protobuf to bypass descriptor conflicts on Streamlit Cloud
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import logging
import time

import streamlit as st
from dotenv import load_dotenv

from typing import Any
from utils.vector_store import load_vector_store, retrieve_context
from utils.llm_client import query_llm
from utils.prompts import (
    SYSTEM_PROMPT,
    HALLUCINATION_GUARD_MESSAGE,
    SIMILARITY_THRESHOLD,
    build_user_prompt,
)

# ---------------------------------------------------------------------------
# RAG Constants
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR: str = "chroma_db"
TOP_K: int = 3  # Number of chunks to retrieve per query

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Upwork API Technical Support Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS — Professional, modern dark-theme friendly styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #FF6B00 0%, #FF8C00 50%, #E85D00 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(255, 107, 0, 0.3);
    }

    .main-header h1 {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.05rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }

    /* Answer container */
    .answer-container {
        background: linear-gradient(145deg, #0a0a0a 0%, #1a1a1a 100%);
        border: 1px solid rgba(255, 107, 0, 0.3);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        color: #e0e0e0;
        line-height: 1.7;
    }

    /* Source container */
    .source-container {
        background: rgba(255, 107, 0, 0.06);
        border-left: 4px solid #FF6B00;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.5rem;
        margin: 0.75rem 0;
        font-size: 0.9rem;
        color: #c0c0c0;
    }

    .source-container .source-header {
        color: #FF8C00;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }

    /* Metrics bar */
    .metrics-bar {
        display: flex;
        gap: 1.5rem;
        margin: 1rem 0;
        flex-wrap: wrap;
    }

    .metric-badge {
        background: linear-gradient(135deg, rgba(255, 107, 0, 0.12), rgba(255, 140, 0, 0.12));
        border: 1px solid rgba(255, 107, 0, 0.25);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-size: 0.85rem;
        color: #FFa040;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Sidebar styling */
    .sidebar-section {
        background: rgba(255, 107, 0, 0.06);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }

    .sidebar-section h3 {
        color: #FF6B00;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.75rem;
    }

    /* Question history */
    .history-item {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        transition: all 0.2s ease;
    }

    .history-item:hover {
        border-color: rgba(255, 107, 0, 0.3);
        box-shadow: 0 2px 12px rgba(255, 107, 0, 0.1);
    }

    .history-question {
        color: #FF8C00;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #FF6B00 0%, #FF8C00 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 107, 0, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(255, 107, 0, 0.4);
    }

    /* Divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 107, 0, 0.3), transparent);
        margin: 1.5rem 0;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "selected_question" not in st.session_state:
    st.session_state.selected_question = ""


# ---------------------------------------------------------------------------
# RAG Core Pipeline
# ---------------------------------------------------------------------------
def answer_question(question: str) -> dict[str, Any]:
    """
    Process a user question through the RAG pipeline.
    """
    if not question or not question.strip():
        raise ValueError("Question must not be empty.")

    logger.info("=" * 50)
    logger.info("Processing question: '%s'", question[:100])
    logger.info("=" * 50)

    # Step 1: Load the vector store
    logger.info("Loading vector store from '%s'...", CHROMA_PERSIST_DIR)
    try:
        vector_store = load_vector_store(persist_directory=CHROMA_PERSIST_DIR)
    except FileNotFoundError as exc:
        logger.error("Vector store not found: %s", exc)
        raise RuntimeError(
            "Vector database not found. Please run 'python ingest.py' first "
            "to process the documentation and create the database."
        ) from exc

    # Step 2: Retrieve relevant context chunks
    logger.info("Retrieving top %d context chunks...", TOP_K)
    sources: list[dict] = retrieve_context(
        query=question,
        vector_store=vector_store,
        k=TOP_K,
    )

    if not sources:
        logger.warning("No chunks retrieved for the query.")
        return {
            "answer": HALLUCINATION_GUARD_MESSAGE,
            "sources": [],
            "latency": 0.0,
            "confidence": float("inf"),
        }

    # Step 3: Hallucination Guard — check retrieval confidence
    best_distance: float = min(chunk["score"] for chunk in sources)
    logger.info(
        "Best similarity distance: %.4f (threshold: %.2f)",
        best_distance,
        SIMILARITY_THRESHOLD,
    )

    if best_distance > SIMILARITY_THRESHOLD:
        logger.warning(
            "Retrieval confidence too low (%.4f > %.2f). "
            "Returning hallucination guard message.",
            best_distance,
            SIMILARITY_THRESHOLD,
        )
        return {
            "answer": HALLUCINATION_GUARD_MESSAGE,
            "sources": [],
            "latency": 0.0,
            "confidence": best_distance,
        }

    # Step 4: Build the grounded prompt
    user_prompt: str = build_user_prompt(
        context_chunks=sources,
        question=question,
    )
    logger.info("Built user prompt (%d characters).", len(user_prompt))

    # Step 5: Query the LLM
    logger.info("Sending query to LLM...")
    try:
        answer_text, latency = query_llm(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
    except Exception as exc:
        logger.error("LLM query failed: %s", exc)
        raise RuntimeError(f"Failed to get response from LLM: {exc}") from exc

    logger.info("Received answer (%.2fs, %d chars).", latency, len(answer_text))

    # Step 6: Return structured response
    return {
        "answer": answer_text,
        "sources": sources,
        "latency": latency,
        "confidence": best_distance,
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-section">
        <h3>📖 About</h3>
        <p style="font-size: 0.85rem; color: #a0a0a0; line-height: 1.6;">
            This assistant answers technical questions about the
            <strong>Upwork API</strong> using Retrieval-Augmented Generation.
            All answers are grounded in the official API documentation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <h3>⚡ How It Works</h3>
        <p style="font-size: 0.85rem; color: #a0a0a0; line-height: 1.6;">
            1. Your question is embedded as a vector<br>
            2. Similar documentation chunks are retrieved<br>
            3. Context is sent to Meta-Llama 3.1<br>
            4. Answer is generated from documentation only
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### 💡 Sample Questions")

    sample_questions = [
        "What is the specific request-per-second rate limit for the Upwork API, and is it enforced per Key or per IP?",
        "How long is an OAuth access token valid for?",
        "Can I use a Client Credentials Grant to access a user's private contract details?",
    ]

    for i, sq in enumerate(sample_questions):
        if st.button(sq, key=f"sample_{i}", use_container_width=True):
            st.session_state.selected_question = sq

    st.markdown("---")

    # Clear history button
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown(
        "<p style='text-align: center; color: #555; font-size: 0.75rem; margin-top: 2rem;'>"
        "Powered by Meta-Llama 3.1 via DeepInfra<br>"
        "Built with LangChain + ChromaDB</p>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 Upwork API Technical Support Assistant</h1>
    <p>Ask any question about the Upwork API — powered by RAG with official documentation.</p>
</div>
""", unsafe_allow_html=True)

# Question input area
col_input, col_btn = st.columns([5, 1])

# Use selected sample question if available
default_value = st.session_state.selected_question if st.session_state.selected_question else ""

with col_input:
    user_question: str = st.text_input(
        "Your Question",
        value=default_value,
        placeholder="e.g., How do I authenticate with the Upwork API using OAuth 2.0?",
        label_visibility="collapsed",
        key="question_input",
    )

with col_btn:
    ask_clicked: bool = st.button("🔍 Ask", use_container_width=True)

# Clear the selected question after it's been placed in the input
if st.session_state.selected_question:
    st.session_state.selected_question = ""


# ---------------------------------------------------------------------------
# Process Question
# ---------------------------------------------------------------------------
if ask_clicked and user_question.strip():
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    with st.spinner("🔍 Searching documentation and generating answer..."):
        try:
            # Call the RAG pipeline
            result = answer_question(user_question.strip())

            # Store in session history
            st.session_state.chat_history.insert(0, {
                "question": user_question.strip(),
                "result": result,
            })

        except RuntimeError as e:
            st.error(f"⚠️ **Error:** {str(e)}")
            logger.error("RAG pipeline error: %s", e)

        except Exception as e:
            st.error(f"⚠️ **Unexpected error:** {str(e)}")
            logger.error("Unexpected error: %s", e)

elif ask_clicked and not user_question.strip():
    st.warning("Please enter a question before clicking Ask.")


# ---------------------------------------------------------------------------
# Display Chat History
# ---------------------------------------------------------------------------
for idx, entry in enumerate(st.session_state.chat_history):
    question = entry["question"]
    result = entry["result"]

    # Question display
    st.markdown(
        f'<div class="history-question">💬 {question}</div>',
        unsafe_allow_html=True,
    )

    # Metrics bar
    latency_display = f"{result['latency']:.2f}s" if result['latency'] > 0 else "N/A (guard)"
    confidence_val = result.get("confidence", float("inf"))
    confidence_display = f"{confidence_val:.4f}" if confidence_val != float("inf") else "N/A"

    st.markdown(
        f"""<div class="metrics-bar">
            <span class="metric-badge">⏱️ Response Time: {latency_display}</span>
            <span class="metric-badge">🎯 Confidence Distance: {confidence_display}</span>
            <span class="metric-badge">📄 Sources: {len(result.get('sources', []))}</span>
        </div>""",
        unsafe_allow_html=True,
    )

    # Answer display
    st.markdown(
        f'<div class="answer-container">{result["answer"]}</div>',
        unsafe_allow_html=True,
    )

    # Sources display
    sources = result.get("sources", [])
    if sources:
        with st.expander(f"📚 View Sources ({len(sources)} chunks retrieved)", expanded=False):
            for src_idx, source in enumerate(sources, 1):
                page = source.get("page", "N/A")
                score = source.get("score", "N/A")
                content = source.get("content", "")

                st.markdown(
                    f"""<div class="source-container">
                        <div class="source-header">Source {src_idx} (Page {page}) — Distance: {score}</div>
                        <p style="font-style: italic; margin: 0;">"{content}"</p>
                    </div>""",
                    unsafe_allow_html=True,
                )

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Empty State
# ---------------------------------------------------------------------------
if not st.session_state.chat_history:
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem; color: #666;">
        <p style="font-size: 3rem; margin-bottom: 1rem;">🔍</p>
        <p style="font-size: 1.1rem; font-weight: 500;">Ask a question about the Upwork API</p>
        <p style="font-size: 0.9rem; color: #888;">
            Try one of the sample questions from the sidebar, or type your own question above.
        </p>
    </div>
    """, unsafe_allow_html=True)
