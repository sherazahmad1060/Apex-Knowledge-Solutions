"""
Apex Global Solutions - Company Knowledge Assistant
A production-ready RAG application powered by Streamlit, LangChain, FAISS, and OpenAI.
Elevated with animated gradient themes, preloader splash display, and ultra-smooth responsiveness.
"""

import os
import sys
import html
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
import streamlit.components.v1 as components

# Configure page metadata first before any other Streamlit call
st.set_page_config(
    page_title="Apex Knowledge Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import internal modules
from config import (
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
    OPENAI_EMBEDDING_MODEL,
    GEMINI_API_KEY,
    GEMINI_CHAT_MODEL,
    GEMINI_EMBEDDING_MODEL,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
    MIN_CHUNK_OVERLAP,
    MAX_CHUNK_OVERLAP,
    DEFAULT_RETRIEVAL_K,
    MIN_RETRIEVAL_K,
    MAX_RETRIEVAL_K,
    DATA_DIR,
    UPLOAD_DIR,
    ASSETS_DIR,
    SUPPORTED_EXTENSIONS,
    validate_api_key,
    mask_api_key,
    is_gemini_key,
    save_api_key_to_env,
    setup_logging,
)
from document_processor import (
    load_document,
    split_documents,
    validate_file,
    save_uploaded_file,
    remove_document_file,
    compute_file_hash,
)
from rag import RAGPipeline, FALLBACK_ANSWER

logger = setup_logging()

# Session state initialization
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = None

    raw_key = (GEMINI_API_KEY or OPENAI_API_KEY or "").strip()
    if raw_key in ["your_openai_api_key_here", "sk-placeholder", "your-key-here"]:
        raw_key = ""

    if (
        "api_key" not in st.session_state
        or not st.session_state.api_key
        or st.session_state.api_key in ["your_openai_api_key_here", "sk-placeholder", "your-key-here"]
    ):
        st.session_state.api_key = raw_key

    is_gemini = is_gemini_key(st.session_state.api_key)
    if "chat_model" not in st.session_state or not st.session_state.chat_model:
        st.session_state.chat_model = (
            GEMINI_CHAT_MODEL if is_gemini else OPENAI_CHAT_MODEL
        )
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = DEFAULT_CHUNK_SIZE
    if "chunk_overlap" not in st.session_state:
        st.session_state.chunk_overlap = DEFAULT_CHUNK_OVERLAP
    if "retrieval_k" not in st.session_state:
        st.session_state.retrieval_k = DEFAULT_RETRIEVAL_K
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "ui_theme" not in st.session_state:
        st.session_state.ui_theme = "Executive Aurora"
    if "show_preloader" not in st.session_state:
        st.session_state.show_preloader = True
    if "preloader_shown" not in st.session_state:
        st.session_state.preloader_shown = False


init_session_state()


# Theme definition presets
THEME_PRESETS = {
    "Executive Aurora": {
        "gradient": "linear-gradient(-45deg, #090d16, #0e1e38, #063445, #14274e)",
        "accent": "#38bdf8",
        "glow": "rgba(56, 189, 248, 0.35)",
        "card_border": "rgba(56, 189, 248, 0.2)",
        "badge_ready": "rgba(56, 189, 248, 0.15)",
        "header_grad": "linear-gradient(135deg, #0b132b 0%, #1c2541 40%, #0b525b 100%)",
    },
    "Cyber Deep Blue": {
        "gradient": "linear-gradient(-45deg, #030712, #1e1b4b, #1d4ed8, #312e81)",
        "accent": "#60a5fa",
        "glow": "rgba(96, 165, 250, 0.4)",
        "card_border": "rgba(96, 165, 250, 0.25)",
        "badge_ready": "rgba(96, 165, 250, 0.15)",
        "header_grad": "linear-gradient(135deg, #030712 0%, #1e1b4b 50%, #2563eb 100%)",
    },
    "Emerald Matrix": {
        "gradient": "linear-gradient(-45deg, #022c22, #064e3b, #047857, #065f46)",
        "accent": "#34d399",
        "glow": "rgba(52, 211, 153, 0.4)",
        "card_border": "rgba(52, 211, 153, 0.25)",
        "badge_ready": "rgba(52, 211, 153, 0.15)",
        "header_grad": "linear-gradient(135deg, #064e3b 0%, #022c22 60%, #10b981 100%)",
    },
    "Midnight Obsidian": {
        "gradient": "linear-gradient(-45deg, #18181b, #27272a, #3f3f46, #09090b)",
        "accent": "#f59e0b",
        "glow": "rgba(245, 158, 11, 0.35)",
        "card_border": "rgba(245, 158, 11, 0.2)",
        "badge_ready": "rgba(245, 158, 11, 0.15)",
        "header_grad": "linear-gradient(135deg, #18181b 0%, #27272a 50%, #b45309 100%)",
    },
}

current_theme = THEME_PRESETS.get(st.session_state.ui_theme, THEME_PRESETS["Executive Aurora"])

# Custom CSS with Animated Gradients, Preloader, Micro-Interactions & Responsive Rules
CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {{
    --theme-accent: {current_theme['accent']};
    --theme-glow: {current_theme['glow']};
    --theme-border: {current_theme['card_border']};
}}

html, body, [class*="css"] {{
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    letter-spacing: -0.01em;
}}

/* Animated Gradient Keyframes */
@keyframes gradientFlow {{
    0% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}

@keyframes pulseGlow {{
    0%, 100% {{
        box-shadow: 0 0 15px {current_theme['glow']};
        transform: scale(1);
    }}
    50% {{
        box-shadow: 0 0 25px {current_theme['glow']}, 0 0 45px {current_theme['glow']};
        transform: scale(1.02);
    }}
}}

@keyframes floatOrb {{
    0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
    50% {{ transform: translateY(-8px) rotate(3deg); }}
}}

@keyframes shimmer {{
    0% {{ transform: translateX(-150%); }}
    100% {{ transform: translateX(150%); }}
}}

@keyframes pulseDot {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: 0.4; transform: scale(0.85); }}
}}

/* Top Hero Header with Animated Gradient */
.main-header {{
    background: {current_theme['header_grad']};
    background-size: 250% 250%;
    animation: gradientFlow 14s ease infinite;
    padding: 2rem 2.2rem;
    border-radius: 20px;
    color: #ffffff;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255, 255, 255, 0.12);
    box-shadow: 0 12px 36px rgba(0, 0, 0, 0.25), 0 0 20px {current_theme['glow']};
    position: relative;
    overflow: hidden;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}}

.main-header::before {{
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 50%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.08), transparent);
    animation: shimmer 8s infinite;
}}

.main-header h1 {{
    font-size: 1.95rem;
    font-weight: 800;
    margin: 0;
    color: #ffffff;
    letter-spacing: -0.03em;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}}

.main-header p {{
    font-size: 0.96rem;
    margin-top: 0.5rem;
    margin-bottom: 0;
    color: #cbd5e1;
    line-height: 1.5;
    max-width: 850px;
}}

/* Dynamic Status Badges with Pulsing Dot */
.badge-ready {{
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: rgba(16, 185, 129, 0.14);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.35);
    padding: 0.35rem 0.85rem;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 600;
    box-shadow: 0 2px 10px rgba(16, 185, 129, 0.15);
}}

.badge-ready::before {{
    content: '';
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    animation: pulseDot 1.8s infinite ease-in-out;
}}

.badge-pending {{
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: rgba(245, 158, 11, 0.14);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.35);
    padding: 0.35rem 0.85rem;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 600;
}}

.badge-pending::before {{
    content: '';
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #f59e0b;
    animation: pulseDot 1.4s infinite ease-in-out;
}}

.badge-error {{
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: rgba(239, 68, 68, 0.14);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.35);
    padding: 0.35rem 0.85rem;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 600;
}}

.badge-error::before {{
    content: '';
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #ef4444;
}}

/* Glassmorphic Metric Cards */
div[data-testid="stMetric"] {{
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    border-radius: 14px;
    padding: 0.75rem 1rem;
    transition: all 0.3s ease;
}}

div[data-testid="stMetric"]:hover {{
    transform: translateY(-2px);
    border-color: {current_theme['accent']};
    box-shadow: 0 4px 20px {current_theme['glow']};
}}

/* Interactive Citation Cards with Metadata Pills */
.citation-box {{
    background: rgba(15, 23, 42, 0.65);
    border-left: 4px solid {current_theme['accent']};
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    border-right: 1px solid rgba(255, 255, 255, 0.06);
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(10px);
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
    border-radius: 0 12px 12px 0;
    font-size: 0.88rem;
    line-height: 1.55;
    transition: all 0.25s ease;
}}

.citation-box:hover {{
    transform: translateX(4px);
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
    border-left-color: #818cf8;
}}

.citation-header {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.6rem;
}}

.pill-doc {{
    background: rgba(56, 189, 248, 0.18);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.35);
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 600;
}}

.pill-sheet {{
    background: rgba(16, 185, 129, 0.18);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.35);
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 600;
}}

.pill-row {{
    background: rgba(245, 158, 11, 0.18);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.35);
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 600;
}}

.pill-chunk {{
    background: rgba(148, 163, 184, 0.12);
    color: #94a3b8;
    border: 1px solid rgba(148, 163, 184, 0.25);
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
}}

.citation-text {{
    color: #f1f5f9;
    font-size: 0.88rem;
    line-height: 1.55;
    background: rgba(0, 0, 0, 0.28);
    padding: 0.75rem 0.95rem;
    border-radius: 8px;
    border-left: 2px solid rgba(255, 255, 255, 0.2);
    margin-top: 0.4rem;
    word-break: break-word;
}}

/* Suggested Question Buttons with Hover Lift */
div[data-testid="stHorizontalBlock"] button {{
    border-radius: 24px !important;
    font-size: 0.86rem !important;
    padding: 0.55rem 0.95rem !important;
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    white-space: normal !important;
    height: auto !important;
    min-height: 48px !important;
}}

div[data-testid="stHorizontalBlock"] button:hover {{
    transform: translateY(-3px) scale(1.01) !important;
    border-color: {current_theme['accent']} !important;
    background: rgba(56, 189, 248, 0.1) !important;
    box-shadow: 0 6px 20px {current_theme['glow']} !important;
}}

/* Chat Message Elevated Containers */
div[data-testid="stChatMessage"] {{
    border-radius: 16px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1rem;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    transition: all 0.2s ease;
}}

div[data-testid="stChatMessage"]:hover {{
    border-color: rgba(255, 255, 255, 0.12);
}}

/* Responsiveness adjustments */
@media (max-width: 768px) {{
    .main-header {{
        padding: 1.4rem 1.25rem;
    }}
    .main-header h1 {{
        font-size: 1.45rem;
    }}
    .preloader-card {{
        padding: 1.8rem 1.5rem;
    }}
}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Preloader Component (Guaranteed strictly once per browser page refresh)
def render_preloader():
    if not st.session_state.get("show_preloader", True):
        return

    accent_color = current_theme['accent']
    glow_color = current_theme['glow']
    card_border = current_theme['card_border']

    splash_code = f"""
    <script>
    (function() {{
        try {{
            var parentWin = window.parent;
            var parentDoc = window.parent.document;

            // Hide this helper iframe immediately
            if (window.frameElement) {{
                window.frameElement.style.display = 'none';
                window.frameElement.style.height = '0px';
                window.frameElement.style.margin = '0px';
                window.frameElement.style.padding = '0px';
            }}

            // If splash already played in this page view or is currently mounted, NEVER recreate
            if (parentWin._apex_splash_played || parentDoc.getElementById('apex-preloader-root')) {{
                return;
            }}
            parentWin._apex_splash_played = true;

            // Inject styles once into parent head
            if (!parentDoc.getElementById('apex-preloader-styles')) {{
                var style = parentDoc.createElement('style');
                style.id = 'apex-preloader-styles';
                style.textContent = `
                    @keyframes floatOrb {{
                        0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
                        50% {{ transform: translateY(-8px) rotate(3deg); }}
                    }}
                    @keyframes pulseGlow {{
                        0%, 100% {{ box-shadow: 0 0 15px {glow_color}; transform: scale(1); }}
                        50% {{ box-shadow: 0 0 25px {glow_color}, 0 0 45px {glow_color}; transform: scale(1.02); }}
                    }}
                    @keyframes preloaderBarAnim {{
                        0% {{ width: 0%; }}
                        50% {{ width: 75%; }}
                        100% {{ width: 100%; }}
                    }}
                    #apex-preloader-root {{
                        position: fixed;
                        top: 0;
                        left: 0;
                        width: 100vw;
                        height: 100vh;
                        background: radial-gradient(circle at 50% 45%, #0f172a 0%, #020617 100%);
                        z-index: 99999999;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        pointer-events: all;
                        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
                        transition: opacity 0.35s cubic-bezier(0.4, 0, 0.2, 1);
                    }}
                    #apex-preloader-root .preloader-card {{
                        text-align: center;
                        padding: 2.2rem 3.2rem;
                        background: rgba(15, 23, 42, 0.85);
                        backdrop-filter: blur(20px);
                        border-radius: 24px;
                        border: 1px solid {card_border};
                        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6), 0 0 40px {glow_color};
                        max-width: 480px;
                        width: 90%;
                        animation: floatOrb 4s ease-in-out infinite;
                    }}
                    #apex-preloader-root .preloader-logo {{
                        width: 76px;
                        height: 76px;
                        margin: 0 auto 1.25rem;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        border-radius: 50%;
                        background: linear-gradient(135deg, rgba(56, 189, 248, 0.2), rgba(99, 102, 241, 0.2));
                        border: 2px solid {accent_color};
                        animation: pulseGlow 3s infinite ease-in-out;
                        font-size: 2.3rem;
                    }}
                    #apex-preloader-root .preloader-title {{
                        font-size: 1.4rem;
                        font-weight: 800;
                        color: #ffffff;
                        margin-bottom: 0.35rem;
                        letter-spacing: -0.02em;
                        background: linear-gradient(90deg, #ffffff, {accent_color}, #ffffff);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                    }}
                    #apex-preloader-root .preloader-sub {{
                        font-size: 0.86rem;
                        color: #94a3b8;
                        margin-bottom: 1.4rem;
                    }}
                    #apex-preloader-root .preloader-track {{
                        width: 100%;
                        height: 6px;
                        background: rgba(255, 255, 255, 0.1);
                        border-radius: 999px;
                        overflow: hidden;
                        margin-bottom: 1rem;
                    }}
                    #apex-preloader-root .preloader-fill {{
                        height: 100%;
                        width: 100%;
                        background: linear-gradient(90deg, {accent_color}, #818cf8, {accent_color});
                        background-size: 200% 100%;
                        border-radius: 999px;
                        animation: preloaderBarAnim 1.1s ease-in-out forwards;
                    }}
                    #apex-preloader-root .preloader-status {{
                        font-size: 0.8rem;
                        color: {accent_color};
                        font-family: 'JetBrains Mono', monospace;
                        display: inline-flex;
                        align-items: center;
                        gap: 0.5rem;
                    }}
                `;
                parentDoc.head.appendChild(style);
            }}

            // Mount the overlay directly to parent body outside React's control
            var overlay = parentDoc.createElement('div');
            overlay.id = 'apex-preloader-root';
            overlay.innerHTML = `
                <div class="preloader-card">
                    <div class="preloader-logo">🏢</div>
                    <div class="preloader-title">APEX GLOBAL SOLUTIONS</div>
                    <div class="preloader-sub">Enterprise RAG Neural Assistant</div>
                    <div class="preloader-track">
                        <div class="preloader-fill"></div>
                    </div>
                    <div class="preloader-status">⚡ Initializing Vector Engine & Knowledge Base...</div>
                </div>
            `;
            parentDoc.body.appendChild(overlay);

            // Exactly once: fade out after 1.25s and cleanly remove from body
            setTimeout(function() {{
                overlay.style.opacity = '0';
                setTimeout(function() {{
                    if (overlay && overlay.parentNode) {{
                        overlay.parentNode.removeChild(overlay);
                    }}
                }}, 380);
            }}, 1250);

        }} catch(e) {{
            console.error('Preloader initialization error:', e);
        }}
    }})();
    </script>
    """
    components.html(splash_code, height=0)


render_preloader()


def get_pipeline() -> RAGPipeline:
    """Retrieve or instantiate the RAGPipeline instance."""
    if st.session_state.rag_pipeline is None:
        st.session_state.rag_pipeline = RAGPipeline(
            api_key=st.session_state.api_key,
            chat_model_name=st.session_state.chat_model,
        )
    else:
        # Dynamically sync API key, provider, and reload vector store if needed
        st.session_state.rag_pipeline.update_config(
            api_key=st.session_state.api_key,
            chat_model_name=st.session_state.chat_model,
        )
    return st.session_state.rag_pipeline


def discover_available_documents() -> List[Path]:
    """Find all valid documents in both data/ and data/uploads/."""
    docs: List[Path] = []
    # Base data directory (e.g. sample data)
    for p in DATA_DIR.glob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.append(p)
    # Uploaded files
    for p in UPLOAD_DIR.glob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.append(p)
    # Deduplicate by filename
    seen = {}
    for p in docs:
        if p.name not in seen:
            seen[p.name] = p
    return list(seen.values())


def process_all_documents(pipeline: RAGPipeline, chunk_size: int, chunk_overlap: int) -> tuple[bool, str]:
    """Process and index all discovered documents into FAISS."""
    is_valid, msg = validate_api_key(pipeline.api_key)
    if not is_valid:
        return False, f"Cannot process documents: {msg}"

    doc_paths = discover_available_documents()
    if not doc_paths:
        return False, "No documents found to process. Please upload a TXT or XLSX file first."

    chunks_map = {}
    total_chunks = 0

    try:
        for p in doc_paths:
            file_hash = compute_file_hash(p)
            doc_id = file_hash[:10]
            raw_docs = load_document(p, doc_id=doc_id)
            if not raw_docs:
                continue
            chunks = split_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks_map[p.name] = (doc_id, file_hash, chunks)
            total_chunks += len(chunks)

        if not chunks_map or total_chunks == 0:
            return False, "Failed to extract valid text chunks from the selected documents."

        pipeline.build_index(chunks_map)
        return True, f"Successfully indexed {len(chunks_map)} document(s) with {total_chunks} chunk(s)."

    except Exception as e:
        logger.error(f"Error during document processing: {e}", exc_info=True)
        return False, f"Processing failed: {str(e)}"


# ---------------------------------------------------------
# Sidebar Component
# ---------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        # Logo & App Title
        logo_path = ASSETS_DIR / "logo.png"
        if logo_path.exists():
            st.image(str(logo_path), width=74)
        st.title("Apex Knowledge Assistant")
        st.caption("Enterprise RAG Knowledge System")
        st.divider()

        # UI Theme Selector
        st.subheader("🎨 Gradient Theme")
        chosen_theme = st.selectbox(
            "Select Interface Palette",
            options=list(THEME_PRESETS.keys()),
            index=list(THEME_PRESETS.keys()).index(st.session_state.ui_theme),
            help="Choose an animated gradient theme preset.",
        )
        if chosen_theme != st.session_state.ui_theme:
            st.session_state.ui_theme = chosen_theme
            st.rerun()

        st.divider()

        # API Configuration Status
        pipeline = get_pipeline()
        is_api_valid, api_msg = validate_api_key(st.session_state.api_key)

        st.subheader("🔑 API Configuration")
        is_gemini = is_gemini_key(st.session_state.api_key)
        provider_name = "Google Gemini" if is_gemini else "OpenAI"
        if is_api_valid:
            st.markdown(
                f"<span class='badge-ready'>Connected ({provider_name})</span> <span style='font-size:0.85rem; color:#94a3b8; margin-left:6px;'>({mask_api_key(st.session_state.api_key)})</span>",
                unsafe_allow_html=True,
            )
            with st.expander("Change / Update Key"):
                new_key = st.text_input(
                    "Enter new API Key",
                    type="password",
                    placeholder="sk-... or AQ...",
                    key="sidebar_change_key_field",
                )
                if st.button("Apply & Save Key", key="save_new_key_btn", use_container_width=True):
                    if new_key:
                        st.session_state.api_key = new_key.strip()
                        save_api_key_to_env(new_key.strip())
                        pipeline.update_config(api_key=new_key.strip())
                        st.success("API key updated and saved to .env!")
                        st.rerun()
        else:
            st.markdown("<span class='badge-error'>API Key Required</span>", unsafe_allow_html=True)
            new_key = st.text_input(
                "Enter API Key (OpenAI or Gemini)",
                type="password",
                placeholder="sk-... or AQ...",
                help="Your API key will be saved to your local .env configuration.",
                key="sidebar_initial_key_field",
            )
            if new_key:
                st.session_state.api_key = new_key.strip()
                save_api_key_to_env(new_key.strip())
                pipeline.update_config(api_key=new_key.strip())
                st.rerun()

        st.divider()

        # Knowledge Base Status & Metrics
        st.subheader("📚 Knowledge Base")
        available_docs = discover_available_documents()
        is_ready = pipeline.is_ready()

        if is_ready:
            st.markdown(
                f"<span class='badge-ready'>Ready ({pipeline.state.total_chunks} Chunks)</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<span class='badge-pending'>Index Pending</span>",
                unsafe_allow_html=True,
            )

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", len(available_docs))
        with col2:
            st.metric("Indexed Chunks", pipeline.state.total_chunks)

        st.divider()

        # Document Upload
        st.subheader("📤 Upload Documents")
        st.caption(f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")

        uploaded_files = st.file_uploader(
            "Choose TXT or XLSX files",
            type=["txt", "xlsx"],
            accept_multiple_files=True,
            help="Files are validated and stored locally in data/uploads.",
        )

        if uploaded_files:
            upload_count = 0
            for uploaded_file in uploaded_files:
                file_bytes = uploaded_file.read()
                is_valid, err_msg = validate_file(uploaded_file.name, len(file_bytes))
                if not is_valid:
                    st.error(f"Error for '{uploaded_file.name}': {err_msg}")
                    continue

                saved_path, doc_id, file_hash = save_uploaded_file(file_bytes, uploaded_file.name)
                upload_count += 1

            if upload_count > 0:
                st.success(f"Saved {upload_count} file(s). Click 'Process Documents' below to index.")

        # Document Management List
        if available_docs:
            st.markdown("**Available Documents:**")
            for doc in available_docs:
                size_kb = round(doc.stat().st_size / 1024, 1)
                is_indexed = doc.name in pipeline.state.indexed_files
                badge = "✅" if is_indexed else "⏳"
                st.text(f"{badge} {doc.name} ({size_kb} KB)")

        # Process Documents Button
        if st.button("⚡ Process / Re-index Documents", use_container_width=True, type="primary"):
            with st.spinner("Parsing documents and generating embeddings..."):
                success, message = process_all_documents(
                    pipeline,
                    st.session_state.chunk_size,
                    st.session_state.chunk_overlap,
                )
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

        # Remove Document with Confirmation
        if available_docs:
            with st.expander("🗑️ Remove Document"):
                doc_to_remove = st.selectbox(
                    "Select document to delete",
                    options=[d.name for d in available_docs],
                    key="delete_doc_select",
                )
                confirm_del = st.checkbox(f"Confirm deletion of '{doc_to_remove}'")
                if st.button("Delete Document", disabled=not confirm_del, use_container_width=True):
                    # Check if it's in uploads or data
                    target_p = UPLOAD_DIR / doc_to_remove
                    if not target_p.exists():
                        target_p = DATA_DIR / doc_to_remove
                    if target_p.exists():
                        target_p.unlink()
                        st.info(f"Removed '{doc_to_remove}'. Re-indexing knowledge base...")
                        process_all_documents(
                            pipeline,
                            st.session_state.chunk_size,
                            st.session_state.chunk_overlap,
                        )
                        st.rerun()

        # Clear Chat Button
        st.divider()
        if st.button("🧹 Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # Advanced Settings
        with st.expander("⚙️ Advanced RAG Settings"):
            is_gemini_active = is_gemini_key(st.session_state.api_key)
            model_options = (
                ["gemini-flash-lite-latest", "gemini-3.8-flash", "gemini-pro-latest"]
                if is_gemini_active
                else ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
            )
            current_idx = (
                model_options.index(st.session_state.chat_model)
                if st.session_state.chat_model in model_options
                else 0
            )
            st.session_state.chat_model = st.selectbox(
                "Chat Model",
                options=model_options,
                index=current_idx,
            )
            st.session_state.retrieval_k = st.slider(
                "Retrieval Count (Top K)",
                min_value=MIN_RETRIEVAL_K,
                max_value=MAX_RETRIEVAL_K,
                value=st.session_state.retrieval_k,
                help="Number of relevant chunks to pass to the model.",
            )
            st.session_state.chunk_size = st.slider(
                "Chunk Size (characters)",
                min_value=MIN_CHUNK_SIZE,
                max_value=MAX_CHUNK_SIZE,
                value=st.session_state.chunk_size,
                step=50,
            )
            st.session_state.chunk_overlap = st.slider(
                "Chunk Overlap (characters)",
                min_value=MIN_CHUNK_OVERLAP,
                max_value=min(MAX_CHUNK_OVERLAP, st.session_state.chunk_size - 10),
                value=min(st.session_state.chunk_overlap, st.session_state.chunk_size - 10),
                step=10,
            )
            st.session_state.show_preloader = st.checkbox(
                "Enable Preloader Splash on Refresh",
                value=st.session_state.show_preloader,
            )


def render_citation_box(src: Dict[str, Any], s_idx: int) -> str:
    """Render a clean, sanitized HTML card for a source citation without markdown indentation bugs."""
    source_name = html.escape(str(src.get("source", "Unknown")))
    sheet_val = src.get("sheet_name")
    row_val = src.get("row_index")
    chunk_id = html.escape(str(src.get("chunk_id", f"c{s_idx + 1}")))
    raw_excerpt = str(src.get("excerpt", "")).strip()
    escaped_excerpt = html.escape(raw_excerpt).replace("\n", "<br/>")

    sheet_pill = (
        f"<span class='pill-sheet'>Sheet: {html.escape(str(sheet_val))}</span>"
        if sheet_val and str(sheet_val).strip() not in ["N/A", ""]
        else ""
    )
    row_pill = (
        f"<span class='pill-row'>Row: {html.escape(str(row_val))}</span>"
        if row_val and str(row_val).strip() not in ["N/A", ""]
        else ""
    )
    chunk_pill = f"<span class='pill-chunk'>#{chunk_id}</span>"

    return (
        f'<div class="citation-box">'
        f'<div class="citation-header">'
        f'<span class="pill-doc">📄 {source_name}</span>'
        f'{sheet_pill}'
        f'{row_pill}'
        f'{chunk_pill}'
        f'</div>'
        f'<div class="citation-text">"{escaped_excerpt}"</div>'
        f'</div>'
    )


# ---------------------------------------------------------
# Main Chat Area
# ---------------------------------------------------------
def render_main_area():
    pipeline = get_pipeline()
    is_ready = pipeline.is_ready()

    # Welcome Header with Dynamic Flowing Gradient
    st.markdown(
        """
        <div class="main-header">
            <h1>🏢 Apex Global Knowledge Assistant</h1>
            <p>Empowering enterprise teams with strictly grounded answers from corporate documents, HR handbooks, and multi-sheet product catalogs with verified citations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Prompt if knowledge base isn't ready
    if not is_ready:
        is_api_valid, _ = validate_api_key(pipeline.api_key)
        if is_api_valid:
            available_docs = discover_available_documents()
            st.info(
                f"👋 **Knowledge Base Ready to Initialize**: Found {len(available_docs)} corporate document(s). "
                "You can click **'⚡ Process Documents'** in the sidebar, or simply **type your question below** and it will automatically index and answer!"
            )
        else:
            st.warning(
                "👋 **Welcome!** Please enter your OpenAI or Google Gemini API key in the sidebar to activate the assistant."
            )

    # Suggested Questions
    st.markdown("**💡 Quick Inquiries:**")
    suggested_cols = st.columns(4)
    suggested_questions = [
        "What products does our company offer?",
        "What is our company remote work policy?",
        "What are the prices of our products?",
        "Summarize the information in our company documents.",
    ]

    for i, q in enumerate(suggested_questions):
        with suggested_cols[i]:
            if st.button(q, key=f"sugg_{i}", use_container_width=True):
                st.session_state.pending_question = q

    st.divider()

    # Display Conversation History
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])

        with st.chat_message(role):
            st.markdown(content)

            # Display structured citations for assistant answers
            if role == "assistant" and sources:
                with st.expander(f"📑 Verified Citations ({len(sources)} sources cited)"):
                    for s_idx, src in enumerate(sources):
                        st.markdown(render_citation_box(src, s_idx), unsafe_allow_html=True)

    # Question Input Processing (Never blocked)
    user_input = st.chat_input(
        "Ask a question about company documents, policies, or products...",
    )

    # Handle clicked suggested questions
    active_question = user_input
    if st.session_state.pending_question:
        active_question = st.session_state.pending_question
        st.session_state.pending_question = None

    if active_question:
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": active_question})
        with st.chat_message("user"):
            st.markdown(active_question)

        # Assistant Response with Spinner
        with st.chat_message("assistant"):
            # Step 1: Validate API Key
            is_valid_key, key_err = validate_api_key(pipeline.api_key)
            if not is_valid_key:
                err_text = f"⚠️ **API Key Required**: {key_err}\n\nPlease enter your API key in the sidebar to activate the assistant."
                st.warning(err_text)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": err_text,
                    "sources": [],
                })
            else:
                # Step 2: Auto-index if knowledge base is not ready yet
                if not pipeline.is_ready():
                    with st.spinner("⚡ Initializing knowledge base from company documents (one-time setup)..."):
                        success, message = process_all_documents(
                            pipeline,
                            st.session_state.chunk_size,
                            st.session_state.chunk_overlap,
                        )
                        if not success:
                            err_msg = f"❌ Could not initialize knowledge base: {message}"
                            st.error(err_msg)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": err_msg,
                                "sources": [],
                            })
                            return

                # Step 3: Run RAG Query
                with st.spinner("Searching company documents and generating verified answer..."):
                    try:
                        result = pipeline.query(
                            active_question,
                            k=st.session_state.retrieval_k,
                        )
                        answer = result["answer"]
                        sources = result["sources"]

                        st.markdown(answer)

                        if sources:
                            with st.expander(f"📑 Verified Citations ({len(sources)} sources cited)"):
                                for s_idx, src in enumerate(sources):
                                    st.markdown(render_citation_box(src, s_idx), unsafe_allow_html=True)

                        # Save to conversation history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        })

                    except Exception as e:
                        logger.error(f"Error handling user question: {e}", exc_info=True)
                        error_msg = f"An error occurred while generating the response: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "sources": [],
                        })


def main():
    render_sidebar()
    render_main_area()


if __name__ == "__main__":
    main()
