"""
RAG Pipeline Module.
Implements FAISS vector store management, similarity retrieval, and OpenAI-grounded answer generation.
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
except ImportError:
    GoogleGenerativeAIEmbeddings = None
    ChatGoogleGenerativeAI = None

from config import (
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
    OPENAI_EMBEDDING_MODEL,
    GEMINI_API_KEY,
    GEMINI_CHAT_MODEL,
    GEMINI_EMBEDDING_MODEL,
    VECTOR_STORE_DIR,
    DEFAULT_RETRIEVAL_K,
    SYSTEM_PROMPT,
    validate_api_key,
    is_gemini_key,
)

logger = logging.getLogger("company_chatbot.rag")
MANIFEST_FILENAME = "manifest.json"
INDEX_NAME = "index"
FALLBACK_ANSWER = "I don't know."


class KnowledgeBaseState:
    """Represents the manifest of the indexed knowledge base."""

    def __init__(self, embedding_model: str = OPENAI_EMBEDDING_MODEL):
        self.embedding_model = embedding_model
        self.indexed_files: Dict[str, Dict[str, Any]] = {}
        self.total_chunks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "embedding_model": self.embedding_model,
            "total_chunks": self.total_chunks,
            "indexed_files": self.indexed_files,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeBaseState":
        state = cls(embedding_model=data.get("embedding_model", OPENAI_EMBEDDING_MODEL))
        state.total_chunks = data.get("total_chunks", 0)
        state.indexed_files = data.get("indexed_files", {})
        return state


class RAGPipeline:
    """
    RAG Pipeline handling vector store operations and grounded question answering.
    Supports both OpenAI and Google Gemini models.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        chat_model_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
    ):
        self.api_key = (api_key or OPENAI_API_KEY or GEMINI_API_KEY or "").strip()
        self.is_gemini = is_gemini_key(self.api_key)

        if self.is_gemini:
            self.chat_model_name = (chat_model_name or GEMINI_CHAT_MODEL).strip()
            self.embedding_model_name = (embedding_model_name or GEMINI_EMBEDDING_MODEL).strip()
        else:
            self.chat_model_name = (chat_model_name or OPENAI_CHAT_MODEL).strip()
            self.embedding_model_name = (embedding_model_name or OPENAI_EMBEDDING_MODEL).strip()

        self.vector_store: Optional[FAISS] = None
        self.state = KnowledgeBaseState(embedding_model=self.embedding_model_name)
        self.load_vector_store()

    def update_config(
        self,
        api_key: Optional[str] = None,
        chat_model_name: Optional[str] = None,
    ) -> None:
        """
        Dynamically update API key and model names, re-evaluating provider and loading index.
        """
        cleaned_key = (api_key if api_key is not None else self.api_key).strip()
        key_changed = (cleaned_key != self.api_key)
        self.api_key = cleaned_key
        self.is_gemini = is_gemini_key(self.api_key)

        if self.is_gemini:
            self.embedding_model_name = GEMINI_EMBEDDING_MODEL
            if chat_model_name and not chat_model_name.startswith("gpt"):
                self.chat_model_name = chat_model_name
            else:
                self.chat_model_name = GEMINI_CHAT_MODEL
        else:
            self.embedding_model_name = OPENAI_EMBEDDING_MODEL
            if chat_model_name and not chat_model_name.startswith("gemini"):
                self.chat_model_name = chat_model_name
            else:
                self.chat_model_name = OPENAI_CHAT_MODEL

        self.state.embedding_model = self.embedding_model_name

        if key_changed or self.vector_store is None:
            self.load_vector_store()

    def get_embeddings(self):
        """Initialize and return the appropriate Embeddings model (Gemini or OpenAI)."""
        is_valid, msg = validate_api_key(self.api_key)
        if not is_valid:
            raise ValueError(f"Cannot initialize embeddings: {msg}")

        if self.is_gemini:
            if GoogleGenerativeAIEmbeddings is None:
                raise ImportError("langchain-google-genai is not installed.")
            return GoogleGenerativeAIEmbeddings(
                model=self.embedding_model_name,
                google_api_key=self.api_key,
            )

        return OpenAIEmbeddings(
            model=self.embedding_model_name,
            openai_api_key=self.api_key,
        )

    def get_chat_model(self):
        """Initialize and return the appropriate Chat model (Gemini or OpenAI)."""
        is_valid, msg = validate_api_key(self.api_key)
        if not is_valid:
            raise ValueError(f"Cannot initialize chat model: {msg}")

        if self.is_gemini:
            if ChatGoogleGenerativeAI is None:
                raise ImportError("langchain-google-genai is not installed.")
            return ChatGoogleGenerativeAI(
                model=self.chat_model_name,
                google_api_key=self.api_key,
                temperature=0.0,
            )

        return ChatOpenAI(
            model=self.chat_model_name,
            openai_api_key=self.api_key,
            temperature=0.0,
        )

    def manifest_path(self) -> Path:
        return VECTOR_STORE_DIR / MANIFEST_FILENAME

    def load_manifest(self) -> None:
        path = self.manifest_path()
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.state = KnowledgeBaseState.from_dict(data)
            except Exception as e:
                logger.warning(f"Could not load manifest: {e}")
                self.state = KnowledgeBaseState(embedding_model=self.embedding_model_name)

    def save_manifest(self) -> None:
        path = self.manifest_path()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.state.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write manifest: {e}")

    def load_vector_store(self) -> bool:
        """
        Load existing FAISS index from disk if present and compatible.
        """
        index_file = VECTOR_STORE_DIR / f"{INDEX_NAME}.faiss"
        pkl_file = VECTOR_STORE_DIR / f"{INDEX_NAME}.pkl"

        if not (index_file.exists() and pkl_file.exists()):
            self.vector_store = None
            return False

        self.load_manifest()

        # Check embedding model compatibility
        if self.state.embedding_model != self.embedding_model_name:
            logger.warning(
                f"Index was built with {self.state.embedding_model}, "
                f"current model is {self.embedding_model_name}. Re-indexing required."
            )
            self.vector_store = None
            return False

        is_valid, _ = validate_api_key(self.api_key)
        if not is_valid:
            # Cannot load without valid embeddings instance
            return False

        try:
            embeddings = self.get_embeddings()
            self.vector_store = FAISS.load_local(
                folder_path=str(VECTOR_STORE_DIR),
                embeddings=embeddings,
                index_name=INDEX_NAME,
                allow_dangerous_deserialization=True,
            )
            logger.info("Successfully loaded existing FAISS vector store from disk.")
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            self.vector_store = None
            return False

    def build_index(
        self,
        chunks_by_file: Dict[str, Tuple[str, str, List[Document]]],
    ) -> bool:
        """
        Build or rebuild FAISS index from a map of filename -> (doc_id, file_hash, chunks).
        Uses atomic file replacement to prevent index corruption.
        """
        all_chunks: List[Document] = []
        new_manifest_files: Dict[str, Dict[str, Any]] = {}

        for filename, (doc_id, file_hash, chunks) in chunks_by_file.items():
            if not chunks:
                continue
            all_chunks.extend(chunks)
            new_manifest_files[filename] = {
                "doc_id": doc_id,
                "file_hash": file_hash,
                "chunk_count": len(chunks),
            }

        if not all_chunks:
            logger.warning("No chunks to index.")
            self.clear_index()
            return False

        embeddings = self.get_embeddings()

        # Build in temporary directory for atomic update
        temp_dir = VECTOR_STORE_DIR / "temp_index"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            temp_vector_store = FAISS.from_documents(all_chunks, embeddings)
            temp_vector_store.save_local(folder_path=str(temp_dir), index_name=INDEX_NAME)

            # Move files from temp_dir to VECTOR_STORE_DIR
            for fname in [f"{INDEX_NAME}.faiss", f"{INDEX_NAME}.pkl"]:
                src = temp_dir / fname
                dst = VECTOR_STORE_DIR / fname
                if src.exists():
                    shutil.move(str(src), str(dst))

            shutil.rmtree(temp_dir, ignore_errors=True)

            self.vector_store = temp_vector_store
            self.state.embedding_model = self.embedding_model_name
            self.state.indexed_files = new_manifest_files
            self.state.total_chunks = len(all_chunks)
            self.save_manifest()

            logger.info(f"Indexed {len(all_chunks)} chunks across {len(new_manifest_files)} file(s).")
            return True

        except Exception as e:
            logger.error(f"Failed to build vector index: {e}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def clear_index(self) -> None:
        """Clear the vector store and manifest."""
        self.vector_store = None
        self.state = KnowledgeBaseState(embedding_model=self.embedding_model_name)
        for fname in [f"{INDEX_NAME}.faiss", f"{INDEX_NAME}.pkl", MANIFEST_FILENAME]:
            p = VECTOR_STORE_DIR / fname
            if p.exists():
                p.unlink()
        logger.info("Cleared FAISS vector store and manifest.")

    def is_ready(self) -> bool:
        """Check if the knowledge base has an initialized vector store with indexed items."""
        return self.vector_store is not None and self.state.total_chunks > 0

    def query(
        self,
        question: str,
        k: int = DEFAULT_RETRIEVAL_K,
    ) -> Dict[str, Any]:
        """
        Execute RAG query pipeline:
        1. Validate readiness.
        2. Retrieve top-k relevant chunks.
        3. Formulate strict grounded prompt.
        4. Generate answer with OpenAI chat model.
        5. Return answer and structured citations.
        """
        cleaned_question = question.strip()
        if not cleaned_question:
            return {
                "answer": "Please ask a question.",
                "sources": [],
                "retrieved_count": 0,
            }

        if not self.is_ready():
            return {
                "answer": "The knowledge base is not ready yet. Please upload and process company documents first.",
                "sources": [],
                "retrieved_count": 0,
            }

        # Retrieve relevant chunks
        try:
            docs_with_scores = self.vector_store.similarity_search_with_score(
                cleaned_question,
                k=k,
            )
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return {
                "answer": f"Error during retrieval: {str(e)}",
                "sources": [],
                "retrieved_count": 0,
            }

        if not docs_with_scores:
            return {
                "answer": FALLBACK_ANSWER,
                "sources": [],
                "retrieved_count": 0,
            }

        # Build context string and source references
        context_parts: List[str] = []
        structured_sources: List[Dict[str, Any]] = []

        for i, (doc, score) in enumerate(docs_with_scores):
            meta = doc.metadata
            filename = meta.get("filename", meta.get("source", "Unknown"))
            sheet = meta.get("sheet_name", "N/A")
            row = meta.get("row_index", "N/A")
            chunk_id = meta.get("chunk_id", f"c{i}")
            content = doc.page_content.strip()

            header = f"[Source {i+1}: {filename}"
            if sheet != "N/A":
                header += f", Sheet: {sheet}"
            if row != "N/A":
                header += f", Row: {row}"
            header += "]"

            context_parts.append(f"{header}\n{content}")

            structured_sources.append({
                "source": filename,
                "sheet_name": sheet,
                "row_index": row,
                "chunk_id": chunk_id,
                "score": float(score) if score is not None else None,
                "excerpt": content[:250] + ("..." if len(content) > 250 else ""),
            })

        full_context = "\n\n---\n\n".join(context_parts)

        # Build Prompt with System & Human messages
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Context from company documents:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            ),
        ])

        try:
            chat_model = self.get_chat_model()
            messages = prompt_template.format_messages(
                context=full_context,
                question=cleaned_question,
            )
            response = chat_model.invoke(messages)
            content_obj = getattr(response, "content", response)
            if isinstance(content_obj, list):
                raw_answer = "".join(
                    part if isinstance(part, str) else (part.get("text", str(part)) if isinstance(part, dict) else str(part))
                    for part in content_obj
                ).strip()
            else:
                raw_answer = str(content_obj).strip()

            # Verify if the answer is the fallback
            if FALLBACK_ANSWER.lower() in raw_answer.lower() and len(raw_answer) < 30:
                answer = FALLBACK_ANSWER
            else:
                answer = raw_answer

            return {
                "answer": answer,
                "sources": structured_sources if answer != FALLBACK_ANSWER else [],
                "retrieved_count": len(docs_with_scores),
            }

        except Exception as e:
            logger.error(f"OpenAI Chat API error: {e}")
            raise
