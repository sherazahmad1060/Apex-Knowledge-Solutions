"""
Document Processor Module.
Handles loading, validation, metadata extraction, and chunking for TXT and XLSX files.
"""

import os
import hashlib
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    MAX_FILE_SIZE_MB,
    SUPPORTED_EXTENSIONS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
    MIN_CHUNK_OVERLAP,
    MAX_CHUNK_OVERLAP,
    UPLOAD_DIR,
)

logger = logging.getLogger("company_chatbot.document_processor")


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file to detect content changes and duplicates."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_bytes_hash(content: bytes) -> str:
    """Compute SHA-256 hash of raw bytes."""
    return hashlib.sha256(content).hexdigest()


def validate_file(file_name: str, file_size_bytes: int) -> tuple[bool, str]:
    """
    Validate file extension and file size.
    Returns (is_valid, error_message).
    """
    ext = Path(file_name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return False, f"Unsupported file type: '{ext}'. Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}."

    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size_bytes > max_bytes:
        return False, f"File size exceeds maximum limit of {MAX_FILE_SIZE_MB}MB."

    if file_size_bytes == 0:
        return False, "File is empty (0 bytes)."

    return True, ""


def load_txt_file(file_path: Path, doc_id: Optional[str] = None) -> List[Document]:
    """
    Read text file handling UTF-8, UTF-8-BOM, and fallback encodings gracefully.
    Returns a list of Document objects.
    """
    filename = file_path.name
    doc_id = doc_id or str(uuid.uuid4())[:8]

    # Try common encodings in order: utf-8-sig (handles BOM and utf-8), utf-8, latin-1
    content = ""
    for encoding in ["utf-8-sig", "utf-8", "latin-1", "cp1252"]:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue
    else:
        # Ultimate fallback with replace error handler
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

    cleaned = content.lstrip("\ufeff").strip()
    if not cleaned:
        logger.warning(f"TXT file '{filename}' contains no readable text content.")
        return []

    lines = cleaned.splitlines()
    doc = Document(
        page_content=cleaned,
        metadata={
            "source": filename,
            "filename": filename,
            "doc_id": doc_id,
            "file_type": "txt",
            "file_size": file_path.stat().st_size if file_path.exists() else len(cleaned),
            "line_count": len(lines),
            "sheet_name": "N/A",
            "row_index": "N/A",
        },
    )
    return [doc]


def load_xlsx_file(file_path: Path, doc_id: Optional[str] = None) -> List[Document]:
    """
    Read an Excel workbook across all worksheets.
    Converts each non-empty row into meaningful key-value textual representation,
    preserving column names, sheet names, row numbers, and source metadata.
    """
    filename = file_path.name
    doc_id = doc_id or str(uuid.uuid4())[:8]
    documents: List[Document] = []

    try:
        with pd.ExcelFile(file_path, engine="openpyxl") as excel_file:
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                except Exception as e:
                    logger.warning(f"Could not read sheet '{sheet_name}' in '{filename}': {e}")
                    continue

                if df.empty or len(df.columns) == 0:
                    logger.info(f"Skipping empty sheet '{sheet_name}' in '{filename}'.")
                    continue

                # Clean column headers
                df.columns = [str(c).strip() for c in df.columns]

                # Iterate over rows
                for idx, row in df.iterrows():
                    row_num = idx + 2  # Excel row number (1-based, row 1 is header)
                    fields: List[str] = []

                    for col in df.columns:
                        val = row[col]
                        # Skip null / NaN / empty values
                        if pd.isna(val) or val is None:
                            continue
                        val_str = str(val).strip()
                        if val_str:
                            fields.append(f"{col}: {val_str}")

                    if not fields:
                        # Skip empty row
                        continue

                    row_text = f"Sheet: {sheet_name} | Row {row_num} -> " + " | ".join(fields)
                    doc = Document(
                        page_content=row_text,
                        metadata={
                            "source": filename,
                            "filename": filename,
                            "doc_id": doc_id,
                            "file_type": "xlsx",
                            "sheet_name": sheet_name,
                            "row_index": row_num,
                        },
                    )
                    documents.append(doc)
    except Exception as e:
        logger.error(f"Failed to open Excel file '{filename}': {e}")
        raise ValueError(f"Corrupted or invalid Excel workbook '{filename}': {e}")

    if not documents:
        logger.warning(f"No valid rows extracted from Excel file '{filename}'.")

    return documents


def load_document(file_path: Path, doc_id: Optional[str] = None) -> List[Document]:
    """
    Dispatch loading based on file extension.
    """
    ext = file_path.suffix.lower()
    if ext == ".txt":
        return load_txt_file(file_path, doc_id=doc_id)
    elif ext == ".xlsx":
        return load_xlsx_file(file_path, doc_id=doc_id)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def split_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split documents into smaller chunks using RecursiveCharacterTextSplitter.
    Preserves all document metadata and appends chunk_id.
    """
    # Validate settings
    if chunk_size < MIN_CHUNK_SIZE or chunk_size > MAX_CHUNK_SIZE:
        raise ValueError(f"Chunk size must be between {MIN_CHUNK_SIZE} and {MAX_CHUNK_SIZE}.")
    if chunk_overlap < MIN_CHUNK_OVERLAP or chunk_overlap > MAX_CHUNK_OVERLAP:
        raise ValueError(f"Chunk overlap must be between {MIN_CHUNK_OVERLAP} and {MAX_CHUNK_OVERLAP}.")
    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " | ", ". ", " ", ""],
    )

    chunks: List[Document] = []
    chunk_counter = 0

    for doc in documents:
        doc_chunks = splitter.split_documents([doc])
        for chunk in doc_chunks:
            chunk.metadata = dict(chunk.metadata)  # Shallow copy
            chunk.metadata["chunk_id"] = f"{chunk.metadata.get('doc_id', 'doc')}-c{chunk_counter}"
            chunk_counter += 1
            chunks.append(chunk)

    logger.info(f"Split {len(documents)} document item(s) into {len(chunks)} chunk(s).")
    return chunks


def save_uploaded_file(file_bytes: bytes, original_filename: str) -> tuple[Path, str, str]:
    """
    Save an uploaded file safely to the upload directory.
    Avoids overwriting by checking hash or adding unique suffix if names collide.
    Returns (saved_path, doc_id, file_hash).
    """
    # Sanitize filename (remove path traversal)
    clean_name = Path(original_filename).name
    file_hash = compute_bytes_hash(file_bytes)
    doc_id = file_hash[:10]

    # Target path
    target_path = UPLOAD_DIR / clean_name
    if target_path.exists():
        existing_hash = compute_file_hash(target_path)
        if existing_hash == file_hash:
            # Identical file already stored
            return target_path, doc_id, file_hash
        else:
            # Different file with same name -> append short hash
            name_stem = target_path.stem
            name_ext = target_path.suffix
            clean_name = f"{name_stem}_{doc_id[:6]}{name_ext}"
            target_path = UPLOAD_DIR / clean_name

    with open(target_path, "wb") as f:
        f.write(file_bytes)

    logger.info(f"Saved uploaded file: {clean_name} ({len(file_bytes)} bytes)")
    return target_path, doc_id, file_hash


def remove_document_file(filename: str) -> bool:
    """
    Remove an uploaded document from the uploads directory safely.
    Returns True if removed, False otherwise.
    """
    clean_name = Path(filename).name
    target_path = UPLOAD_DIR / clean_name
    if target_path.exists() and target_path.is_file():
        try:
            target_path.unlink()
            logger.info(f"Removed uploaded file: {clean_name}")
            return True
        except Exception as e:
            logger.error(f"Error removing file {clean_name}: {e}")
            return False
    return False
