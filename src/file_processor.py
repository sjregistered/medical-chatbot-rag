"""
File Processor Module – Medical Chatbot
Validates and parses user uploads so the chatbot can "read" them.

Supported upload kinds (chosen by the user from a dropdown, or auto-detected):
- image : photos, X-rays, prescriptions, medicine labels  (PNG/JPG/WEBP/HEIC/GIF)
- pdf   : lab reports, discharge summaries, prescriptions (PDF)
- data  : tabular data such as blood-test values or vitals logs (CSV/XLSX/XLS)
- text  : plain-text notes or reports (TXT/MD)

Every upload is normalized into an `Attachment` dict:
    {
        "kind":     "image" | "pdf" | "data" | "text",
        "filename": str,
        "mime":     str,
        "bytes":    raw bytes (sent to the multimodal LLM for image/pdf),
        "text":     extracted text / data summary (used for RAG + local fallback),
        "preview":  short human-readable summary for the UI,
    }
"""

import io
import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_TEXT_CHARS = 30_000              # cap on extracted text passed to the LLM

KIND_EXTENSIONS = {
    "image": {".png", ".jpg", ".jpeg", ".webp", ".gif", ".heic", ".heif"},
    "pdf": {".pdf"},
    "data": {".csv", ".xlsx", ".xls"},
    "text": {".txt", ".md"},
}

MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".pdf": "application/pdf",
    ".csv": "text/csv",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".txt": "text/plain",
    ".md": "text/markdown",
}


class UploadError(ValueError):
    """Raised when an upload is invalid; the message is safe to show users."""


def detect_kind(filename: str) -> Optional[str]:
    """Return the upload kind for a filename based on its extension."""
    ext = os.path.splitext(filename.lower())[1]
    for kind, exts in KIND_EXTENSIONS.items():
        if ext in exts:
            return kind
    return None


def process_upload(filename: str, data: bytes, kind: str = "auto") -> Dict:
    """
    Validate and parse an uploaded file.

    Args:
        filename: Original filename (used for extension checks).
        data: Raw file bytes.
        kind: Kind selected by the user ("auto", "image", "pdf", "data", "text").

    Returns:
        Attachment dict (see module docstring).

    Raises:
        UploadError: If the file is empty, too large, or doesn't match the kind.
    """
    filename = os.path.basename(filename or "upload")
    if not data:
        raise UploadError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError("File is too large. Maximum size is 10 MB.")

    detected = detect_kind(filename)
    if detected is None:
        allowed = ", ".join(sorted(e for exts in KIND_EXTENSIONS.values() for e in exts))
        raise UploadError(f"Unsupported file type. Allowed: {allowed}")

    if kind in (None, "", "auto"):
        kind = detected
    elif kind not in KIND_EXTENSIONS:
        raise UploadError(f"Unknown upload type '{kind}'.")
    elif kind != detected:
        raise UploadError(
            f"'{filename}' looks like a {detected} file, but '{kind}' was selected. "
            f"Pick the matching type in the dropdown (or choose Auto-detect)."
        )

    ext = os.path.splitext(filename.lower())[1]
    mime = MIME_BY_EXT.get(ext, "application/octet-stream")
    handler = {
        "image": _process_image,
        "pdf": _process_pdf,
        "data": _process_data,
        "text": _process_text,
    }[kind]

    text, preview = handler(data, ext)
    logger.info(f"Processed upload '{filename}' as {kind} ({len(data)} bytes)")
    return {
        "kind": kind,
        "filename": filename,
        "mime": mime,
        "bytes": data,
        "text": (text or "")[:MAX_TEXT_CHARS],
        "preview": preview,
    }


# ─── Per-kind handlers: each returns (extracted_text, preview) ───

def _process_image(data: bytes, ext: str):
    """Verify the image is readable and describe its dimensions."""
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as img:
            img.verify()  # cheap integrity check
        with Image.open(io.BytesIO(data)) as img:
            preview = f"Image {img.width}×{img.height} ({img.format or ext.lstrip('.').upper()})"
    except ImportError:
        preview = "Image"
    except Exception:
        # HEIC etc. may not be decodable by Pillow; the LLM can still read them.
        if ext in (".heic", ".heif"):
            preview = "Image (HEIC)"
        else:
            raise UploadError("The image file appears to be corrupted or unreadable.")
    # Images carry no extractable text locally; the vision model reads them.
    return "", preview


def _process_pdf(data: bytes, ext: str):
    """Extract text from a PDF (scanned PDFs are still read by the vision model)."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return "", "PDF document"

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise UploadError("This PDF is password-protected. Please upload an unlocked copy.")
        pages = []
        for i, page in enumerate(reader.pages[:30]):  # cap very long docs
            page_text = (page.extract_text() or "").strip()
            if page_text:
                pages.append(f"--- Page {i + 1} ---\n{page_text}")
        text = "\n\n".join(pages)
        n = len(reader.pages)
        preview = f"PDF, {n} page{'s' if n != 1 else ''}"
        if not text:
            preview += " (scanned – no text layer)"
        return text, preview
    except UploadError:
        raise
    except Exception as e:
        logger.warning(f"PDF parse failed: {e}")
        raise UploadError("Could not read this PDF. It may be corrupted.")


def _process_data(data: bytes, ext: str):
    """Parse CSV/Excel and build a compact statistical summary for the LLM."""
    try:
        import pandas as pd
    except ImportError:
        raise UploadError("Data file support requires pandas (pip install pandas openpyxl).")

    try:
        if ext == ".csv":
            df = pd.read_csv(io.BytesIO(data), sep=None, engine="python")  # sniff delimiter
        else:
            df = pd.read_excel(io.BytesIO(data))
    except Exception as e:
        logger.warning(f"Data parse failed: {e}")
        raise UploadError("Could not parse this data file. Check that it's a valid CSV/Excel file.")

    if df.empty:
        raise UploadError("The data file has no rows.")

    df = df.dropna(how="all").dropna(axis=1, how="all")
    rows, cols = df.shape

    parts = [
        f"Dataset: {rows} rows × {cols} columns",
        "Columns and types: " + ", ".join(f"{c} ({df[c].dtype})" for c in df.columns),
    ]

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        stats = numeric.describe().T[["count", "mean", "min", "max", "std"]].round(2)
        parts.append("Numeric column statistics:\n" + stats.to_string())
        # First → last change is useful for vitals / repeated lab logs.
        if rows >= 2:
            trend_lines = []
            for c in numeric.columns:
                series = numeric[c].dropna()
                if len(series) >= 2:
                    first, last = series.iloc[0], series.iloc[-1]
                    trend_lines.append(f"{c}: {first:g} → {last:g} (change {last - first:+.2f})")
            if trend_lines:
                parts.append("First-to-last trend:\n" + "\n".join(trend_lines))

    # Include the raw rows (all of them if small, otherwise head + tail).
    if rows <= 60:
        parts.append("All rows:\n" + df.to_string(index=False))
    else:
        parts.append("First 30 rows:\n" + df.head(30).to_string(index=False))
        parts.append("Last 10 rows:\n" + df.tail(10).to_string(index=False))

    preview = f"{ext.lstrip('.').upper()} data, {rows} rows × {cols} columns"
    return "\n\n".join(parts), preview


def _process_text(data: bytes, ext: str):
    """Decode a plain-text upload."""
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UploadError("Could not decode this text file.")
    text = text.strip()
    if not text:
        raise UploadError("The text file is empty.")
    words = len(text.split())
    return text, f"Text document, {words} words"
