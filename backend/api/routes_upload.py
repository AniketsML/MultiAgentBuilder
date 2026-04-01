"""API routes for file upload and state extraction."""

import traceback
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.utils.doc_parser import extract_text
from backend.agents.agent0_extractor import extract_states

router = APIRouter()
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".txt", ".docx", ".pdf", ".doc", ".md"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document and extract its text content."""
    import os

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Max 10 MB.")

    try:
        text = extract_text(file_bytes, file.filename or "document.txt")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {e}")

    return {"filename": file.filename, "text": text, "char_count": len(text)}


@router.post("/extract-states")
async def extract_states_endpoint(body: dict):
    """Use Agent 0 to auto-extract conversational states from context text."""
    context_doc = body.get("context_doc", "")
    if not context_doc or len(context_doc.strip()) < 20:
        raise HTTPException(status_code=400, detail="Document too short (min 20 chars).")

    try:
        print(f"[Agent 0] Starting state extraction ({len(context_doc)} chars)...")
        result = await extract_states(context_doc)
        print(f"[Agent 0] Success - {len(result.get('states', []))} states extracted")
        return result
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[Agent 0] FAILED:\n{tb}")
        logger.error(f"State extraction failed:\n{tb}")
        raise HTTPException(status_code=500, detail=f"State extraction failed: {str(e)}")
