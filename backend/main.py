"""FastAPI application entry point for the multi-agent conversational flow builder."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.kb.sqlite_db import init_db
from backend.kb.chroma_client import init_chroma
from backend.api.routes_run import router as run_router
from backend.api.routes_approve import router as approve_router
from backend.api.routes_kb import router as kb_router
from backend.api.routes_upload import router as upload_router
from backend.api.routes_master import router as master_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DB and ChromaDB on startup."""
    await init_db()
    init_chroma()
    yield


app = FastAPI(
    title="Multi-Agent Conversational Flow Builder",
    description="RAG-augmented prompt generation with human-in-the-loop approval",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(run_router, prefix="/api", tags=["Pipeline"])
app.include_router(approve_router, prefix="/api", tags=["Approval"])
app.include_router(kb_router, prefix="/api", tags=["Knowledge Base"])
app.include_router(upload_router, prefix="/api", tags=["Upload & Extract"])
app.include_router(master_router, prefix="/api", tags=["Master System"])


@app.get("/health")
async def health_check():
    return {"status": "ok"}
