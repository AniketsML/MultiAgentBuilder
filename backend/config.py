"""Configuration loader for the multi-agent pipeline."""

import os
from dotenv import load_dotenv

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "nvidia/llama-3.1-nemotron-ultra-253b-v1")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./kb_data/chroma")
SQLITE_PATH = os.getenv("SQLITE_PATH", "./kb_data/metadata.db")
