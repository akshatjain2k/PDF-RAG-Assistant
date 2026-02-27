import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"

CHROMA_DB_PATH = "data/chroma_db"
UPLOAD_DIR = "uploads"
COLLECTION_NAME = "pdf_documents"
SESSION_DIR = "session"
# ==============================
# FAISS PATHS
# ==============================

FAISS_INDEX_PATH =  "data/faiss.index"
METADATA_PATH =  "data/metadata.pkl"