import fitz
import os
import uuid
import pickle
import faiss
import numpy as np

from google import genai

from config import GEMINI_API_KEY, EMBEDDING_MODEL


# =====================================================
# CONFIG
# =====================================================

DATA_DIR = "data"

FAISS_INDEX_PATH = f"{DATA_DIR}/faiss.index"
METADATA_PATH = f"{DATA_DIR}/metadata.pkl"

os.makedirs(DATA_DIR, exist_ok=True)


# =====================================================
# GEMINI CLIENT
# =====================================================

print("🔐 Initializing Gemini client...")

client = genai.Client(api_key=GEMINI_API_KEY)

print("✅ Gemini client ready\n")


# =====================================================
# LOAD / CREATE FAISS
# =====================================================

def load_or_create_faiss(dim: int):

    if os.path.exists(FAISS_INDEX_PATH):

        print("📂 Loading existing FAISS index...")

        index = faiss.read_index(FAISS_INDEX_PATH)

        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)

        print(f"✅ Loaded {index.ntotal} existing vectors\n")

    else:

        print("📦 Creating new FAISS index...")

        index = faiss.IndexFlatL2(dim)

        metadata = []

        print("✅ New index created\n")

    return index, metadata


# =====================================================
# 1. EXTRACT WITH PAGE NO
# =====================================================

def extract_text_from_pdf(pdf_path: str):

    print("📄 Extracting text...")

    doc = fitz.open(pdf_path)

    pages = []

    for page_no, page in enumerate(doc, start=1):

        text = page.get_text().strip()

        print(f"   ✔ Page {page_no} → {len(text)} chars")

        if text:
            pages.append((page_no, text))

    print(f"✅ {len(pages)} pages extracted\n")

    return pages


# =====================================================
# 2. OVERLAP
# =====================================================

def overlap_pages(pages, overlap_ratio=0.2):

    print("🔁 Applying overlap...")

    chunks = []

    for i in range(len(pages)):

        page_no, current = pages[i]

        if i == 0:
            chunks.append((page_no, current))
            print("   ✔ Chunk 1 | No overlap")
            continue

        prev_page, prev_text = pages[i - 1]

        overlap_len = int(len(prev_text) * overlap_ratio)

        overlap = prev_text[-overlap_len:]

        combined = overlap + "\n" + current

        chunks.append((page_no, combined))

        print(f"   ✔ Chunk {i+1} | Page {page_no}")

    print(f"✅ {len(chunks)} chunks created\n")

    return chunks


# =====================================================
# 3. EMBEDDING
# =====================================================

def generate_embedding(text: str):

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text
    )

    return response.embeddings[0].values


# =====================================================
# 4. INGEST (APPEND MODE)
# =====================================================

def ingest_chunks(chunks, file_name):

    print("💾 Ingesting (append mode)...")

    # Embed first chunk to get dimension
    first_vector = generate_embedding(chunks[0][1])

    dim = len(first_vector)

    # Load or create FAISS
    index, metadata = load_or_create_faiss(dim)

    vectors = [first_vector]
    records = []

    # First chunk
    page, text = chunks[0]

    records.append({
        "id": str(uuid.uuid4()),
        "file": file_name,
        "page": page,
        "chunk_no": len(metadata) + 1,
        "content": text
    })

    print("   ✔ Embedded chunk 1")

    # Remaining chunks
    for i, (page_no, chunk_text) in enumerate(chunks[1:], start=2):

        embedding = generate_embedding(chunk_text)

        vectors.append(embedding)

        record = {
            "id": str(uuid.uuid4()),
            "file": file_name,
            "page": page_no,
            "chunk_no": len(metadata) + i,
            "content": chunk_text
        }

        records.append(record)

        print(f"   ✔ Embedded chunk {i}")

    # Convert to numpy
    vectors = np.array(vectors).astype("float32")

    # Append to FAISS
    index.add(vectors)

    # Append metadata
    metadata.extend(records)

    # Save
    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print(f"📊 Total vectors now: {index.ntotal}")
    print("✅ Ingestion completed\n")

    return len(records)


# =====================================================
# 5. MAIN PIPELINE
# =====================================================

def pdf_pipeline(pdf_path: str):

    print("\n=================================================")
    print("🚀 PIPELINE STARTED")
    print(f"📂 {pdf_path}")
    print("=================================================\n")

    file_name = os.path.basename(pdf_path)

    pages = extract_text_from_pdf(pdf_path)

    chunks = overlap_pages(pages)

    count = ingest_chunks(chunks, file_name)

    print("=================================================")
    print(f"🎉 COMPLETED | {count} new chunks added")
    print("=================================================\n")

    return count


# # =====================================================
# # TEST
# # =====================================================

# if __name__ == "__main__":

#     pdf = "human_eye.pdf"

#     if not os.path.exists(pdf):
#         print("❌ PDF not found")
#     else:
#         pdf_pipeline(pdf)