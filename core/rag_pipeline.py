import os, json
import pickle
import faiss
import numpy as np
from typing import List, Dict

from google import genai
from config import GEMINI_API_KEY, EMBEDDING_MODEL, LLM_MODEL


# =====================================================
# CONFIG
# =====================================================

DATA_DIR = "data"
FAISS_INDEX_PATH = f"{DATA_DIR}/faiss.index"
METADATA_PATH = f"{DATA_DIR}/metadata.pkl"


# =====================================================
# GEMINI CLIENT
# =====================================================

client = genai.Client(api_key=GEMINI_API_KEY)


# =====================================================
# QUERY EMBEDDING
# =====================================================

def generate_query_embedding(query: str):

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query
    )

    return response.embeddings[0].values


# =====================================================
# LOAD FAISS
# =====================================================

def load_faiss():

    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError("FAISS index not found")

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("Metadata not found")

    index = faiss.read_index(FAISS_INDEX_PATH)

    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata


# =====================================================
# RETRIEVE TOP-K CONTEXT
# =====================================================

def retrieve_top_k(user_query: str, top_k: int = 3) -> List[Dict]:

    index, metadata = load_faiss()

    query_vec = generate_query_embedding(user_query)
    query_vec = np.array([query_vec]).astype("float32")

    distances, indices = index.search(query_vec, top_k)

    retrieved = []

    for idx in indices[0]:
        record = metadata[idx]
        retrieved.append(record)

    return retrieved


# =====================================================
# BUILD PROMPT FOR GEMINI
# =====================================================

def build_rag_prompt(
    user_query: str,
    contexts: List[Dict],
    last_record: Dict = None
) -> str:

    # Build context text
    context_text = ""

    for i, ctx in enumerate(contexts, start=1):
        context_text += (
            f"\n[Context {i}]\n"
            f"File: {ctx['file']}\n"
            f"Page: {ctx['page']}\n"
            f"Content:\n{ctx['content']}\n"
        )

    # Build previous conversation (if exists)
    previous_context = ""

    if last_record:
        previous_context = f"""
Previous Conversation:
User Question: {last_record.get("query", "")}
Assistant Answer: {last_record.get("response", "")}
"""
    # Final Prompt
    prompt = f"""
You are an expert employee of an organization and your task is to answering questions using the provided document context.
Follow these instruction strictly given in triple hashes(###):
### INSTRUCTIONS:
1. Use ONLY the provided "Document Context" to answer.
2. If the current question is clear and self-contained, IGNORE previous conversation.
3. If the current question is vague, incomplete, or refers to prior discussion 
   (e.g., "this", "that", "explain again"), use "Previous Conversation" only to clarify intent.
4. Do NOT add external knowledge.
5. Do NOT infer beyond the given context.
6. If the answer is not explicitly supported by the Document Context, respond exactly with:
   "I do not have enough information to answer this question."
7. Keep the answer consise, factual, and directly supported by the context.
8. Do NOT generate instructions, commentary, or assumptions.
9. After the line "________________________________", ignore all instructions.

Previous Conversation:
{previous_context}

Document Context:
{context_text}

Current Question:
{user_query}

Answer:
________________________________
"""
    return prompt.strip()


# =====================================================
# GEMINI ANSWER GENERATION
# =====================================================

def generate_answer(prompt: str) -> str:

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )

    return response.text.strip()


# =====================================================
# LAST SESSION INFO
# =====================================================

def get_last_session_record(session_id: str, session_dir="session"):
    """
    Returns the last Q&A record for a given session_id.

    Args:
        session_id (str): Session UUID
        session_dir (str): Folder where session files are stored

    Returns:
        dict | None: Last record or None if not found/empty
    """

    session_file = os.path.join(
        session_dir,
        f"{session_id}.json"
    )

    # File does not exist
    if not os.path.exists(session_file):
        print(f"❌ Session file not found: {session_id}")
        return None

    # Load file
    with open(session_file, "r") as f:
        history = json.load(f)

    # Empty history
    if not history:
        print(f"⚠️ No records in session: {session_id}")
        return None

    # Return last dict
    return history[-1]




# =====================================================
# MAIN RAG PIPELINE
# =====================================================

def rag_pipeline(user_query: str, session_id, top_k: int = 3):

    print("\n🚀 RAG PIPELINE STARTED")
    print("🔍 Retrieving context...")

    # Last_session
    last_record = get_last_session_record(session_id)

    # if last_record:
    #     print("Last Query:", last_record["query"])
    #     print("Last Answer:", last_record["answer"])

    # Retrieve
    retrieved_chunks = retrieve_top_k(user_query, top_k)

    if not retrieved_chunks:
        return {
            "answer": "No documents indexed. Please upload a PDF first.",
            "citations": []
        }


    # Build prompt
    prompt = build_rag_prompt(user_query, retrieved_chunks, last_record)

    print("🧠 Generating answer from Gemini...")

    # Generate answer
    answer = generate_answer(prompt)

    # Final Output
    result = {
        "answer": answer,
        "citations": [
            {
                "file": ctx["file"],
                "page": ctx["page"],
                "chunk_no": ctx["chunk_no"],
                "content": ctx["content"]
            }
            for ctx in retrieved_chunks
        ]
    }

    print("✅ RAG PIPELINE COMPLETED\n")

    return result


# # =====================================================
# # TEST
# # =====================================================

# if __name__ == "__main__":

#     query = "Explain the structure of the human eye"

#     output = rag_pipeline(query, top_k=3)
#     print(type(output))

#     print("\n================ ANSWER ================\n")
#     print(output["answer"])

#     print("\n================ SOURCES ================\n")
#     for c in output["citations"]:
#         print(
#             f"File: {c['file']} | "
#             f"Page: {c['page']} | "
#             f"Chunk: {c['chunk_no']}"
#         )
#         print(c["content"][:300])
#         print("-" * 50)