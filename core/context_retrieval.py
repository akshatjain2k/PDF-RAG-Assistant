# import json
# import os
# import pickle
# import faiss
# import numpy as np

# from google import genai

# from config import (
#     GEMINI_API_KEY,
#     EMBEDDING_MODEL
# )


# # =====================================================
# # CONFIG
# # =====================================================

# DATA_DIR = "data"

# FAISS_INDEX_PATH = f"{DATA_DIR}/faiss.index"
# METADATA_PATH = f"{DATA_DIR}/metadata.pkl"


# # =====================================================
# # GEMINI CLIENT
# # =====================================================

# print("🔐 Initializing Gemini client...")

# client = genai.Client(api_key=GEMINI_API_KEY)

# print("✅ Gemini client ready\n")


# # =====================================================
# # QUERY EMBEDDING
# # =====================================================

# def generate_query_embedding(query: str):

#     print("🧠 Generating query embedding...")

#     response = client.models.embed_content(
#         model=EMBEDDING_MODEL,
#         contents=query
#     )

#     vector = response.embeddings[0].values

#     print(f"   ✔ Query embedded | Dim: {len(vector)}\n")

#     return vector


# # =====================================================
# # LOAD INDEX + METADATA
# # =====================================================

# def load_faiss():

#     if not os.path.exists(FAISS_INDEX_PATH):
#         print("⚠ FAISS index not found.")
#         return None, None

#     if not os.path.exists(METADATA_PATH):
#         print("⚠ Metadata file not found.")
#         return None, None

#     index = faiss.read_index(FAISS_INDEX_PATH)

#     with open(METADATA_PATH, "r", encoding="utf-8") as f:
#         metadata = json.load(f)

#     return index, metadata


# # =====================================================
# # RETRIEVE CONTEXT
# # =====================================================

# def retrieve_context(user_query: str, top_k: int = 3):

#     print("🔍 Retrieving relevant context...\n")

#     index, metadata = load_faiss()

#     # If index not available
#     if index is None or metadata is None:
#         return []

#     query_vec = generate_query_embedding(user_query)
#     query_vec = np.array([query_vec]).astype("float32")

#     distances, indices = index.search(query_vec, top_k)

#     results = []

#     print("📌 Top Matches:\n")

#     for rank, idx in enumerate(indices[0], start=1):

#         record = metadata[idx]

#         score = float(distances[0][rank - 1])

#         result = {
#             "rank": rank,
#             "score": score,
#             "file": record["file"],
#             "page": record["page"],
#             "chunk_no": record["chunk_no"],
#             "content": record["content"]
#         }

#         results.append(result)

#         print(
#             f"#{rank} | "
#             f"File: {record['file']} | "
#             f"Page: {record['page']} | "
#             f"Chunk: {record['chunk_no']} | "
#             f"Score: {round(score,4)}"
#         )

#     print("\n✅ Retrieval completed\n")

#     return results


# # # =====================================================
# # # TEST
# # # =====================================================

# # if __name__ == "__main__":

# #     query = "Explain the structure of human eye"

# #     results = retrieve_context(query, top_k=3)

# #     for r in results:

# #         print("\n----------------------------")
# #         print(f"Rank: {r['rank']}")
# #         print(f"File: {r['file']}")
# #         print(f"Page: {r['page']}")
# #         print(f"Chunk: {r['chunk_no']}")
# #         print(f"Score: {r['score']}")
# #         print("\nText Preview:\n")
# #         print(r["content"])