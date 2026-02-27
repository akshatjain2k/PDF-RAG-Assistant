import streamlit as st
import os
import uuid
import json

from config import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
    UPLOAD_DIR,
    SESSION_DIR
)

from core.pdf_ingestion import pdf_pipeline
from core.rag_pipeline import rag_pipeline


# =====================================================
# INITIAL SETUP
# =====================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    layout="wide"
)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SESSION_DIR, exist_ok=True)


# =====================================================
# SESSION INIT
# =====================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# =====================================================
# HEADER
# =====================================================

st.title("📚 PDF RAG Assistant")
st.caption("Upload PDFs and chat with your documents")
st.caption(f"🆔 Session ID: `{st.session_state.session_id}`")

st.divider()


# =====================================================
# PDF UPLOAD
# =====================================================

st.subheader("📤 Upload PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF file (Max 10MB)",
    type=["pdf"]
)

if st.button("Upload & Index", use_container_width=True):

    if uploaded_file is None:
        st.warning("Please upload a PDF first.")
    else:

        if uploaded_file.size > 10 * 1024 * 1024:
            st.error("File exceeds 10MB limit.")
            st.stop()

        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

        print("📥 Upload detected:", uploaded_file.name)

        with st.spinner("Processing and indexing..."):

            # Save file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            print("💾 File saved:", file_path)

            # Ingest
            chunk_count = pdf_pipeline(file_path)

            print("🧠 Chunks added:", chunk_count)

        st.success(f"{uploaded_file.name} indexed successfully")
        st.info(f"Chunks Added: {chunk_count}")

st.divider()


# =====================================================
# CHAT
# =====================================================

st.subheader("💬 Ask a Question")

query = st.chat_input("Type your question and press Enter...")

if query:

    # Add user message to UI
    st.session_state.chat_history.append({
        "role": "user",
        "content": query
    })

    # Guard: Check index existence
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):

        response_text = "No documents indexed. Please upload a PDF first."
        citations = []

    else:

        with st.spinner("Thinking..."):

            result = rag_pipeline(
                user_query=query,
                session_id=st.session_state.session_id,
                top_k=3
            )

        response_text = result["answer"]
        citations = result.get("citations", [])

    # Add assistant message to UI
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response_text,
        "citations": citations
    })

    # ==========================================
    # SAVE STRUCTURED SESSION FORMAT (CLEAN)
    # ==========================================

    session_file = os.path.join(
        SESSION_DIR,
        f"{st.session_state.session_id}.json"
    )

    # Load old records if exist
    if os.path.exists(session_file):
        with open(session_file, "r", encoding="utf-8") as f:
            session_records = json.load(f)
    else:
        session_records = []

    # Append new structured record
    session_records.append({
        "query": query,
        "response": response_text,
        "citations": citations
    })

    # Save updated session
    with open(session_file, "w", encoding="utf-8") as f:
        json.dump(
            session_records,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("💾 Structured session saved:", session_file)


# =====================================================
# DISPLAY CHAT
# =====================================================

for msg_index, msg in enumerate(st.session_state.chat_history):

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        # Display citations for assistant messages
        if msg["role"] == "assistant" and msg.get("citations"):

            if msg["citations"]:

                st.markdown("### 📖 Sources")

                for cite_index, cite in enumerate(msg["citations"]):

                    with st.expander(
                        f"{cite['file']} | Page {cite['page']} | Chunk {cite['chunk_no']}"
                    ):

                        content = cite["content"]

                        preview_len = 300
                        preview = (
                            content[:preview_len] + "..."
                            if len(content) > preview_len
                            else content
                        )

                        st.markdown(preview)

                        checkbox_key = (
                            f"{st.session_state.session_id}_{msg_index}_{cite_index}"
                        )

                        if st.checkbox("Show full content", key=checkbox_key):
                            st.markdown(content)

            else:
                st.info("No sources found.")