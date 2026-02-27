# PDF RAG Assistant

A lightweight Streamlit web application that allows you to upload PDF documents and interact with them using Retrieval-Augmented Generation (RAG). Powered by Google Gemini and FAISS.

## Features
- **PDF Uploads**: Upload and index your PDF files (up to 10MB).
- **AI-Powered QA**: Get accurate answers to your questions based solely on the uploaded document using the `gemini-2.5-flash` model.
- **Source Citations**: Every answer includes expandable citations, showing exactly which page and chunk the information came from.
- **Fast Vector Local Search**: Uses FAISS CPU for fast, local document retrieval and `gemini-embedding-001` for embeddings.
- **Session History**: Automatically saves chat sessions, including queries, responses, and citations in a structured JSON format.

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Environment Variables**:
   Create a `.env` file in the root directory and configure your Gemini API Key:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## Usage

1. **Start the application**:
   ```bash
   streamlit run app.py
   ```
2. **Interact**: 
   - Open your browser to the provided local URL (typically `http://localhost:8501`).
   - Upload a PDF file.
   - Start chatting to ask questions about the document!

## Project Structure
- `app.py`: Main Streamlit application and UI.
- `config.py`: Configuration and environment variables.
- `core/`: Core RAG pipelines containing PDF ingestion, context retrieval, and model generation logic.
- `data/`: Local storage for the FAISS index and metadata.
- `session/`: Saved chat session histories.
- `uploads/`: Temporary storage for uploaded PDF files.
