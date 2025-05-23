# Chat with PDF

This project implements a Retrieval-Augmented Generation (RAG) chatbot with OpenSearch for both text and vector storage to process and chat with your PDF.

## Features

- PDF upload and processing
- Semantic chunking of documents
- Dual storage in OpenSearch (full text + vector embeddings)
- Hybrid search combining semantic and keyword-based approaches
- RAG pipeline with LangChain integration
- Interactive chat interface

## Architecture

The system consists of:

1. **Backend (FastAPI)**:
   - PDF uploading and processing
   - Semantic chunking
   - OpenSearch integration
   - RAG pipeline

2. **Frontend (Streamlit)**:
   - User interface for PDF upload
   - Chat interface
   - Search functionality

## Requirements

- Python 3.10+
- OpenSearch instance (local or remote)
- (Optional) OpenAI API key for better LLM responses

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables (optional):

```bash
# OpenSearch configuration
export OPENSEARCH_HOST=localhost
export OPENSEARCH_PORT=9200
export OPENSEARCH_USER=admin
export OPENSEARCH_PASSWORD=admin
export OPENSEARCH_USE_SSL=false

# OpenAI API key (optional)
export OPENAI_API_KEY=your_api_key
```

## Running the application

1. Start the backend server:

```bash
cd backend
uvicorn main:app --reload
```

2. Start the frontend:

```bash
cd frontend
streamlit run app.py
```

3. Open your browser and navigate to:
   - Backend API: http://localhost:8000
   - Frontend: http://localhost:8501

## Usage

1. Upload PDF documents using the sidebar
2. Ask questions about your documents in the chat interface
3. Explore chunks directly using the search functionality
4. Switch between search types (hybrid, semantic, keyword) to compare results

## Development

- **Backend**: The FastAPI server is defined in `backend/main.py` with modular components for PDF processing, OpenSearch integration, and the RAG pipeline.
- **Frontend**: The Streamlit app is defined in `frontend/app.py`.
