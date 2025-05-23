import os
import json
import requests
import streamlit as st
from typing import List, Dict, Any

# API endpoint
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Semantic Chunking RAG System",
    page_icon="📚",
    layout="wide"
)

# App title and description
st.title("📚 Semantic Chunking RAG System")
st.markdown("""
Upload PDF documents and chat with them using semantic chunking and OpenSearch.
""")

# Sidebar with upload functionality
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                # Save the file to a temporary location
                file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
                
                # Send file to API
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post(f"{API_URL}/upload-pdf", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"PDF processed successfully! {result['document_info']['total_chunks']} chunks created.")
                    st.json(result)
                else:
                    st.error(f"Error processing PDF: {response.text}")
    
    # Search type selection
    st.header("Search Settings")
    search_type = st.radio(
        "Search Type",
        ["hybrid", "semantic", "keyword"],
        help="Select the type of search to use when retrieving context"
    )

# Main area for chat
st.header("Chat with your Documents")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Prepare request
            data = {
                "query": prompt,
                "search_type": search_type
            }
            
            # Send request to API
            response = requests.post(f"{API_URL}/chat", json=data)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result["response"]
                st.markdown(ai_response)
                
                # Add AI response to chat history
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
            else:
                error_message = f"Error: {response.text}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# Explore chunks functionality
st.header("Explore Document Chunks")
with st.expander("Search Chunks"):
    chunk_query = st.text_input("Search for chunks")
    chunk_search_button = st.button("Search")
    
    if chunk_search_button and chunk_query:
        with st.spinner("Searching chunks..."):
            params = {
                "query": chunk_query,
                "search_type": search_type
            }
            
            # Get chunks from API
            response = requests.get(f"{API_URL}/chunks", params=params)
            
            if response.status_code == 200:
                chunks = response.json()
                
                if chunks:
                    for i, chunk in enumerate(chunks):
                        with st.expander(f"{chunk['title']}"):
                            st.write(chunk["content"])
                else:
                    st.warning("No chunks found matching your query")
            else:
                st.error(f"Error retrieving chunks: {response.text}")

# Add footer with documentation
st.markdown("---")
st.markdown("""
### About This RAG System

This system uses a combination of techniques:
- **Semantic Chunking**: Divides documents into meaningful chunks
- **Vector & Keyword Search**: Uses OpenSearch for both vector and keyword searches
- **RAG Pipeline**: Enhances LLM responses with relevant document context
""")
