import os
import re
from typing import List, Dict, Any
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from sentence_transformers import SentenceTransformer

from opensearch_client import OpenSearchClient

# Initialize the sentence transformer model for embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using PyMuPDF"""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        raise e

def clean_text(text: str) -> str:
    """Clean the extracted text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove page numbers
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    return text.strip()

def semantic_chunking(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Perform semantic chunking on the text
    
    Args:
        text: The text to chunk
        chunk_size: The size of each chunk
        chunk_overlap: The overlap between chunks
        
    Returns:
        A list of chunks, each with content and metadata
    """
    # Create a text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    # Split the text
    chunks = text_splitter.create_documents([text])
    
    # Process the chunks to extract semantic information
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_text = chunk.page_content
        # Generate a title based on the first line or first few words
        title = chunk_text.split('\n', 1)[0][:50] if '\n' in chunk_text else chunk_text[:50]
        title = f"Chunk {i+1}: {title}..."
        
        # Generate embedding for the chunk
        embedding = model.encode(chunk_text)
        
        processed_chunks.append({
            "title": title,
            "content": chunk_text,
            "embedding": embedding.tolist(),
            "chunk_id": i,
        })
    
    return processed_chunks

def process_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Process a PDF file:
    1. Extract text
    2. Clean text
    3. Perform semantic chunking
    4. Index in OpenSearch
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Information about the processed document
    """
    # Get filename without extension
    filename = os.path.basename(pdf_path)
    document_name = os.path.splitext(filename)[0]
    
    # Extract and clean text
    raw_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(raw_text)
    
    # Create semantic chunks
    chunks = semantic_chunking(cleaned_text)
    
    # Get OpenSearch client instance
    opensearch_client = OpenSearchClient.get_instance()
    
    # Index the chunks in OpenSearch
    for chunk in chunks:
        opensearch_client.index_document(
            index_name="documents",
            document={
                "title": chunk["title"],
                "content": chunk["content"],
                "document_name": document_name,
                "embedding": chunk["embedding"]
            }
        )
    
    return {
        "document_name": document_name,
        "total_chunks": len(chunks),
        "chunks": [{"title": chunk["title"], "content": chunk["content"][:100] + "..."} for chunk in chunks]
    } 