import os
import re
import pathlib
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.domain.interfaces import PDFParserInterface
from backend.domain.entities import Document, Chunk


class PyMuPDFParser(PDFParserInterface):
    """PDF Parser implementation using PyMuPDF"""
    
    def __init__(self, embedder):
        """Initialize the PDF parser with an embedder"""
        self.embedder = embedder
    
    def extract_text(self, file_path: pathlib.Path) -> str:
        """Extract text from a PDF file using PyMuPDF"""
        text = ""
        try:
            doc = fitz.open(str(file_path))
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            raise e
    
    def clean_text(self, text: str) -> str:
        """Clean the extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """Split text into chunks with semantic consideration"""
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
            embedding = self.embedder.embed_text(chunk_text)
            
            processed_chunks.append({
                "title": title,
                "content": chunk_text,
                "embedding": embedding,
                "chunk_id": i,
            })
        
        return processed_chunks
    
    def process_pdf(self, file_path: pathlib.Path, document_name: str = None) -> Document:
        """Process a PDF file into a Document with chunks"""
        # Get filename without extension
        if document_name is None:
            filename = file_path.name
            document_name = file_path.stem
        
        # Extract and clean text
        raw_text = self.extract_text(file_path)
        cleaned_text = self.clean_text(raw_text)
        
        # Create semantic chunks
        chunks_data = self.chunk_text(cleaned_text)
        
        # Create the document
        document = Document(
            name=document_name,
            chunks=[]
        )
        
        # Add chunks to the document
        for chunk_data in chunks_data:
            chunk = Chunk(
                file_id=document.id,
                title=chunk_data["title"],
                content=chunk_data["content"],
                embedding=chunk_data["embedding"]
            )
            document.chunks.append(chunk)
        
        return document 