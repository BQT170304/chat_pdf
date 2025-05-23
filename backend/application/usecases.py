import pathlib
import tempfile
import os
from typing import List, Dict, Any, Optional, BinaryIO

from backend.domain.interfaces import PDFParserInterface, DatabaseInterface, LLMInterface
from backend.domain.entities import Document, Chunk, QueryResult


class UploadPDFUseCase:
    """Use case for uploading and processing PDF files"""
    
    def __init__(self, pdf_parser: PDFParserInterface, database: DatabaseInterface):
        """Initialize the use case with dependencies"""
        self.pdf_parser = pdf_parser
        self.database = database
    
    async def execute(self, file_content: BinaryIO, filename: str) -> Dict[str, Any]:
        """
        Process a PDF file:
        1. Save the file temporarily
        2. Extract and process text
        3. Index in database
        
        Args:
            file_content: Content of the uploaded file
            filename: Name of the uploaded file
            
        Returns:
            Information about the processed document
        """
        # Save the file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_content.read())
            temp_file_path = temp_file.name
        
        try:
            # Extract document name from filename
            document_name = os.path.splitext(filename)[0]
            
            # Process the PDF
            document = self.pdf_parser.process_pdf(
                pathlib.Path(temp_file_path), 
                document_name=document_name
            )
            
            # Index the document in the database
            self.database.index_document(document)
            
            # Return information about the processing
            return {
                "document_name": document.name,
                "total_chunks": len(document.chunks),
                "chunks": [{"title": chunk.title, "content": chunk.content[:100] + "..."} 
                           for chunk in document.chunks]
            }
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)


class ChatUseCase:
    """Use case for chatting with the RAG system"""
    
    def __init__(self, database: DatabaseInterface, llm: LLMInterface):
        """Initialize the use case with dependencies"""
        self.database = database
        self.llm = llm
    
    async def execute(self, query: str, search_type: str = "hybrid") -> QueryResult:
        """
        Generate a response to a user's query using RAG
        
        Args:
            query: User query
            search_type: Type of search to perform (semantic, keyword, or hybrid)
            
        Returns:
            Query result with response and context chunks
        """
        # Retrieve relevant chunks from the database
        chunks = self.database.search_chunks(query, search_type=search_type)
        
        # Generate a response using the LLM
        response = self.llm.generate_response(query, context=chunks)
        
        # Return the response with context
        return QueryResult(
            response=response,
            context=chunks
        ) 