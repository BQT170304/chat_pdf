from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pathlib

from .entities import Chunk, Document, QueryResult


class EmbedderInterface(ABC):
    """Interface for embedding models"""
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for a text string"""
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple text strings"""
        pass


class DatabaseInterface(ABC):
    """Interface for database operations"""
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the database, create indices if needed"""
        pass
    
    @abstractmethod
    def index_document(self, document: Document) -> None:
        """Index a document and its chunks in the database"""
        pass
    
    @abstractmethod
    def search_chunks(self, query: str, search_type: str = "hybrid", limit: int = 5) -> List[Chunk]:
        """Search for chunks based on a query, using the specified search type"""
        pass


class PDFParserInterface(ABC):
    """Interface for PDF parsing operations"""
    
    @abstractmethod
    def extract_text(self, file_path: pathlib.Path) -> str:
        """Extract text from a PDF file"""
        pass
    
    @abstractmethod
    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """Split text into chunks with semantic consideration"""
        pass
    
    @abstractmethod
    def process_pdf(self, file_path: pathlib.Path, document_name: str = None) -> Document:
        """Process a PDF file into a Document with chunks"""
        pass


class LLMInterface(ABC):
    """Interface for Large Language Models"""
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the LLM"""
        pass
    
    @abstractmethod
    def generate_response(self, query: str, context: List[Chunk] = None) -> str:
        """Generate a response to a query, optionally using context"""
        pass 