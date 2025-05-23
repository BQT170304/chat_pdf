from typing import List
import os

from sentence_transformers import SentenceTransformer

from backend.domain.interfaces import EmbedderInterface


class SentenceTransformerEmbedder(EmbedderInterface):
    """Embedder implementation using SentenceTransformer models"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedder with a model name"""
        self.model_name = model_name
        self.model = None
    
    def initialize(self):
        """Initialize the model if it hasn't been loaded yet"""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for a text string"""
        self.initialize()
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple text strings"""
        self.initialize()
        embeddings = self.model.encode(texts)
        return [embedding.tolist() for embedding in embeddings]


class OpenAIEmbedder(EmbedderInterface):
    """Embedder implementation using OpenAI embedding models"""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """Initialize the embedder with a model name"""
        self.model_name = model_name
        self.client = None
    
    def initialize(self):
        """Initialize the client if it hasn't been loaded yet"""
        if self.client is None:
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key is required for OpenAIEmbedder")
            self.client = OpenAI(api_key=api_key)
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for a text string"""
        self.initialize()
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        return response.data[0].embedding
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple text strings"""
        self.initialize()
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        return [item.embedding for item in response.data] 