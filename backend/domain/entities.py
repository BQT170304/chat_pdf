from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import uuid


class Chunk(BaseModel):
    """Represents a text chunk with its embedding"""
    id: str = ""
    file_id: str
    title: str
    content: str
    embedding: Optional[List[float]] = None
    
    def __init__(self, **data):
        if "id" not in data or not data["id"]:
            data["id"] = str(uuid.uuid4())
        super().__init__(**data)


class Document(BaseModel):
    """Represents a document with metadata"""
    id: str = ""
    name: str
    chunks: List[Chunk] = []
    
    def __init__(self, **data):
        if "id" not in data or not data["id"]:
            data["id"] = str(uuid.uuid4())
        super().__init__(**data)


class QueryResult(BaseModel):
    """Represents a query result with response and context"""
    response: str
    context: List[Chunk] = [] 