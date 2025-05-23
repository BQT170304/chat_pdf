from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    """Request schema for chat queries"""
    query: str
    search_type: str = "hybrid"  # hybrid, semantic, or keyword


class ChunkResponse(BaseModel):
    """Response schema for text chunks"""
    title: str
    content: str


class UploadResponse(BaseModel):
    """Response schema for PDF uploads"""
    message: str
    document_info: Dict[str, Any]


class ChatResponse(BaseModel):
    """Response schema for chat"""
    response: str
    context: Optional[List[ChunkResponse]] = None 