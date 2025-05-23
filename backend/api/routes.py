import os
from typing import List, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Depends

from backend.application.usecases import UploadPDFUseCase, ChatUseCase
from backend.domain.entities import QueryResult, Chunk
from backend.api.schemas import QueryRequest, ChunkResponse

from backend.infrastructure.embedder import SentenceTransformerEmbedder
from backend.infrastructure.pdf_parser import PyMuPDFParser
from backend.infrastructure.database import OpenSearchDatabase

from backend.infrastructure.llm import LangChainOpenAILLM, HuggingFaceLLM
from backend.infrastructure.database import OpenSearchDatabase


# Create dependency injection container in a real application
def get_upload_pdf_usecase():
    """Get the UploadPDFUseCase instance"""
    embedder = SentenceTransformerEmbedder()
    database = OpenSearchDatabase()
    pdf_parser = PyMuPDFParser(embedder)
    
    return UploadPDFUseCase(pdf_parser, database)


def get_chat_usecase():
    """Get the ChatUseCase instance"""
    database = OpenSearchDatabase()
    
    # Use OpenAI if API key is available, otherwise fall back to HuggingFace
    if os.environ.get("OPENAI_API_KEY"):
        llm = LangChainOpenAILLM()
    else:
        llm = HuggingFaceLLM()
    
    return ChatUseCase(database, llm)


# Create router
router = APIRouter(tags=["RAG"])


@router.post("/upload", response_model=Dict[str, Any])
async def upload_pdf(
    file: UploadFile = File(...),
    upload_pdf_usecase: UploadPDFUseCase = Depends(get_upload_pdf_usecase)
):
    """Upload a PDF and process it with semantic chunking"""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        document_info = await upload_pdf_usecase.execute(file.file, file.filename)
        return {"message": "PDF processed successfully", "document_info": document_info}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@router.get("/chunks", response_model=List[ChunkResponse])
async def get_chunks(
    query: str, 
    search_type: str = Query("hybrid"),
    chat_usecase: ChatUseCase = Depends(get_chat_usecase)
):
    """Get chunks from database based on query"""
    try:
        result = await chat_usecase.execute(query, search_type)
        
        # Convert Chunk objects to ChunkResponse objects
        responses = []
        for chunk in result.context:
            responses.append(ChunkResponse(title=chunk.title, content=chunk.content))
        
        return responses
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching chunks: {str(e)}")


@router.post("/chat", response_model=Dict[str, Any])
async def chat(
    request: QueryRequest,
    chat_usecase: ChatUseCase = Depends(get_chat_usecase)
):
    """Generate a response using the RAG pipeline"""
    try:
        result = await chat_usecase.execute(request.query, request.search_type)
        return {"response": result.response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}") 