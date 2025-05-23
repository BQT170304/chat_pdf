import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from backend.api.routes import router as rag_router

# Load environment variables from .env file
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Semantic Chunking RAG API",
    description="API for document upload, semantic chunking, and RAG-powered chat",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag_router, prefix="/api")

# Add a health check endpoint
@app.get("/health")
async def health_check():
    """Check if the API is running"""
    return {"status": "ok"}

# Add a welcome message
@app.get("/")
async def root():
    """Welcome message"""
    return {
        "message": "Welcome to the Semantic Chunking RAG API",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "upload_pdf": "/api/upload",
            "chat": "/api/chat",
            "get_chunks": "/api/chunks"
        }
    }

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=int(os.environ.get("PORT", 8000)),
        reload=True
    )
