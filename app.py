"""
FastAPI Backend for PDF Q&A Chatbot

A production-ready REST API backend that serves the Q&A functionality
with proper error handling, CORS support, and async operations.
"""

import os
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

# Import our Q&A system components
from src.pdf_processor import PDFProcessor
from src.chunking_strategy import IntelligentChunker
from src.vector_store import HybridVectorStore
from src.qa_engine import GeminiQAEngine
from src.adaptive_config import AdaptiveConfigManager

# Load environment variables
load_dotenv()

# FastAPI app initialization
app = FastAPI(
    title="PDF Q&A Chatbot API",
    description="Advanced Q&A system for 500-page PDF manuals using Google Gemini Flash 2.0",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    question: str
    conversation_history: Optional[List[Dict]] = None

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[Dict]
    processing_time: float
    follow_up_questions: List[str]
    conversation_id: Optional[str] = None

class InitRequest(BaseModel):
    pdf_path: Optional[str] = None
    use_oasis_manual: bool = True

class FileUploadResponse(BaseModel):
    filename: str
    size: int
    message: str
    file_path: str

class StatusResponse(BaseModel):
    status: str
    message: str
    details: Optional[Dict] = None

# Global variables for the Q&A system
qa_system = {
    "pdf_processor": None,
    "chunker": None,
    "vector_store": None,
    "qa_engine": None,
    "config_manager": None,
    "initialized": False,
    "processing": False
}

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend HTML page"""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Frontend not found</h1><p>Please ensure static/index.html exists</p>",
            status_code=404
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "system_initialized": qa_system["initialized"]
    }

@app.get("/api/status")
async def get_status():
    """Get current system status"""
    if qa_system["processing"]:
        return StatusResponse(
            status="processing",
            message="System is currently processing PDF and building vector store",
            details={"stage": "initialization"}
        )
    elif qa_system["initialized"]:
        return StatusResponse(
            status="ready",
            message="Q&A system is ready to answer questions",
            details={
                "chunks": len(qa_system["vector_store"].chunks) if qa_system["vector_store"] else 0,
                "model": "gemini-2.0-flash-exp"
            }
        )
    else:
        return StatusResponse(
            status="uninitialized",
            message="System needs to be initialized with a PDF document"
        )

@app.post("/api/upload", response_model=FileUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file for processing"""
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Validate file size (max 50MB)
    max_size = 50 * 1024 * 1024  # 50MB
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 50MB")
    
    # Create uploads directory if it doesn't exist
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    
    # Save the uploaded file
    file_path = upload_dir / file.filename
    
    try:
        with open(file_path, "wb") as f:
            f.write(content)
        
        return FileUploadResponse(
            filename=file.filename,
            size=len(content),
            message="File uploaded successfully",
            file_path=str(file_path)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

@app.post("/api/initialize")
async def initialize_system(request: InitRequest, background_tasks: BackgroundTasks):
    """Initialize the Q&A system with a PDF document"""
    
    if qa_system["processing"]:
        raise HTTPException(status_code=400, detail="System is already processing")
    
    # Start background initialization
    background_tasks.add_task(
        _initialize_system_background,
        request.pdf_path,
        request.use_oasis_manual
    )
    
    return StatusResponse(
        status="initializing",
        message="PDF processing started in background",
        details={"estimated_time": "2-3 minutes"}
    )

@app.post("/api/initialize-with-file")
async def initialize_with_uploaded_file(
    background_tasks: BackgroundTasks,
    file_path: str = Form(...)
):
    """Initialize the Q&A system with an uploaded PDF file"""
    
    if qa_system["processing"]:
        raise HTTPException(status_code=400, detail="System is already processing")
    
    # Check if the uploaded file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Uploaded file not found")
    
    # Start background initialization with the uploaded file
    background_tasks.add_task(
        _initialize_system_background,
        file_path,
        False  # Don't use OASIS manual, use uploaded file
    )
    
    return StatusResponse(
        status="initializing",
        message=f"Processing uploaded PDF: {Path(file_path).name}",
        details={"estimated_time": "2-3 minutes", "file": Path(file_path).name}
    )

async def _initialize_system_background(pdf_path: Optional[str], use_oasis_manual: bool):
    """Background task to initialize the Q&A system"""
    
    qa_system["processing"] = True
    
    try:
        # Determine PDF path
        if use_oasis_manual or not pdf_path:
            oasis_path = "draft-oasis-e1-manual-04-28-2024.pdf"
            if os.path.exists(oasis_path):
                pdf_path = oasis_path
            else:
                raise FileNotFoundError("OASIS E1 manual not found")
        
        # Check if PDF exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Get API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        print(f"🚀 Initializing Q&A system with PDF: {pdf_path}")
        
        # Initialize components
        qa_system["config_manager"] = AdaptiveConfigManager()
        qa_system["pdf_processor"] = PDFProcessor()
        
        # Process PDF
        print("📄 Processing PDF...")
        full_text, page_info = qa_system["pdf_processor"].process_pdf(pdf_path)
        
        # Analyze document characteristics
        print("🧠 Analyzing document characteristics...")
        doc_chars = qa_system["config_manager"].analyze_document_characteristics(full_text, page_info)
        
        # Get adaptive chunking configuration
        chunking_config = qa_system["config_manager"].get_adaptive_chunking_config()
        
        # Initialize chunker with adaptive configuration
        qa_system["chunker"] = IntelligentChunker(
            min_chunk_size=chunking_config["min_chunk_size"],
            max_chunk_size=chunking_config["max_chunk_size"],
            overlap_size=chunking_config["overlap_size"],
            quality_threshold=chunking_config["quality_threshold"]
        )
        
        qa_system["vector_store"] = HybridVectorStore(document_type=doc_chars.document_type)
        
        # Create chunks
        print("🧠 Creating semantic chunks...")
        chunks = qa_system["chunker"].process_document(full_text, page_info)
        
        # Build vector store
        print("🔍 Building vector store...")
        qa_system["vector_store"].add_chunks(chunks)
        
        # Initialize Q&A engine
        print("🤖 Initializing Gemini Flash 2.0...")
        qa_system["qa_engine"] = GeminiQAEngine(
            api_key=api_key,
            vector_store=qa_system["vector_store"]
        )
        
        # Save index for faster future loads
        try:
            index_path = "saved_index"
            qa_system["vector_store"].save_index(index_path)
            print(f"💾 Vector store saved to {index_path}")
        except Exception as e:
            print(f"⚠️ Could not save index: {e}")
        
        qa_system["initialized"] = True
        qa_system["processing"] = False
        
        print("✅ Q&A system initialization complete!")
        
    except Exception as e:
        qa_system["processing"] = False
        qa_system["initialized"] = False
        print(f"❌ Initialization failed: {str(e)}")
        raise e

@app.post("/api/query", response_model=QueryResponse)
async def answer_question(request: QueryRequest):
    """Answer a question using the Q&A system"""
    
    if not qa_system["initialized"]:
        raise HTTPException(status_code=400, detail="System not initialized. Please initialize first.")
    
    if qa_system["processing"]:
        raise HTTPException(status_code=400, detail="System is currently processing. Please wait.")
    
    try:
        start_time = time.time()
        
        # Get answer from Q&A engine
        response = qa_system["qa_engine"].answer_question(
            request.question,
            conversation_history=request.conversation_history
        )
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            answer=response.answer,
            confidence=response.confidence,
            sources=response.sources,
            processing_time=processing_time,
            follow_up_questions=response.follow_up_questions,
            conversation_id=None  # Can be implemented for session management
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/api/statistics")
async def get_statistics():
    """Get system statistics and metrics"""
    
    if not qa_system["initialized"]:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    try:
        stats = qa_system["vector_store"].get_statistics()
        return {
            "status": "ready",
            "statistics": stats,
            "system_info": {
                "model": "gemini-2.0-flash-exp",
                "embedding_model": stats.get("embedding_model", "all-MiniLM-L6-v2"),
                "embedding_dimension": stats.get("embedding_dimension", 384)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")

# Auto-initialize with OASIS manual on startup
@app.on_event("startup")
async def startup_event():
    """Auto-initialize the system on startup"""
    print("🚀 Starting PDF Q&A Chatbot API...")
    
    # Check if we should auto-initialize
    auto_init = os.getenv("AUTO_INITIALIZE", "true").lower() == "true"
    if auto_init:
        # Start background initialization
        asyncio.create_task(_initialize_system_background(None, True))

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
