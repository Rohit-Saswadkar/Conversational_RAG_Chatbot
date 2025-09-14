#!/usr/bin/env python3
"""
Production server runner for PDF Q&A Chatbot
This runs the server without auto-reload for better performance
"""

import uvicorn
import os
from pathlib import Path

def run_production_server():
    print("🚀 Starting PDF Q&A Chatbot API (Production Mode)")
    print("=" * 50)
    
    # Check for GOOGLE_API_KEY
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Warning: GOOGLE_API_KEY not found in .env. Please set it for full functionality.")
    else:
        print("✅ Environment check passed")
    
    print("🌐 Starting production web server...")
    print("📱 Frontend: http://127.0.0.1:8000")
    print("📡 API Docs: http://127.0.0.1:8000/docs")
    print("⚠️  Auto-reload is DISABLED (production mode)")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Run without auto-reload for production
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False,  # No auto-reload
        workers=1,     # Single worker for development
        log_level="info"
    )

if __name__ == "__main__":
    # Add current directory to path for module imports
    import sys
    sys.path.append(str(Path(__file__).parent))
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    run_production_server()
