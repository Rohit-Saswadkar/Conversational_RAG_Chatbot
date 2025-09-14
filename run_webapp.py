#!/usr/bin/env python3
"""
Startup script for the PDF Q&A Chatbot Web Application

This script starts the FastAPI backend server with the modern web interface.
"""

import os
import sys
import uvicorn
from pathlib import Path

def main():
    """Start the web application"""
    
    print("🚀 Starting PDF Q&A Chatbot Web Application")
    print("=" * 50)
    
    # Check if .env file exists
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ .env file not found!")
        print("Please create a .env file with your GOOGLE_API_KEY")
        print("Example:")
        print("GOOGLE_API_KEY=your_api_key_here")
        print("PDF_PATH=path_to_your_pdf.pdf")
        return
    
    # Check if OASIS manual exists
    pdf_path = Path("draft-oasis-e1-manual-04-28-2024.pdf")
    if not pdf_path.exists():
        print("⚠️  OASIS E1 manual not found at expected location")
        print(f"Expected: {pdf_path.absolute()}")
        print("The app will still start, but you'll need to provide a PDF path during initialization")
    
    # Check if static directory exists
    static_path = Path("static")
    if not static_path.exists():
        print("❌ Static directory not found!")
        print("Please ensure the static/ directory with frontend files exists")
        return
    
    print("✅ Environment check passed")
    print()
    print("🌐 Starting web server...")
    print("📱 Frontend: http://127.0.0.1:8000")
    print("📡 API Docs: http://127.0.0.1:8000/docs")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start the FastAPI server
    try:
        uvicorn.run(
            "app:app",
            host="127.0.0.1",
            port=8000,
            reload=True,
            log_level="warning",  # Reduced logging
            access_log=False      # Disable access logs
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")

if __name__ == "__main__":
    main()
