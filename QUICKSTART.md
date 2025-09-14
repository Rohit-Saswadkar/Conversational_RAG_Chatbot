# Quick Start Guide - 500-Page PDF Q&A Chatbot

This guide will get you up and running quickly with your PDF Q&A chatbot.

## 🚀 Prerequisites

1. **Python 3.8+** installed on your system
2. **Google AI API Key** - Get one from [Google AI Studio](https://aistudio.google.com/)
3. **Your 500-page PDF manual** ready to process

## ⚡ Quick Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download NLTK data (required)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 3. Set up environment variables
echo "GOOGLE_API_KEY=your_actual_api_key_here" > .env
echo "PDF_PATH=/path/to/your/500page_manual.pdf" >> .env
```

## 🎯 How to Run

### Option 1: Web Interface (Recommended)
```bash
streamlit run main.py
```
- Open browser to `http://localhost:8501`
- Enter your API key in sidebar
- Upload your PDF or load saved index
- Start asking questions!

### Option 2: Command Line Interface
```bash
# Process new PDF
python main.py --pdf_path "/path/to/your/manual.pdf" --api_key "your_api_key"

# Load saved index
python main.py --load_index "saved_index" --api_key "your_api_key"
```

### Option 3: Demo and Testing
```bash
# Run demo (requires PDF_PATH in .env)
python demo_test.py --demo

# Run automated tests
python demo_test.py --test

# Run performance benchmark
python demo_test.py --benchmark

# Run comprehensive evaluation
python demo_test.py --evaluate

# Run everything
python demo_test.py --all
```

## 📁 Required Environment Variables

Create a `.env` file in the project root:
```env
# Required: Google AI API key
GOOGLE_API_KEY=your_actual_google_ai_api_key

# Required for demo/testing: Path to your PDF
PDF_PATH=/full/path/to/your/500page_manual.pdf
```

## 🔧 Example Usage Commands

### Web Interface
```bash
# Just run this and use the web UI
streamlit run main.py
```

### Command Line Processing
```bash
# Process your PDF
python main.py \
  --pdf_path "/home/user/documents/technical_manual.pdf" \
  --api_key "AIzaSyD..."

# Use saved index for faster startup
python main.py \
  --load_index "saved_index" \
  --api_key "AIzaSyD..."
```

### Testing with Your PDF
```bash
# Set environment variables first
export GOOGLE_API_KEY="AIzaSyD..."
export PDF_PATH="/home/user/documents/technical_manual.pdf"

# Run demo
python demo_test.py --demo

# Run tests
python demo_test.py --test

# Run benchmark
python demo_test.py --benchmark
```

## ⏱️ Expected Processing Times

For a 500-page PDF:
- **PDF Extraction**: 2-3 minutes
- **Intelligent Chunking**: 30-60 seconds  
- **Vector Store Building**: 3-5 minutes
- **First Query Response**: 2-4 seconds
- **Subsequent Queries**: 1-3 seconds

## 🎯 Test Queries to Try

Once your system is running, try these example queries:

```
"What are the safety requirements?"
"How do I get started with installation?"
"What tools are needed for maintenance?"
"Can you explain the troubleshooting process?"
"What are the system specifications?"
"How do I configure the advanced settings?"
"What is the warranty policy?"
"What are the emergency procedures?"
```

## 🚨 Troubleshooting Quick Fixes

### "Module not found" errors
```bash
pip install --upgrade -r requirements.txt
```

### "API key invalid" errors
- Check your Google AI API key is correct
- Verify it has access to Gemini models
- Try regenerating the key

### PDF processing fails
- Check if PDF is password-protected
- Try with a smaller PDF first (10-20 pages)
- Ensure you have enough RAM (4GB+ recommended)

### Slow performance
```bash
# Use GPU if available
pip uninstall faiss-cpu
pip install faiss-gpu
```

## 📊 Success Indicators

You'll know everything is working when:
- ✅ PDF processing completes without errors
- ✅ Chunks are created (should see ~1000-3000 for 500 pages)
- ✅ Vector store builds successfully
- ✅ First query returns a relevant answer with sources
- ✅ Confidence scores are reasonable (0.5-0.9)

## 💡 Pro Tips

1. **First time**: Use web interface for easier setup
2. **Development**: Use command line for faster iteration
3. **Production**: Save indices to avoid reprocessing
4. **Testing**: Start with demo mode to verify everything works
5. **Performance**: Use GPU acceleration for large documents

## 🔄 Typical Workflow

```bash
# 1. First time setup
pip install -r requirements.txt
echo "GOOGLE_API_KEY=your_key" > .env
echo "PDF_PATH=/path/to/pdf" >> .env

# 2. Run demo to test
python demo_test.py --demo

# 3. Use web interface for interactive Q&A
streamlit run main.py

# 4. Or use command line for automation
python main.py --pdf_path "your_pdf.pdf" --api_key "your_key"
```

That's it! You should now have a working Q&A chatbot for your 500-page PDF manual.
