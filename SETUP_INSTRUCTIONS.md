# Setup Instructions - 500-Page PDF Q&A Chatbot

This guide provides detailed setup instructions for running the Q&A chatbot locally.

## 📋 Prerequisites

### System Requirements
- **Operating System**: Windows 10+, macOS 10.14+, or Linux
- **Python**: Version 3.8 or higher
- **RAM**: Minimum 4GB, recommended 8GB+ for large PDFs
- **Storage**: At least 2GB free space for dependencies and indices
- **Internet**: Required for downloading models and API access

### API Requirements
- **Google AI API Key**: Required for Gemini Flash 2.0 access
  - Visit [Google AI Studio](https://aistudio.google.com/)
  - Create an account or sign in
  - Generate an API key
  - Note: Some regions may have restricted access

## 🔧 Installation Steps

### 1. Environment Setup

#### Option A: Using Python Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv qa_chatbot_env

# Activate environment
# On Windows:
qa_chatbot_env\Scripts\activate
# On macOS/Linux:
source qa_chatbot_env/bin/activate
```

#### Option B: Using Conda
```bash
# Create conda environment
conda create -n qa_chatbot python=3.9
conda activate qa_chatbot
```

### 2. Install Dependencies

```bash
# Install main dependencies
pip install -r requirements.txt

# Download NLTK data (required for text processing)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 3. Environment Configuration

Create a `.env` file in the project root:
```bash
cp .env.example .env
```

Edit the `.env` file:
```env
# Google AI API Key for Gemini Flash 2.0
GOOGLE_API_KEY=your_actual_api_key_here

# Optional: Default PDF path
PDF_PATH=path/to/your/500page_manual.pdf
```

### 4. Verify Installation

Run the verification script:
```bash
python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import streamlit
    print('✅ Streamlit installed')
except ImportError:
    print('❌ Streamlit not found')

try:
    import sentence_transformers
    print('✅ Sentence Transformers installed')
except ImportError:
    print('❌ Sentence Transformers not found')

try:
    import google.generativeai as genai
    print('✅ Google Generative AI installed')
except ImportError:
    print('❌ Google Generative AI not found')

try:
    import faiss
    print('✅ FAISS installed')
except ImportError:
    print('❌ FAISS not found')
"
```

## 🚀 Running the Application

### Method 1: Web Interface (Streamlit)

1. **Start the application**:
```bash
streamlit run main.py
```

2. **Open your browser** to `http://localhost:8501`

3. **Configure the application**:
   - Enter your Google AI API key in the sidebar
   - Choose "Upload new PDF" or "Load saved index"
   - If uploading PDF: Select your 500-page manual file
   - Wait for processing to complete

4. **Start asking questions**!

### Method 2: Command Line Interface

#### Process a new PDF:
```bash
python main.py --pdf_path "path/to/your/manual.pdf" --api_key "your_google_api_key"
```

#### Load from saved index:
```bash
python main.py --load_index "saved_index" --api_key "your_google_api_key"
```

## 📊 First-Time Setup Process

### Expected Processing Times (500-page PDF)
1. **PDF Extraction**: 2-3 minutes
2. **Intelligent Chunking**: 30-60 seconds
3. **Embedding Generation**: 3-5 minutes
4. **Vector Store Building**: 1-2 minutes
5. **Total Setup Time**: ~6-10 minutes

### During Processing, You'll See:
```
🔄 Processing PDF file...
✅ Extracted text from 487 pages
✅ Created 1,247 semantic chunks
✅ Vector store ready
✅ Q&A engine ready
✅ Index saved to saved_index
```

## 🔍 Testing the System

### Sample Questions to Try
```
1. "What is the main purpose of this manual?"
2. "How do I get started with the basic procedures?"
3. "What are the safety requirements mentioned?"
4. "Can you explain the troubleshooting process?"
5. "What tools or equipment are needed?"
```

### Expected Response Format
```
📝 Answer (Confidence: 0.87):
[Detailed answer based on manual content]

📖 Sources (3 found):
  1. Page 45 - Safety Requirements
  2. Page 120 - Basic Procedures
  3. Page 203 - Equipment List

🤔 Follow-up suggestions:
  • What specific safety equipment is required?
  • How often should equipment be inspected?
  • Are there any regulatory requirements?
```

## 🛠️ Troubleshooting Common Issues

### Issue 1: "Module not found" errors
**Solution**:
```bash
# Ensure virtual environment is activated
# Reinstall requirements
pip install --upgrade -r requirements.txt
```

### Issue 2: PDF processing fails
**Symptoms**: Error during PDF extraction
**Solutions**:
1. Check if PDF is password-protected
2. Try a different PDF file first
3. Ensure sufficient RAM (close other applications)
4. Check file path is correct

### Issue 3: Gemini API errors
**Symptoms**: "API key invalid" or quota errors
**Solutions**:
1. Verify API key is correct in `.env` file
2. Check API quota in Google AI Studio
3. Ensure your region has access to Gemini Flash 2.0
4. Try a different API key

### Issue 4: Slow performance
**Symptoms**: Long response times
**Solutions**:
1. Use GPU acceleration: `pip install faiss-gpu` (if CUDA available)
2. Reduce chunk count by increasing quality threshold
3. Use smaller embedding model
4. Close other memory-intensive applications

### Issue 5: Memory errors
**Symptoms**: "Out of memory" during processing
**Solutions**:
1. Process smaller PDF files first
2. Increase system virtual memory
3. Use cloud instance with more RAM
4. Reduce max_chunk_size in configuration

## ⚙️ Advanced Configuration

### Custom Chunking Settings
Edit `main.py` to modify chunking parameters:
```python
self.chunker = IntelligentChunker(
    min_chunk_size=150,      # Reduce for more chunks
    max_chunk_size=600,      # Reduce for smaller chunks
    overlap_size=50,         # Reduce overlap
    quality_threshold=0.5    # Increase for higher quality
)
```

### Different Embedding Models
Modify the vector store initialization:
```python
# Faster but less accurate
self.vector_store = HybridVectorStore(
    embedding_model_name="all-MiniLM-L12-v2"
)

# More accurate but slower
self.vector_store = HybridVectorStore(
    embedding_model_name="all-mpnet-base-v2"
)
```

### GPU Acceleration
For systems with CUDA-compatible GPUs:
```bash
# Install GPU version of FAISS
pip uninstall faiss-cpu
pip install faiss-gpu

# Enable GPU in vector store
self.vector_store = HybridVectorStore(use_gpu=True)
```

## 📁 File Structure After Setup

```
ya_labs_qa_chatbot/
├── main.py                 # Main application
├── requirements.txt        # Dependencies
├── .env                   # Environment variables
├── README.md              # Documentation
├── SETUP_INSTRUCTIONS.md  # This file
├── src/                   # Source code
│   ├── pdf_processor.py
│   ├── chunking_strategy.py
│   ├── vector_store.py
│   └── qa_engine.py
├── saved_index/           # Generated after first use
│   ├── faiss_index.bin
│   ├── chunks.pkl
│   ├── metadata.json
│   ├── tfidf_vectorizer.pkl
│   └── tfidf_matrix.pkl
└── temp_files/           # Temporary processing files
```

## 🔄 Regular Usage Workflow

### For New PDFs:
1. Start application: `streamlit run main.py`
2. Upload PDF in sidebar
3. Wait for processing
4. Index automatically saved for future use

### For Previously Processed PDFs:
1. Start application: `streamlit run main.py`
2. Select "Load saved index"
3. Choose index path (default: "saved_index")
4. Start asking questions immediately

## 📞 Support

If you encounter issues during setup:

1. **Check the logs** - Look for error messages in the terminal
2. **Verify requirements** - Ensure all prerequisites are met
3. **Test with smaller files** - Try a 10-20 page PDF first
4. **Check online resources** - Many Python package issues have known solutions

### Common Error Messages and Solutions

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'streamlit'` | Run `pip install streamlit` |
| `google.api_core.exceptions.Unauthenticated` | Check your Google AI API key |
| `RuntimeError: CUDA out of memory` | Disable GPU or increase GPU memory |
| `FileNotFoundError: [Errno 2] No such file` | Check file paths are correct |

## ✅ Success Indicators

You'll know the setup is successful when:
- ✅ All Python imports work without errors
- ✅ PDF processing completes without crashes
- ✅ Vector store builds successfully
- ✅ First query returns a relevant answer
- ✅ Index files are created in the saved_index folder

Now you're ready to start using the 500-Page PDF Q&A Chatbot!
