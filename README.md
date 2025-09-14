# 500-Page PDF Q&A Chatbot

An advanced Question-Answering system that transforms large PDF manuals into intelligent, conversational interfaces using state-of-the-art retrieval techniques and Google Gemini Flash 2.0.

## 🎯 Key Features

### Advanced Chunking Strategy
- **Structure-Aware Segmentation**: Preserves document hierarchy (chapters, sections, subsections)
- **Quality Filtering**: Automatically filters low-quality chunks using multiple metrics
- **Contextual Overlap**: Maintains continuity between chunks with intelligent overlap
- **Metadata Enrichment**: Adds semantic metadata for enhanced retrieval

### Hybrid Retrieval System
- **Multi-Modal Search**: Combines semantic similarity, keyword matching, and metadata filtering
- **Query Preprocessing**: Expands and optimizes user queries for better results
- **Relevance Ranking**: Advanced scoring system considering multiple factors
- **Context Expansion**: Includes surrounding content for better understanding

### Intelligent Answer Generation
- **Google Gemini Flash 2.0**: Latest language model for accurate responses
- **Context-Aware Prompting**: Optimized prompts for technical manual Q&A
- **Confidence Scoring**: Provides confidence levels for each answer
- **Source Attribution**: Links answers back to specific pages and sections

### User Experience
- **Interactive Web Interface**: Built with Streamlit for easy use
- **Conversation History**: Maintains context across multiple questions
- **Follow-up Suggestions**: Generates relevant follow-up questions
- **Multi-turn Support**: Handles complex, multi-part conversations

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google AI API Key (for Gemini Flash 2.0)
- At least 4GB RAM for processing large PDFs

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd ya_labs_qa_chatbot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your Google AI API key
```

### Usage

#### Option 1: Web Interface (Recommended)
```bash
streamlit run main.py
```
- Open your browser to `http://localhost:8501`
- Enter your Google AI API key in the sidebar
- Upload your PDF or load a saved index
- Start asking questions!

#### Option 2: Command Line Interface
```bash
# Process a new PDF
python main.py --pdf_path "path/to/your/manual.pdf" --api_key "your_google_api_key"

# Load from saved index
python main.py --load_index "saved_index" --api_key "your_google_api_key"
```

## 🏗️ Architecture Overview

### 1. PDF Processing Pipeline (`src/pdf_processor.py`)
- **Multi-Method Extraction**: Uses PyPDF2, pdfplumber, and PyMuPDF with automatic fallback
- **Quality Assessment**: Selects the best extraction method based on content quality
- **Text Cleaning**: Removes artifacts and normalizes formatting
- **Structure Enhancement**: Adds semantic markers for better chunking

### 2. Intelligent Chunking (`src/chunking_strategy.py`)
- **Hierarchical Processing**: Identifies document structure (titles, sections, lists)
- **Semantic Chunking**: Creates meaningful chunks that preserve context
- **Quality Scoring**: Evaluates chunk quality using multiple factors
- **Overlap Management**: Adds intelligent overlap to maintain context

### 3. Vector Store System (`src/vector_store.py`)
- **Hybrid Architecture**: Combines semantic and keyword search
- **FAISS Integration**: Efficient similarity search for large documents
- **Metadata Filtering**: Uses document structure for better retrieval
- **Query Expansion**: Preprocesses queries for optimal results

### 4. Q&A Engine (`src/qa_engine.py`)
- **Gemini Integration**: Uses Google's latest Gemini Flash 2.0 model
- **Context Management**: Maintains conversation history
- **Answer Validation**: Quality checks and confidence scoring
- **Source Attribution**: Links answers to specific document sections

## 🔧 Configuration Options

### Chunking Parameters
```python
chunker = IntelligentChunker(
    min_chunk_size=200,      # Minimum chunk size in characters
    max_chunk_size=800,      # Maximum chunk size in characters
    overlap_size=100,        # Overlap between chunks
    quality_threshold=0.3    # Minimum quality score for inclusion
)
```

### Vector Store Settings
```python
vector_store = HybridVectorStore(
    embedding_model_name="all-MiniLM-L6-v2",  # Sentence transformer model
    vector_dimension=384,                      # Embedding dimension
    use_gpu=False                             # GPU acceleration
)
```

### Q&A Engine Parameters
```python
qa_engine = GeminiQAEngine(
    api_key="your_api_key",
    model_name="gemini-2.0-flash-exp",  # Model name
    temperature=0.1,                     # Generation temperature
    max_tokens=2048                      # Maximum response length
)
```

## 📊 Design Decisions & Rationale

### 1. Multi-Method PDF Extraction
**Decision**: Use three different PDF libraries with automatic fallback
**Rationale**: Different PDFs have varying complexity. Some work better with specific libraries. This ensures maximum extraction quality.

### 2. Hierarchical Chunking Strategy
**Decision**: Preserve document structure instead of simple text splitting
**Rationale**: Technical manuals have inherent structure (chapters, sections) that provides important context for understanding.

### 3. Hybrid Retrieval System
**Decision**: Combine semantic search with keyword matching and metadata filtering
**Rationale**: No single retrieval method is perfect. Hybrid approach covers edge cases and improves overall accuracy.

### 4. Quality-Based Filtering
**Decision**: Filter chunks based on quality metrics before indexing
**Rationale**: Poor quality chunks (page numbers, headers, fragmented text) hurt retrieval performance.

### 5. Contextual Overlap
**Decision**: Add intelligent overlap between chunks
**Rationale**: Maintains context across chunk boundaries, crucial for understanding complex procedures.

### 6. Metadata Enrichment
**Decision**: Extract and store rich metadata with each chunk
**Rationale**: Enables more sophisticated retrieval strategies and better source attribution.

## 🧪 Testing & Evaluation

### Retrieval Quality Tests
The system includes built-in evaluation metrics:
- **Completeness**: Does the answer address all parts of the question?
- **Accuracy**: Does the answer stick to the source context?
- **Relevance**: Is the answer relevant to the question?
- **Clarity**: Is the answer well-structured and clear?

### Performance Benchmarks
- PDF Processing: ~2-3 minutes for 500 pages
- Chunking: ~30-60 seconds for typical manuals
- Vector Store Building: ~2-5 minutes depending on chunk count
- Query Response: ~2-4 seconds per question

### Example Test Queries
```python
test_queries = [
    "What are the safety requirements?",
    "How do I troubleshoot connection issues?",
    "What tools are needed for installation?",
    "Can you explain the maintenance procedure?",
    "What are the system requirements?"
]
```

## 🔄 Optimization Strategies

### Memory Management
- **Streaming Processing**: Process large PDFs in chunks to manage memory
- **Index Persistence**: Save/load vector indices to avoid reprocessing
- **Batch Operations**: Process multiple queries efficiently

### Query Optimization
- **Query Expansion**: Automatically expand queries with synonyms
- **Preprocessing**: Clean and normalize queries before processing
- **Caching**: Cache frequent query results for faster responses

### Retrieval Tuning
- **Dynamic K**: Adjust number of retrieved chunks based on query complexity
- **Score Thresholding**: Filter low-relevance results automatically
- **Re-ranking**: Multiple passes of ranking for better results

## 📝 API Reference

### Main Classes

#### `QAChatbot`
Main application class that orchestrates all components.

```python
chatbot = QAChatbot()
chatbot.initialize_from_pdf("manual.pdf", "api_key")
response = chatbot.answer_question("Your question here")
```

#### `PDFProcessor`
Handles PDF extraction with multiple methods.

```python
processor = PDFProcessor()
full_text, page_info = processor.process_pdf("manual.pdf")
```

#### `IntelligentChunker`
Creates semantic chunks from document text.

```python
chunker = IntelligentChunker()
chunks = chunker.process_document(full_text, page_info)
```

#### `HybridVectorStore`
Manages embeddings and retrieval.

```python
vector_store = HybridVectorStore()
vector_store.add_chunks(chunks)
results = vector_store.hybrid_search("query", k=5)
```

#### `GeminiQAEngine`
Generates answers using Gemini Flash 2.0.

```python
qa_engine = GeminiQAEngine(api_key, vector_store)
response = qa_engine.answer_question("query")
```

## 🐛 Troubleshooting

### Common Issues

1. **PDF Extraction Fails**
   - Try different PDF libraries in the processor
   - Check if PDF is password-protected or corrupted
   - Ensure sufficient memory for large files

2. **Poor Retrieval Quality**
   - Adjust chunking parameters
   - Increase quality threshold
   - Try different embedding models

3. **Gemini API Errors**
   - Verify API key is correct and active
   - Check quota and rate limits
   - Ensure model name is correct

4. **Memory Issues**
   - Process smaller PDFs first
   - Increase system RAM
   - Use GPU acceleration if available

### Performance Tuning

- **For faster processing**: Reduce chunk quality threshold
- **For better accuracy**: Increase overlap size and retrieval k
- **For lower memory usage**: Reduce max_chunk_size and vector_dimension

## 🔮 Future Enhancements

1. **Multi-Document Support**: Handle multiple PDFs simultaneously
2. **Advanced Analytics**: Usage analytics and query insights
3. **Fine-tuning**: Custom embedding models for domain-specific content
4. **Integration**: API endpoints for external application integration
5. **Evaluation Suite**: Comprehensive evaluation framework with benchmark datasets

## 📄 License

This project is provided as a demonstration for the AI Engineer position at YA Labs. Please see the accompanying documentation for usage guidelines.

## 🤝 Contributing

This is a demonstration project. For questions or discussions about the implementation, please reach out during the interview process.

---

**Note**: This implementation prioritizes intelligent design decisions and retrieval quality over production-grade features, as requested in the assignment. The focus is on demonstrating thoughtful problem-solving and technical depth rather than creating a fully polished product.
