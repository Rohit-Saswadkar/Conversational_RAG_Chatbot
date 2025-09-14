"""
Demo and Testing Script for 500-Page PDF Q&A Chatbot

This script demonstrates the capabilities of the Q&A chatbot and provides
automated testing functionality to validate system performance.

Usage:
    python demo_test.py --demo          # Run interactive demo
    python demo_test.py --test          # Run automated tests
    python demo_test.py --benchmark     # Run performance benchmark
    python demo_test.py --evaluate      # Run comprehensive evaluation
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.pdf_processor import PDFProcessor
from src.chunking_strategy import IntelligentChunker
from src.vector_store import HybridVectorStore
from src.qa_engine import GeminiQAEngine
from src.adaptive_config import AdaptiveConfigManager
from src.evaluation import QAChatbotEvaluator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_pdf_path():
    """
    Get PDF path - using the specific OASIS E1 manual
    """
    # Use the specific OASIS E1 manual path
    oasis_path = "C:\\Users\\Jeeva\\Desktop\\Projects\\ya_labs_qa_chatbot\\draft-oasis-e1-manual-04-28-2024.pdf"
    
    if os.path.exists(oasis_path):
        return oasis_path
    
    # Fallback to environment variable
    pdf_path = os.getenv("PDF_PATH")
    if pdf_path and os.path.exists(pdf_path):
        return pdf_path
    
    print("❌ OASIS E1 manual not found!")
    print(f"Expected location: {oasis_path}")
    print("Please ensure the OASIS E1 manual is in the correct location")
    return None


def run_demo():
    """Run interactive demo of the Q&A chatbot"""
    print("🚀 Starting Q&A Chatbot Demo")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Please set GOOGLE_API_KEY environment variable")
        return
    
    # Get PDF path
    pdf_path = get_pdf_path()
    if not pdf_path:
        return
    
    # Initialize adaptive components
    print("🔧 Initializing adaptive components...")
    
    config_manager = AdaptiveConfigManager()
    pdf_processor = PDFProcessor()
    
    # Process PDF and analyze characteristics
    print(f"📄 Processing OASIS E1 Manual: {pdf_path}")
    full_text, page_info = pdf_processor.process_pdf(pdf_path)
    print(f"✅ Extracted text from {len(page_info)} pages")
    
    # Analyze document characteristics for adaptive configuration
    print("🧠 Analyzing document characteristics...")
    doc_chars = config_manager.analyze_document_characteristics(full_text, page_info)
    print(f"✅ Document type: {doc_chars.document_type}, Language complexity: {doc_chars.language_complexity:.2f}")
    
    # Get adaptive chunking configuration
    chunking_config = config_manager.get_adaptive_chunking_config()
    print(f"📊 Adaptive chunking config: {chunking_config['min_chunk_size']}-{chunking_config['max_chunk_size']} chars, quality: {chunking_config['quality_threshold']:.2f}")
    
    # Initialize chunker with adaptive configuration
    chunker = IntelligentChunker(
        min_chunk_size=chunking_config["min_chunk_size"],
        max_chunk_size=chunking_config["max_chunk_size"],
        overlap_size=chunking_config["overlap_size"],
        quality_threshold=chunking_config["quality_threshold"]
    )
    
    vector_store = HybridVectorStore(document_type=doc_chars.document_type)
    
    # Process content with adaptive chunking
    print("🧠 Creating adaptive semantic chunks...")
    chunks = chunker.process_document(full_text, page_info)
    print(f"✅ Created {len(chunks)} adaptive semantic chunks")
    
    # Build vector store
    print("🔍 Building vector store...")
    vector_store.add_chunks(chunks)
    print("✅ Vector store ready")
    
    # Initialize adaptive Q&A engine
    print("🤖 Initializing Adaptive Gemini Flash 2.0...")
    qa_engine = AdaptiveGeminiQAEngine(
        api_key=api_key, 
        vector_store=vector_store,
        document_type=doc_chars.document_type
    )
    print("✅ Adaptive Q&A engine ready")
    
    # Demo queries specific to OASIS E1 Manual
    demo_queries = [
        "What is the OASIS E1 system?",
        "How do I set up the OASIS E1 system?",
        "What are the safety requirements for OASIS E1?",
        "What tools are needed for OASIS E1 installation?",
        "How do I troubleshoot OASIS E1 issues?",
    ]
    
    print("\n🎯 Running Demo Queries")
    print("=" * 50)
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("-" * 40)
        
        start_time = time.time()
        response = qa_engine.answer_question(query)
        response_time = time.time() - start_time
        
        print(f"🤖 Answer (Confidence: {response.confidence:.2f}, Time: {response_time:.2f}s):")
        print(response.answer)
        
        if response.sources:
            print(f"\n📖 Sources ({len(response.sources)}):")
            for j, source in enumerate(response.sources[:2]):  # Show top 2
                print(f"  {j+1}. Page {source['page_number']} - {source['section_title']}")
        
        if response.follow_up_questions:
            print(f"\n🤔 Follow-up suggestions:")
            for question in response.follow_up_questions[:2]:  # Show top 2
                print(f"  • {question}")
        
        print("\n" + "="*50)


def run_automated_tests():
    """Run automated tests to validate system functionality"""
    print("🧪 Running Automated Tests")
    print("=" * 50)
    
    # Test 1: Component initialization
    print("Test 1: Component Initialization")
    try:
        chunker = IntelligentChunker()
        vector_store = HybridVectorStore()
        pdf_processor = PDFProcessor()
        print("✅ Components initialized successfully")
    except Exception as e:
        print(f"❌ Component initialization failed: {e}")
        return
    
    # Test 2: PDF Processing and Chunking functionality
    print("\nTest 2: PDF Processing and Chunking")
    pdf_path = get_pdf_path()
    if not pdf_path:
        print("⚠️ Skipping PDF tests - no PDF path provided")
        return
    
    try:
        # Process PDF
        full_text, page_info = pdf_processor.process_pdf(pdf_path)
        assert len(page_info) > 0, "No pages extracted"
        assert len(full_text) > 0, "No text extracted"
        print(f"✅ PDF processed: {len(page_info)} pages, {len(full_text)} characters")
        
        # Test chunking
        chunks = chunker.process_document(full_text, page_info)
        assert len(chunks) > 0, "No chunks created"
        assert all(chunk.metadata.quality_score > 0 for chunk in chunks), "Invalid quality scores"
        print(f"✅ Created {len(chunks)} valid chunks")
    except Exception as e:
        print(f"❌ PDF processing test failed: {e}")
        return
    
    # Test 3: Vector store functionality
    print("\nTest 3: Vector Store Functionality")
    try:
        vector_store.add_chunks(chunks)
        assert vector_store.index.ntotal > 0, "No vectors in index"
        
        # Test search
        results = vector_store.hybrid_search("safety requirements", k=3)
        assert len(results) > 0, "No search results"
        print(f"✅ Vector store working, found {len(results)} results")
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        return
    
    # Test 4: Q&A engine (if API key available)
    print("\nTest 4: Q&A Engine")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️ Skipping Q&A test - no API key provided")
    else:
        try:
            # Use adaptive Q&A engine for testing
            qa_engine = AdaptiveGeminiQAEngine(api_key, vector_store, document_type="manual")
            response = qa_engine.answer_question("What are the safety requirements?")
            
            assert response.answer, "Empty answer"
            assert response.confidence > 0, "Invalid confidence score"
            print(f"✅ Adaptive Q&A engine working, confidence: {response.confidence:.2f}")
        except Exception as e:
            print(f"❌ Q&A engine test failed: {e}")
            return
    
    print("\n🎉 All tests passed!")


def run_benchmark():
    """Run performance benchmark"""
    print("⚡ Running Performance Benchmark")
    print("=" * 50)
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Please set GOOGLE_API_KEY environment variable")
        return
    
    pdf_path = get_pdf_path()
    if not pdf_path:
        return
    
    # Initialize system
    print("🔧 Setting up benchmark environment...")
    pdf_processor = PDFProcessor()
    chunker = IntelligentChunker()
    vector_store = HybridVectorStore()
    
    # Measure PDF processing
    print("📄 Benchmarking PDF processing...")
    start_time = time.time()
    full_text, page_info = pdf_processor.process_pdf(pdf_path)
    pdf_time = time.time() - start_time
    print(f"PDF Processing: {len(page_info)} pages in {pdf_time:.2f}s")
    
    # Measure chunking performance
    print("📊 Benchmarking chunking performance...")
    start_time = time.time()
    chunks = chunker.process_document(full_text, page_info)
    chunking_time = time.time() - start_time
    print(f"Chunking: {len(chunks)} chunks in {chunking_time:.2f}s")
    
    # Measure vector store building
    print("🔍 Benchmarking vector store building...")
    start_time = time.time()
    vector_store.add_chunks(chunks)
    indexing_time = time.time() - start_time
    print(f"Indexing: {len(chunks)} chunks in {indexing_time:.2f}s")
    
    # Initialize adaptive Q&A engine
    qa_engine = AdaptiveGeminiQAEngine(api_key, vector_store, document_type="manual")
    
    # Benchmark query processing
    print("💬 Benchmarking query processing...")
    test_queries = [
        "What are the safety requirements?",
        "How do I get started?",
        "What tools are needed?",
        "How do I troubleshoot issues?",
        "What is the main purpose?",
    ]
    
    query_times = []
    for query in test_queries:
        start_time = time.time()
        response = qa_engine.answer_question(query)
        query_time = time.time() - start_time
        query_times.append(query_time)
    
    avg_query_time = sum(query_times) / len(query_times)
    
    print("\n📈 Benchmark Results:")
    print(f"PDF Processing Speed: {len(page_info) / pdf_time:.1f} pages/second")
    print(f"Chunking Speed: {len(chunks) / chunking_time:.1f} chunks/second")
    print(f"Indexing Speed: {len(chunks) / indexing_time:.1f} chunks/second")
    print(f"Average Query Time: {avg_query_time:.2f} seconds")
    print(f"Queries per Second: {1/avg_query_time:.1f}")


def run_comprehensive_evaluation():
    """Run comprehensive evaluation with detailed metrics"""
    print("🔬 Running Comprehensive Evaluation")
    print("=" * 50)
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Please set GOOGLE_API_KEY environment variable")
        return
    
    pdf_path = get_pdf_path()
    if not pdf_path:
        return
    
    # Initialize components
    pdf_processor = PDFProcessor()
    chunker = IntelligentChunker()
    vector_store = HybridVectorStore()
    
    # Process PDF
    full_text, page_info = pdf_processor.process_pdf(pdf_path)
    chunks = chunker.process_document(full_text, page_info)
    vector_store.add_chunks(chunks)
    
    # Initialize adaptive Q&A engine and evaluator
    qa_engine = AdaptiveGeminiQAEngine(api_key, vector_store, document_type="manual")
    evaluator = QAChatbotEvaluator(chunker, vector_store, qa_engine)
    
    # Run evaluation
    results = evaluator.run_comprehensive_evaluation()
    
    # Generate report
    report = evaluator.generate_evaluation_report(results)
    
    # Display summary
    print("\n📊 Evaluation Summary:")
    print(f"Total Chunks: {results['summary']['total_chunks']}")
    print(f"Average Chunk Quality: {results['summary']['avg_chunk_quality']:.3f}")
    print(f"Retrieval Precision@5: {results['retrieval_accuracy']['precision_at_5']:.3f}")
    print(f"Answer Confidence: {results['answer_quality']['avg_confidence']:.3f}")
    print(f"Overall Performance: {results['summary']['overall_performance']:.3f}")
    
    # Show recommendations
    if report['recommendations']:
        print("\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")


def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(description="Demo and Testing for Q&A Chatbot")
    parser.add_argument("--demo", action="store_true", help="Run interactive demo")
    parser.add_argument("--test", action="store_true", help="Run automated tests")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--evaluate", action="store_true", help="Run comprehensive evaluation")
    parser.add_argument("--all", action="store_true", help="Run all tests and evaluations")
    
    args = parser.parse_args()
    
    if args.demo or args.all:
        run_demo()
        print("\n")
    
    if args.test or args.all:
        run_automated_tests()
        print("\n")
    
    if args.benchmark or args.all:
        run_benchmark()
        print("\n")
    
    if args.evaluate or args.all:
        run_comprehensive_evaluation()
        print("\n")
    
    if not any([args.demo, args.test, args.benchmark, args.evaluate, args.all]):
        print("Please specify an action: --demo, --test, --benchmark, --evaluate, or --all")
        parser.print_help()


if __name__ == "__main__":
    main()
