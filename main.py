"""
Main Application: 500-Page PDF Q&A Chatbot

This is the main entry point for the Q&A chatbot that processes a 500-page PDF
and enables intelligent question-answering using advanced retrieval and Gemini Flash 2.0.

Usage:
    python main.py --pdf_path <path_to_pdf> --api_key <google_api_key>
    python main.py --load_index <index_path> --api_key <google_api_key>
"""

import os
import sys
import argparse
import streamlit as st
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.pdf_processor import PDFProcessor
from src.chunking_strategy import IntelligentChunker
from src.vector_store import HybridVectorStore
from src.qa_engine import GeminiQAEngine
from src.adaptive_config import AdaptiveConfigManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class QAChatbot:
    """Main Q&A Chatbot Application"""
    
    def __init__(self):
        self.pdf_processor = None
        self.chunker = None
        self.vector_store = None
        self.qa_engine = None
        self.conversation_history = []
    
    def initialize_from_pdf(self, pdf_path: str, api_key: str) -> None:
        """
        Initialize the chatbot by processing a PDF file or loading existing index
        
        Args:
            pdf_path: Path to the PDF file
            api_key: Google AI API key
        """
        # Check if we have a previously saved index for this PDF
        index_path = "saved_index"
        pdf_hash = str(hash(pdf_path + str(os.path.getmtime(pdf_path))))[:10]
        versioned_index_path = f"{index_path}_{pdf_hash}"
        
        if os.path.exists(versioned_index_path):
            st.info("🚀 Found existing processed data - loading quickly...")
            try:
                self.initialize_from_index(versioned_index_path, api_key)
                st.success("✅ Loaded from previous processing - ready to use!")
                return
            except Exception as e:
                st.warning(f"⚠️ Could not load existing data, will reprocess: {str(e)}")
        
        st.info("🔄 Processing PDF file (first time or data changed)...")
        
        # Initialize adaptive config manager
        self.config_manager = AdaptiveConfigManager()
        
        # Initialize components
        self.pdf_processor = PDFProcessor()
        
        # Process PDF first to get document characteristics
        with st.spinner("Analyzing document characteristics..."):
            full_text, page_info = self.pdf_processor.process_pdf(pdf_path)
            doc_chars = self.config_manager.analyze_document_characteristics(full_text, page_info)
        
        st.success(f"✅ Analyzed document: {doc_chars.document_type} type, {len(page_info)} pages")
        
        # Get adaptive chunking configuration
        chunking_config = self.config_manager.get_adaptive_chunking_config()
        
        self.chunker = IntelligentChunker(
            min_chunk_size=chunking_config["min_chunk_size"],
            max_chunk_size=chunking_config["max_chunk_size"],
            overlap_size=chunking_config["overlap_size"],
            quality_threshold=chunking_config["quality_threshold"]
        )
        
        self.vector_store = HybridVectorStore(document_type=doc_chars.document_type)
        
        st.info(f"📊 Adaptive Configuration Applied:")
        st.write(f"- Chunk size: {chunking_config['min_chunk_size']}-{chunking_config['max_chunk_size']} chars")
        st.write(f"- Quality threshold: {chunking_config['quality_threshold']:.2f}")
        st.write(f"- Document type: {doc_chars.document_type}")
        st.write(f"- Language complexity: {doc_chars.language_complexity:.2f}")
        
        # Create chunks (already have full_text and page_info)
        with st.spinner("Creating adaptive semantic chunks..."):
            chunks = self.chunker.process_document(full_text, page_info)
        
        st.success(f"✅ Created {len(chunks)} adaptive semantic chunks")
        
        # Build vector store
        with st.spinner("Building vector store and embeddings..."):
            self.vector_store.add_chunks(chunks)
        
        st.success("✅ Vector store ready")
        
        # Initialize Q&A engine
        with st.spinner("Initializing Gemini Flash 2.0..."):
            try:
                print(f"🔧 Initializing Q&A engine with document type: {doc_chars.document_type}")
                print(f"🔧 API Key length: {len(api_key)}")
                print(f"🔧 Vector store chunks: {len(self.vector_store.chunks) if hasattr(self.vector_store, 'chunks') else 'unknown'}")
                
                # Test API key first
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                
                # Test model access
                model = genai.GenerativeModel('gemini-2.0-flash-exp')
                test_response = model.generate_content("Hello")
                print(f"✅ API key test successful: {test_response.text[:50]}...")
                
                self.qa_engine = GeminiQAEngine(
                    api_key=api_key, 
                    vector_store=self.vector_store
                )
                print("✅ Q&A engine initialized successfully")
            except Exception as e:
                print(f"❌ Error initializing Q&A engine: {str(e)}")
                print(f"❌ Error type: {type(e)}")
                import traceback
                print(f"❌ Traceback: {traceback.format_exc()}")
                st.error(f"Q&A Engine initialization failed: {str(e)}")
                raise e
        
        st.success("✅ Q&A engine ready")
        
        # Save the index for future use (optional)
        with st.spinner("Saving index for future use..."):
            try:
                self.vector_store.save_index(versioned_index_path)
                st.success(f"✅ Index saved for quick loading next time")
            except Exception as e:
                st.warning(f"⚠️ Could not save index (chatbot still works): {str(e)}")
                print(f"⚠️ Index save failed: {e}")
    
    def initialize_from_index(self, index_path: str, api_key: str) -> None:
        """
        Initialize the chatbot from a saved index
        
        Args:
            index_path: Path to the saved index
            api_key: Google AI API key
        """
        st.info("🔄 Loading saved index...")
        
        # Initialize adaptive config manager
        self.config_manager = AdaptiveConfigManager()
        
        # Initialize vector store and load index
        self.vector_store = HybridVectorStore()
        with st.spinner("Loading vector store..."):
            self.vector_store.load_index(index_path)
        
        st.success("✅ Vector store loaded")
        
        # Initialize Q&A engine
        with st.spinner("Initializing Gemini Flash 2.0..."):
            self.qa_engine = GeminiQAEngine(
                api_key=api_key, 
                vector_store=self.vector_store
            )
        
        st.success("✅ Q&A engine ready")
    
    def answer_question(self, query: str) -> dict:
        """
        Answer a question using the Q&A engine
        
        Args:
            query: User question
            
        Returns:
            Response dictionary
        """
        if not self.qa_engine:
            return {"error": "Q&A engine not initialized"}
        
        response = self.qa_engine.answer_question(
            query, 
            conversation_history=self.conversation_history
        )
        
        # Update conversation history
        self.conversation_history.append({
            "question": query,
            "answer": response.answer,
            "confidence": response.confidence
        })
        
        return {
            "answer": response.answer,
            "confidence": response.confidence,
            "sources": response.sources,
            "reasoning": response.reasoning,
            "follow_up_questions": response.follow_up_questions
        }


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="500-Page PDF Q&A Chatbot",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📚 PDF Q&A Chatbot")
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = QAChatbot()
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Google AI API Key",
            type="password",
            value=os.getenv("GOOGLE_API_KEY", ""),
            help="Enter your Google AI API key for Gemini Flash 2.0"
        )
        
        if not api_key:
            st.error("Please provide a Google AI API key")
            st.stop()
        
        st.header("📄 Document Setup")
        
        # Choice between PDF upload and saved index
        setup_option = st.radio(
            "Choose setup option:",
            ["Upload new PDF", "Load saved index"]
        )
        
        if setup_option == "Upload new PDF":
            # Quick option for OASIS E1 Manual
            st.subheader("🚀 Quick Start")
            oasis_path = "C:\\Users\\Jeeva\\Desktop\\Projects\\ya_labs_qa_chatbot\\draft-oasis-e1-manual-04-28-2024.pdf"
            
            if os.path.exists(oasis_path):
                if st.button("📄 Use OASIS E1 Manual", type="primary", disabled=st.session_state.initialized):
                    try:
                        st.session_state.chatbot.initialize_from_pdf(oasis_path, api_key)
                        st.session_state.initialized = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing OASIS E1 manual: {str(e)}")
                        st.error(f"Error type: {type(e)}")
                        import traceback
                        st.error(f"Full traceback: {traceback.format_exc()}")
            else:
                st.warning(f"OASIS E1 manual not found at: {oasis_path}")
            
            st.divider()
            
            # PDF file upload
            st.subheader("📤 Upload Custom PDF")
            uploaded_file = st.file_uploader(
                "Upload your 500-page PDF manual",
                type="pdf",
                help="Upload the PDF file you want to create a Q&A bot for"
            )
            
            if uploaded_file and not st.session_state.initialized:
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                try:
                    st.session_state.chatbot.initialize_from_pdf(temp_path, api_key)
                    st.session_state.initialized = True
                    os.remove(temp_path)  # Clean up
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        
        else:
            # Load from saved index
            index_path = st.text_input(
                "Index path",
                value="saved_index",
                help="Path to the saved vector store index"
            )
            
            if st.button("Load Index") and not st.session_state.initialized:
                if os.path.exists(index_path):
                    try:
                        st.session_state.chatbot.initialize_from_index(index_path, api_key)
                        st.session_state.initialized = True
                    except Exception as e:
                        st.error(f"Error loading index: {str(e)}")
                else:
                    st.error(f"Index path {index_path} does not exist")
        
        # Show initialization status
        if st.session_state.initialized:
            st.success("✅ Chatbot initialized and ready!")
        else:
            st.warning("⚠️ Please initialize the chatbot first")
    
    # Main chat interface
    if st.session_state.initialized:
        st.header("💬 Ask Questions About Your Manual")
        
        # Display conversation history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show additional info for assistant messages
                if message["role"] == "assistant" and "metadata" in message:
                    with st.expander("📊 Response Details"):
                        metadata = message["metadata"]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Confidence", f"{metadata['confidence']:.2f}")
                        with col2:
                            st.metric("Sources", len(metadata['sources']))
                        
                        if metadata['sources']:
                            st.subheader("📖 Sources")
                            for i, source in enumerate(metadata['sources']):
                                st.write(f"**Source {i+1}:** Page {source['page_number']} - {source['section_title']}")
                                st.write(f"**Relevance:** {source['relevance_score']:.3f} | **Type:** {source['chunk_type']}")
                                st.write(f"**Preview:** {source['preview']}")
                                st.write("---")
                        
                        if metadata['follow_up_questions']:
                            st.subheader("🤔 Suggested Follow-up Questions")
                            for i, question in enumerate(metadata['follow_up_questions']):
                                st.write(f"💡 {question}")
                            st.info("Copy and paste any question above into the chat box below.")
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your manual..."):
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": prompt
            })
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.chatbot.answer_question(prompt)
                
                if "error" in response:
                    st.error(response["error"])
                else:
                    st.markdown(response["answer"])
                    
                    # Add assistant message with metadata
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "metadata": response
                    })
        
        # Show clean interface when no messages
        if not st.session_state.messages:
            st.info("💬 Start asking questions about your manual in the chat box below.")
    
    else:
        # Show simple setup message
        st.info("👆 Please initialize the chatbot using the sidebar to start asking questions about your PDF manual.")


if __name__ == "__main__":
    # Check if running with command line arguments
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="500-Page PDF Q&A Chatbot")
        parser.add_argument("--pdf_path", help="Path to PDF file")
        parser.add_argument("--load_index", help="Path to saved index")
        parser.add_argument("--api_key", help="Google AI API key")
        
        args = parser.parse_args()
        
        # Set environment variable if provided
        if args.api_key:
            os.environ["GOOGLE_API_KEY"] = args.api_key
        
        # Run command line version
        chatbot = QAChatbot()
        
        if args.pdf_path:
            print("Processing PDF...")
            chatbot.initialize_from_pdf(args.pdf_path, args.api_key)
        elif args.load_index:
            print("Loading index...")
            chatbot.initialize_from_index(args.load_index, args.api_key)
        else:
            print("Please provide either --pdf_path or --load_index")
            sys.exit(1)
        
        # Simple command line interface
        print("\n" + "="*50)
        print("500-Page PDF Q&A Chatbot Ready!")
        print("Type 'quit' to exit")
        print("="*50 + "\n")
        
        while True:
            query = input("\n🤔 Your question: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if query:
                print("\n🤖 Thinking...")
                response = chatbot.answer_question(query)
                
                if "error" in response:
                    print(f"❌ Error: {response['error']}")
                else:
                    print(f"\n📝 Answer (Confidence: {response['confidence']:.2f}):")
                    print(response['answer'])
                    
                    if response['sources']:
                        print(f"\n📖 Sources ({len(response['sources'])} found):")
                        for i, source in enumerate(response['sources'][:3]):  # Show top 3
                            print(f"  {i+1}. Page {source['page_number']} - {source['section_title']}")
                    
                    if response['follow_up_questions']:
                        print(f"\n🤔 Follow-up suggestions:")
                        for q in response['follow_up_questions']:
                            print(f"  • {q}")
    
    else:
        # Run Streamlit interface
        main()
