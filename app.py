import os
import streamlit as st
from rag_pipeline import RAGPipeline
import tempfile
import shutil
    
# Set page configuration
st.set_page_config(page_title="PDF RAG System", page_icon="📚", layout="wide")

# Initialize session state
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()

# Create directories if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("embeddings", exist_ok=True)

def initialize_pipeline():
    """Initialize the RAG pipeline with user settings."""
    api_key = st.session_state.get("api_key", "")
    embedding_model = st.session_state.get("embedding_model", "embedding-001")
    llm_model = st.session_state.get("llm_model", "gemini-2.0-flash")
    chunk_size = st.session_state.get("chunk_size", 500)
    chunk_overlap = st.session_state.get("chunk_overlap", 50)
    top_k = st.session_state.get("top_k", 5)
    
    try:
        st.session_state.rag_pipeline = RAGPipeline(
            embedding_model=embedding_model,
            llm_model=llm_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            api_key=api_key
        )
        return True
    except Exception as e:
        st.error(f"Error initializing RAG pipeline: {str(e)}")
        return False

# Title and description
st.title("📚 PDF RAG System")
st.markdown("""
This application uses Retrieval Augmented Generation (RAG) to answer questions based on your PDF documents.
Upload PDF files, process them, and ask questions to get answers derived from your documents.
""")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Key input
    api_key = st.text_input("Google API Key", type="password", 
                            help="Enter your Google API key. It's required for generating answers with Gemini.")
    if api_key:
        st.session_state.api_key = api_key
    
    # Model settings
    st.subheader("Model Settings")
    
    embedding_model = st.selectbox(
        "Embedding Model",
        ["embedding-001"],
        help="Google embedding model to use."
    )
    st.session_state.embedding_model = embedding_model
    
    llm_model = st.selectbox(
        "LLM Model",
        ["gemini-2.0-flash"],
        help="Select the Google LLM model for generating answers."
    )
    st.session_state.llm_model = llm_model
    
    # Chunking settings
    st.subheader("Chunking Settings")
    
    chunk_size = st.slider(
        "Chunk Size",
        min_value=100, 
        max_value=1000, 
        value=500, 
        step=50,
        help="Size of document chunks in characters."
    )
    st.session_state.chunk_size = chunk_size
    
    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0, 
        max_value=200, 
        value=50, 
        step=10,
        help="Overlap between consecutive chunks in characters."
    )
    st.session_state.chunk_overlap = chunk_overlap
    
    # Retrieval settings
    st.subheader("Retrieval Settings")
    
    top_k = st.slider(
        "Top K Documents",
        min_value=1, 
        max_value=10, 
        value=5, 
        step=1,
        help="Number of documents to retrieve for each query."
    )
    st.session_state.top_k = top_k
    
    # Initialize button
    if st.button("Initialize Pipeline"):
        if initialize_pipeline():
            st.success("RAG pipeline initialized successfully!")
        else:
            st.error("Failed to initialize RAG pipeline. Check your settings.")

# Main area with tabs
tab1, tab2, tab3 = st.tabs(["Upload & Process", "Ask Questions", "View Documents"])

# Tab 1: Upload and Process
with tab1:
    st.header("1️⃣ Upload PDF Documents")
    
    uploaded_files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} PDF files.")
        
        # Save uploaded files to temporary directory
        with st.spinner("Saving uploaded files..."):
            # Clear existing files
            if os.path.exists(st.session_state.temp_dir):
                shutil.rmtree(st.session_state.temp_dir)
            os.makedirs(st.session_state.temp_dir, exist_ok=True)
            
            # Save new files
            for uploaded_file in uploaded_files:
                file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
        
        st.success(f"Files saved to temporary directory.")
        
        # Process documents button
        if st.button("Process Documents"):
            if not st.session_state.rag_pipeline:
                if not initialize_pipeline():
                    st.error("Please initialize the RAG pipeline first.")
                    st.stop()
            
            with st.spinner("Processing documents... This may take a while."):
                try:
                    st.session_state.rag_pipeline.ingest_documents(
                        st.session_state.temp_dir, 
                        output_dir="embeddings"
                    )
                    st.session_state.documents_loaded = True
                    st.success("Documents processed successfully!")
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")

# Tab 2: Ask Questions
with tab2:
    st.header("2️⃣ Ask Questions")
    
    if not st.session_state.documents_loaded:
        st.warning("Please upload and process documents first.")
    else:
        query = st.text_input("Enter your question:")
        
        if st.button("Ask") and query:
            with st.spinner("Generating answer..."):
                try:
                    result = st.session_state.rag_pipeline.answer_query(query)
                    
                    st.subheader("Answer")
                    st.write(result["answer"])
                    
                    with st.expander("View retrieved documents"):
                        for i, doc in enumerate(result["retrieved_documents"]):
                            st.markdown(f"**Document {i+1}** (Score: {doc['score']:.4f})")
                            st.write(f"Source: {doc['document']['metadata']['source']}")
                            st.text(doc['document']['text'])
                            st.divider()
                            
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")

# Tab 3: View Documents
with tab3:
    st.header("3️⃣ View Processed Documents")
    
    if not st.session_state.documents_loaded:
        st.warning("Please upload and process documents first.")
    else:
        # Load index and documents
        index_path = os.path.join("embeddings", "vector_store.index")
        docs_path = os.path.join("embeddings", "vector_store_docs.pkl")
        
        if os.path.exists(index_path) and os.path.exists(docs_path):
            if not st.session_state.rag_pipeline:
                if not initialize_pipeline():
                    st.error("Please initialize the RAG pipeline first.")
                    st.stop()
                    
            st.session_state.rag_pipeline.load_vector_store(index_path, docs_path)
            
            # Show document sources
            sources = set()
            for doc in st.session_state.rag_pipeline.vector_store.documents:
                sources.add(doc["metadata"]["source"])
            
            st.subheader("Processed Documents")
            st.write(f"Total chunks: {len(st.session_state.rag_pipeline.vector_store.documents)}")
            st.write(f"Document sources: {', '.join(sources)}")
            
            # Show sample chunks
            with st.expander("View sample chunks"):
                max_samples = min(5, len(st.session_state.rag_pipeline.vector_store.documents))
                for i in range(max_samples):
                    doc = st.session_state.rag_pipeline.vector_store.documents[i]
                    st.markdown(f"**Chunk {i+1}** from {doc['metadata']['source']}")
                    st.text(doc['text'][:300] + "..." if len(doc['text']) > 300 else doc['text'])
                    st.divider()
        else:
            st.error("No processed documents found. Please process documents first.")

# Footer
st.markdown("---")
st.caption("PDF RAG System | Powered by FAISS and Google Gemini")