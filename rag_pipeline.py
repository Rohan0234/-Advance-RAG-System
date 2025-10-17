import os
import numpy as np
from typing import List, Dict, Union, Optional
from utils.pdf_loader import PDFLoader
from utils.embedder import Embedder
from utils.vector_store import VectorStore
from utils.llm_wrapper import LLMWrapper

class RAGPipeline:
    """Main class for the Retrieval Augmented Generation pipeline."""
    
    def __init__(self, 
                 embedding_model: str = "embedding-001",
                 llm_model: str = "gemini-2.0-flash",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 top_k: int = 5,
                 api_key: Optional[str] = None):
        """Initialize the RAG pipeline.
        
        Args:
            embedding_model: Name of the embedding model
            llm_model: Name of the LLM model (default: gemini-pro)
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            top_k: Number of documents to retrieve
            api_key: Google API key
        """
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.api_key = api_key
        
        # Initialize components
        self.pdf_loader = PDFLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = Embedder(api_key=api_key, model_name=embedding_model)
        self.vector_store = VectorStore()
        self.llm = LLMWrapper(api_key=api_key, model=llm_model)
        
    def ingest_documents(self, input_dir: str, output_dir: str = "embeddings") -> None:
        """Process documents from a directory and create a vector store.
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory to save embeddings and index
        """
        # Process PDFs to get document chunks
        documents = self.pdf_loader.process_directory(input_dir)
        print(f"Processed {len(documents)} document chunks from {input_dir}")
        
        # Generate embeddings
        result = self.embedder.embed_documents(documents)
        documents = result["documents"]
        embeddings = result["embeddings"]
        print(f"Generated embeddings with shape {embeddings.shape}")
        
        # Save embeddings
        os.makedirs(output_dir, exist_ok=True)
        embed_path = self.embedder.save_embeddings(documents, embeddings, output_dir)
        print(f"Saved embeddings to {embed_path}")
        
        # Create vector store
        self.vector_store.create_index(documents, embeddings)
        index_path, docs_path = self.vector_store.save(output_dir)
        print(f"Saved vector store index to {index_path} and documents to {docs_path}")
    
    def load_vector_store(self, index_path: str, docs_path: str) -> None:
        """Load a previously created vector store.
        
        Args:
            index_path: Path to the FAISS index file
            docs_path: Path to the documents file
        """
        self.vector_store.load(index_path, docs_path)
        print(f"Loaded vector store from {index_path} and {docs_path}")
    
    def answer_query(self, query: str) -> Dict[str, Union[str, List[Dict[str, Union[Dict[str, str], float]]]]]:
        """Answer a query using the RAG pipeline.
        
        Args:
            query: User query
            
        Returns:
            Dictionary containing the answer and retrieved documents
        """
        # Embed the query
        query_embedding = self.embedder.embed_texts([query])
        
        # Retrieve relevant documents
        retrieved_docs = self.vector_store.search(query_embedding, k=self.top_k)
        
        # Generate answer
        answer = self.llm.generate_response(query, retrieved_docs)
        
        return {
            "query": query,
            "answer": answer,
            "retrieved_documents": retrieved_docs
        }