import os
import faiss
import numpy as np
import pickle
from typing import List, Dict, Union, Tuple, Optional

class VectorStore:
    """Class to handle vector storage and retrieval using FAISS."""
    
    def __init__(self, embedding_dim: Optional[int] = None):
        """Initialize the vector store.
        
        Args:
            embedding_dim: Dimension of the embeddings (can be set later)
        """
        self.index = None
        self.documents = []
        self.embedding_dim = embedding_dim
        if embedding_dim is not None:
            self.index = faiss.IndexFlatL2(embedding_dim)
    
    def create_index(self, documents: List[Dict[str, str]], embeddings: np.ndarray) -> None:
        """Create a FAISS index from documents and their embeddings.
        
        Args:
            documents: List of document dictionaries
            embeddings: NumPy array of embeddings
        """
        if embeddings.size == 0:
            raise ValueError("Cannot create index from empty embeddings")
            
        embedding_dim = embeddings.shape[1]
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.index.add(embeddings.astype(np.float32))
        self.documents = documents
    
    def save(self, output_dir: str, filename: str = "vector_store") -> Tuple[str, str]:
        """Save the vector store to disk.
        
        Args:
            output_dir: Directory to save files
            filename: Base filename for the saved files
            
        Returns:
            Tuple of paths to the saved index and documents
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(output_dir, f"{filename}.index")
        faiss.write_index(self.index, index_path)
        
        # Save documents
        docs_path = os.path.join(output_dir, f"{filename}_docs.pkl")
        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)
            
        return index_path, docs_path
    
    def load(self, index_path: str, docs_path: str) -> None:
        """Load the vector store from disk.
        
        Args:
            index_path: Path to the FAISS index file
            docs_path: Path to the documents file
        """
        if not os.path.exists(index_path) or not os.path.exists(docs_path):
            raise FileNotFoundError(f"Index or documents file not found")
            
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load documents
        with open(docs_path, "rb") as f:
            self.documents = pickle.load(f)
            
        # Update embedding dimension
        self.embedding_dim = self.index.d
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Union[Dict[str, str], float]]]:
        """Search for similar documents using the query embedding.
        
        Args:
            query_embedding: Embedding of the query
            k: Number of top results to return
            
        Returns:
            List of dictionaries containing document and similarity score
        """
        if self.index is None:
            raise ValueError("Index not initialized")
            
        # Reshape query embedding if necessary
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # Query the index
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents) and idx >= 0:
                results.append({
                    "document": self.documents[idx],
                    "score": float(distances[0][i])
                })
                
        return results