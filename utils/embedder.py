import numpy as np
from typing import List, Dict, Union, Optional
import os
import pickle
import google.generativeai as genai

class Embedder:
    """Class to generate embeddings for text chunks using Google's Embedding API."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "embedding-001"):
        """Initialize the embedder with Google's embedding API.
        
        Args:
            api_key: Google API key
            model_name: Name of the embedding model
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
            
        self.model_name = model_name
        # Configure Google's API
        genai.configure(api_key=self.api_key)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            NumPy array of embeddings
        """
        if not texts:
            return np.array([])
        
        # Use Google's embedding model
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model=f"models/{self.model_name}",
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result["embedding"])
            
        return np.array(embeddings)
    
    def embed_documents(self, documents: List[Dict[str, str]]) -> Dict[str, Union[List[Dict[str, str]], np.ndarray]]:
        """Generate embeddings for a list of document chunks.
        
        Args:
            documents: List of document dictionaries with 'text' and 'metadata' keys
            
        Returns:
            Dictionary containing the documents and their embeddings
        """
        if not documents:
            return {"documents": [], "embeddings": np.array([])}
            
        texts = [doc["text"] for doc in documents]
        embeddings = self.embed_texts(texts)
        
        return {
            "documents": documents,
            "embeddings": embeddings
        }
    
    def save_embeddings(self, documents: List[Dict[str, str]], embeddings: np.ndarray, 
                         output_dir: str, filename: str = "embeddings") -> str:
        """Save documents and their embeddings to disk.
        
        Args:
            documents: List of document dictionaries
            embeddings: NumPy array of embeddings
            output_dir: Directory to save files
            filename: Base filename for the saved files
            
        Returns:
            Path to the saved embeddings file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save documents and embeddings
        data = {
            "documents": documents,
            "embeddings": embeddings
        }
        
        output_path = os.path.join(output_dir, f"{filename}.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(data, f)
            
        return output_path
    
    def load_embeddings(self, file_path: str) -> Dict[str, Union[List[Dict[str, str]], np.ndarray]]:
        """Load documents and embeddings from disk.
        
        Args:
            file_path: Path to the embeddings file
            
        Returns:
            Dictionary containing the documents and their embeddings
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Embeddings file not found: {file_path}")
            
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            
        return data