import os
from typing import List, Dict, Union, Optional
import google.generativeai as genai

class LLMWrapper:
    """Wrapper for LLM API calls using Google Gemini."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        """Initialize the LLM wrapper.
        
        Args:
            api_key: Google Gemini API key (will use env var GOOGLE_API_KEY if None)
            model: Model name to use (default: gemini-pro)
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
            
        self.model = model
        # Configure the Google Gemini API
        genai.configure(api_key=self.api_key)
    
    def generate_response(self, query: str, context: List[Dict[str, Union[Dict[str, str], float]]]) -> str:
        """Generate a response based on the query and retrieved context.
        
        Args:
            query: User query
            context: List of retrieved document chunks with metadata
            
        Returns:
            Generated response
        """
        # Format context for the prompt
        formatted_context = ""
        for i, item in enumerate(context):
            doc = item["document"]
            formatted_context += f"Document {i+1} [Source: {doc['metadata']['source']}]:\n{doc['text']}\n\n"
        
        # Create system prompt
        system_prompt = (
            "You are a helpful assistant that answers questions based on the provided documents. "
            "Use only the information in the documents to answer the question. "
            "If you cannot find the answer in the documents, say so clearly. "
            "Always cite the source document when providing information."
        )
        
        # Create user prompt
        user_prompt = f"{system_prompt}\n\nContext documents:\n\n{formatted_context}\n\nQuestion: {query}"
        
        # Generate response
        generation_config = {
            "temperature": 0.3,
            "max_output_tokens": 1000,
            "top_p": 0.95,
            "top_k": 40
        }
        
        # Initialize the model
        model = genai.GenerativeModel(model_name=self.model,
                                     generation_config=generation_config)
        
        # Generate response
        response = model.generate_content(user_prompt)
        
        # Return the generated text
        return response.text
    
    def generate_without_context(self, query: str) -> str:
        """Generate a response without context for testing purposes.
        
        Args:
            query: User query
            
        Returns:
            Generated response
        """
        generation_config = {
            "temperature": 0.7,
            "max_output_tokens": 500,
            "top_p": 0.95,
            "top_k": 40
        }
        
        # Initialize the model
        model = genai.GenerativeModel(model_name=self.model,
                                     generation_config=generation_config)
        
        # Generate response
        response = model.generate_content(query)
        
        # Return the generated text
        return response.text