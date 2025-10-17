import fitz  # PyMuPDF
import os
from typing import List, Dict, Union
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFLoader:
    """Class to load and process PDF documents."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """Initialize the PDF loader with chunking parameters.
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract all text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        doc = fitz.open(pdf_path)
        text = ""
        
        for page_num, page in enumerate(doc):
            text += page.get_text()
            
        return text
    
    def process_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        """Process a PDF file by extracting text and splitting into chunks.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of document chunks with metadata
        """
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        
        # Split into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create document objects with metadata
        documents = []
        filename = os.path.basename(pdf_path)
        
        for i, chunk in enumerate(chunks):
            documents.append({
                "text": chunk,
                "metadata": {
                    "source": filename,
                    "chunk_id": i,
                    "path": pdf_path
                }
            })
            
        return documents
    
    def process_directory(self, directory_path: str) -> List[Dict[str, str]]:
        """Process all PDF files in a directory.
        
        Args:
            directory_path: Path to directory containing PDF files
            
        Returns:
            List of document chunks with metadata from all PDFs
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
            
        all_documents = []
        
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(directory_path, filename)
                documents = self.process_pdf(pdf_path)
                all_documents.extend(documents)
                
        return all_documents