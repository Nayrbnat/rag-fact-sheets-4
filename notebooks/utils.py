"""
Utility functions for the climate policy extractor.
"""
import os
import logging

from unstructured.partition.pdf import partition_pdf

# Set up logger
logger = logging.getLogger(__name__)


def get_file_size(file_path: str) -> int:
    """
    Get the size of a file in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Size of the file in bytes
    """
    return os.path.getsize(file_path)


def extract_text_data_from_pdf(pdf_path: str, unstructured_strategy: str = "fast"):
    """
    Extract text from a PDF file using the unstructured library.
    
    This function uses the unstructured library to extract text and metadata
    from a PDF file. The extracted elements include text, page numbers,
    and other metadata that can be used for document analysis.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of extracted elements with their metadata
    """
    
    # TODO: This is up to you to decide if you keep as is or if you want to make changes
    logger.info(f"Extracting text from PDF: {pdf_path}")
    logger.warning("Using starter code. Either edit this function or remove this warning if you are happy with the current implementation.")


    # Extract elements from the PDF
    try:
        # Use unstructured to extract elements from the PDF
        elements = partition_pdf(filename=pdf_path, strategy=unstructured_strategy)
        
        logger.info(f"Extracted {len(elements)} elements from {pdf_path}")
        
        # Convert unstructured elements to a more usable dictionary format
        result = []
        for i, element in enumerate(elements):
            # Skip empty elements
            if not hasattr(element, 'text') or not element.text.strip():
                continue
                
            # Create a dictionary with element information
            element_dict = {
                "id": i,
                "type": element.category,
                "text": element.text,
                "metadata": {}
            }
            
            # Add page number if available
            if hasattr(element, "metadata") and hasattr(element.metadata, "page_number"):
                element_dict["metadata"]["page_number"] = element.metadata.page_number
            
            # Add coordinates if available
            if hasattr(element, "coordinates"):
                element_dict["metadata"]["coordinates"] = element.coordinates
                
            # Add any other metadata that might be useful
            if hasattr(element, "metadata"):
                for key, value in vars(element.metadata).items():
                    if key not in ["page_number"]:  # Skip already added metadata
                        element_dict["metadata"][key] = value
                
            result.append(element_dict)
        
        logger.info(f"Processed {len(result)} non-empty elements from {pdf_path}")
        return result
        
    except Exception as e:
        logger.error(f"Unable to extract text from PDF {pdf_path}: {e}")
        return []


def chunk_document(elements: list, chunk_size: int = 1000, overlap: int = 200) -> list:
    """
    Chunk a document into smaller pieces for processing.
    
    This is a template function that you should implement.
    The goal is to take the elements extracted by unstructured and group them
    into chunks suitable for embedding and retrieval.
    
    Args:
        elements: List of document elements from unstructured
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of chunks with their metadata (page number, etc.)
    """
    logger.warning("Using starter code. Either edit this function or remove this warning if you are happy with the current implementation.")
    
    # Example of a very simple chunking approach:
    chunks = [{"line_number": i + 1, "text": element["text"]} for i, element in enumerate(elements)]
    
    logger.info(f"Created {len(chunks)} chunks (placeholder implementation)")
    return chunks