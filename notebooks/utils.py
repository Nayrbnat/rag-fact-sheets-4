"""
Utility functions for notebook operations including PDF extraction and chunking
"""
import os
import json
import nltk
from typing import List, Dict, Any, Optional
import logging

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

from nltk.tokenize import sent_tokenize
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

def extract_text_from_pdf(
    pdf_path: str, 
    strategy: str = "fast", 
    extract_images: bool = False,
    infer_table_structure: bool = True,
    chunking_strategy: str = "by_element",
    languages: str = "eng",
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Extract text from a PDF file using the unstructured library.
    
    This function uses the unstructured library to extract text and metadata
    from a PDF file. The extracted elements include text, page numbers,
    paragraph information, and other metadata that can be used for document analysis.
    
    Args:
        pdf_path: Path to the PDF file
        strategy: Strategy for extraction:
            - "fast": Fast extraction without OCR (default)
            - "ocr_only": Use only OCR for text extraction
            - "auto": Automatically determine best extraction method
        extract_images: Whether to extract images from the PDF
        infer_table_structure: Whether to infer table structure in the PDF
        chunking_strategy: Strategy for chunking ("by_element" or other options)
        languages: Languages to use for OCR (if applicable)
        **kwargs: Additional parameters to pass to partition_pdf
        
    Returns:
        List of extracted elements with their metadata
    """
    logger.info(f"Extracting text from PDF: {pdf_path}")

    # Validate and map strategy parameter
    # Valid strategies for partition_pdf are "fast", "ocr_only", and "auto"
    valid_strategies = ["fast", "ocr_only", "auto"]
                
    if strategy not in valid_strategies:
        logger.warning(f"Invalid strategy '{strategy}'. Using 'fast' instead.")
        strategy = "fast"

    try:
        # Get the filename for metadata
        filename = os.path.basename(pdf_path)
        languages_list = [lang.strip() for lang in languages.split(',')] if languages else ['eng']
        # Extract elements from the PDF
        elements = partition_pdf(
            filename=pdf_path, 
            strategy=strategy,
            extract_images_in_pdf=extract_images,
            infer_table_structure=infer_table_structure,
            chunking_strategy=chunking_strategy,
            languages=languages_list,  # Updated from ocr_languages to languages
            **kwargs  # Pass additional parameters to partition_pdf
        )
        
        logger.info(f"Extracted {len(elements)} elements from {pdf_path} using strategy '{strategy}'")
        
        # Initialize paragraph tracking
        paragraphs_by_page = {}  # Dict to track paragraph numbers by page
        last_element_type = None
        last_page_number = None
        global_paragraph_count = 0  # Track paragraphs across the entire document
        
        # Convert unstructured elements to a more usable dictionary format
        result = []
        for i, element in enumerate(elements):
            # Skip empty elements
            if not hasattr(element, 'text') or not element.text.strip():
                continue
                
            # Create a dictionary with element information
            element_dict = {
                "id": f"element_{i}",
                "type": element.category if hasattr(element, 'category') else type(element).__name__,
                "text": element.text if hasattr(element, 'text') else str(element),
                "metadata": {
                    "filename": filename
                }
            }
            
            # Get page number if available
            page_number = None
            if hasattr(element, "metadata") and hasattr(element.metadata, "page_number"):
                page_number = element.metadata.page_number
            elif hasattr(element, "page_number"):
                page_number = element.page_number
            else:
                page_number = 0
                
            element_dict["metadata"]["page_number"] = page_number
            
            # Initialize or get paragraph counter for this page
            if page_number not in paragraphs_by_page:
                paragraphs_by_page[page_number] = 0
                
            # Determine if this is a new paragraph based on element type and context
            is_new_paragraph = False
            if page_number != last_page_number:  # New page = new paragraph
                is_new_paragraph = True
            elif element.category != last_element_type if hasattr(element, 'category') else True:  # Type change = new paragraph
                is_new_paragraph = True
            elif (hasattr(element, 'category') and 
                  element.category == "NarrativeText" and last_element_type in ["Title", "ListItem"]):
                is_new_paragraph = True
                
            if is_new_paragraph:
                paragraphs_by_page[page_number] += 1
                global_paragraph_count += 1  # Increment global paragraph counter
                
            # Update tracking variables
            last_element_type = element.category if hasattr(element, 'category') else None
            last_page_number = page_number
            
            # Add paragraph information to metadata
            paragraph_id = f"p{page_number}_para{paragraphs_by_page[page_number]}"
            element_dict["metadata"]["paragraph_id"] = paragraph_id
            element_dict["metadata"]["paragraph_number"] = paragraphs_by_page[page_number]  # Add explicit paragraph number
            element_dict["metadata"]["global_paragraph_number"] = global_paragraph_count  # Add global paragraph number
            
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

def extract_text_from_docx(docx_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from a DOCX file.
    
    Args:
        docx_path: Path to the DOCX file
        
    Returns:
        A list of dictionaries with text and metadata
    """
    try:
        # Only import docx when needed
        from docx import Document
        doc = Document(docx_path)
    except ImportError:
        logger.error("python-docx package is not installed or is misconfigured. Please install it with: pip install python-docx")
        return [{'text': f"ERROR: Could not extract text from {docx_path}. python-docx package is not installed.", 'type': 'error'}]
    except Exception as e:
        logger.error(f"Error opening DOCX file {docx_path}: {str(e)}")
        return [{'text': f"ERROR: Could not extract text from {docx_path}. {str(e)}", 'type': 'error'}]
    
    # Get the filename for metadata
    filename = os.path.basename(docx_path)
    elements = []
    page_number = 1  # DOCX doesn't have a concept of pages, so we simulate it
    paragraph_count = 0
    
    for i, paragraph in enumerate(doc.paragraphs):
        text = paragraph.text.strip()
        if text:
            paragraph_count += 1
            # Every 20 paragraphs, we simulate a new page
            if paragraph_count > 20:
                page_number += 1
                paragraph_count = 1
                
            # Determine element type based on formatting
            element_type = "paragraph"
            if paragraph.style.name.startswith("Heading"):
                element_type = "Heading"
            elif paragraph.style.name == "Title":
                element_type = "Title"
                
            elements.append({
                'id': f"element_{i}",
                'type': element_type,
                'text': text,
                'metadata': {
                    'page_number': page_number,
                    'filename': filename,
                    'paragraph_id': f"p{page_number}_para{paragraph_count}",
                    'style': paragraph.style.name
                }
            })
    
    logger.info(f"Extracted {len(elements)} paragraphs from {docx_path}")
    return elements

def chunk_document_by_sentences(elements: list, max_chunk_size: int = 512, overlap: int = 2) -> list:
    """
    Chunk a document into sentences with context for better semantic meaning.
    
    Args:
        elements: List of document elements from unstructured
        max_chunk_size: Maximum size of each chunk in characters (approx)
        overlap: Number of sentences to overlap between chunks
        
    Returns:
        List of chunks with their metadata (page number, etc.)
    """
    # Initialize variables
    chunks = []
    current_chunk_text = ""
    current_chunk_sentences = []
    current_chunk_metadata = {}
    chunk_id = 0
    current_element_types = set()
    current_paragraph_numbers = set()
    
    # Process each element
    for element in elements:
        element_type = element.get('type', 'Text')
        element_text = element.get('text', '')
        element_metadata = element.get('metadata', {})
        
        # Skip empty text
        if not element_text.strip():
            continue
            
        # Special handling for titles, headings, etc. - keep them as standalone chunks
        if element_type in ['Title', 'Heading', 'Header', 'SubHeading']:
            # First finish any in-progress chunk
            if current_chunk_sentences:
                chunks.append({
                    "id": f"chunk_{chunk_id}",
                    "text": current_chunk_text.strip(),
                    "sentences": current_chunk_sentences,
                    "metadata": current_chunk_metadata
                })
                chunk_id += 1
                
            # Add the title/heading as its own chunk
            chunks.append({
                "id": f"chunk_{chunk_id}",
                "text": element_text.strip(),
                "sentences": [element_text.strip()],
                "metadata": {
                    "element_types": [element_type],
                    "page_number": element_metadata.get('page_number', 0),
                    "paragraph_number": element_metadata.get('paragraph_number'),
                    "global_paragraph_number": element_metadata.get('global_paragraph_number'),
                    "paragraph_id": element_metadata.get('paragraph_id'),
                    "filename": element_metadata.get('filename', ''),
                    "country": element_metadata.get('country', ''),
                    "document_title": element_metadata.get('document_title', ''),
                    "submission_date": element_metadata.get('submission_date', '')
                }
            })
            chunk_id += 1
            
            # Reset tracking variables
            current_chunk_text = ""
            current_chunk_sentences = []
            current_chunk_metadata = {}
            current_element_types = set()
            current_paragraph_numbers = set()
            continue
        
        # For regular text elements, split into sentences
        sentences = sent_tokenize(element_text)
        
        # Start a new chunk if we don't have one yet
        if not current_chunk_sentences:
            current_chunk_metadata = {
                "element_types": [element_type],
                "page_number": element_metadata.get('page_number', 0),
                "paragraph_numbers": [element_metadata.get('paragraph_number')] if element_metadata.get('paragraph_number') is not None else [],
                "paragraph_ids": [element_metadata.get('paragraph_id')] if element_metadata.get('paragraph_id') is not None else [],
                "global_paragraph_numbers": [element_metadata.get('global_paragraph_number')] if element_metadata.get('global_paragraph_number') is not None else [],
                "filename": element_metadata.get('filename', ''),
                "country": element_metadata.get('country', ''),
                "document_title": element_metadata.get('document_title', ''),
                "submission_date": element_metadata.get('submission_date', '')
            }
            current_element_types = {element_type}
            if element_metadata.get('paragraph_number') is not None:
                current_paragraph_numbers = {element_metadata.get('paragraph_number')}
        else:
            # Track element types and paragraph numbers
            current_element_types.add(element_type)
            if element_metadata.get('paragraph_number') is not None:
                current_paragraph_numbers.add(element_metadata.get('paragraph_number'))
                current_chunk_metadata['paragraph_numbers'].append(element_metadata.get('paragraph_number'))
            
            if element_metadata.get('paragraph_id') is not None and element_metadata.get('paragraph_id') not in current_chunk_metadata['paragraph_ids']:
                current_chunk_metadata['paragraph_ids'].append(element_metadata.get('paragraph_id'))
                
            if element_metadata.get('global_paragraph_number') is not None and element_metadata.get('global_paragraph_number') not in current_chunk_metadata['global_paragraph_numbers']:
                current_chunk_metadata['global_paragraph_numbers'].append(element_metadata.get('global_paragraph_number'))
                
            # Add element type if it's a new one
            if element_type not in current_chunk_metadata['element_types']:
                current_chunk_metadata['element_types'].append(element_type)
            
            # Update country if not set yet but available in this element
            if not current_chunk_metadata.get('country') and element_metadata.get('country'):
                current_chunk_metadata['country'] = element_metadata.get('country')
                
            # Update document_title if not set yet but available in this element
            if not current_chunk_metadata.get('document_title') and element_metadata.get('document_title'):
                current_chunk_metadata['document_title'] = element_metadata.get('document_title')
                
            # Update submission_date if not set yet but available in this element
            if not current_chunk_metadata.get('submission_date') and element_metadata.get('submission_date'):
                current_chunk_metadata['submission_date'] = element_metadata.get('submission_date')
        
        # Process each sentence
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Add to current chunk
            if current_chunk_text:
                current_chunk_text += " " + sentence
            else:
                current_chunk_text = sentence
                
            current_chunk_sentences.append(sentence)
            
            # Check if we should start a new chunk
            if len(current_chunk_text) >= max_chunk_size and len(current_chunk_sentences) > overlap + 1:
                # Create a chunk with the accumulated text
                chunks.append({
                    "id": f"chunk_{chunk_id}",
                    "text": current_chunk_text.strip(),
                    "sentences": current_chunk_sentences,
                    "metadata": current_chunk_metadata
                })
                chunk_id += 1
                
                # Start new chunk with overlap
                current_chunk_sentences = current_chunk_sentences[-overlap:] if overlap > 0 else []
                current_chunk_text = " ".join(current_chunk_sentences)
                # Keep the same metadata since we're still in the same element
    
    # Add the last chunk if it has content
    if current_chunk_sentences:
        chunks.append({
            "id": f"chunk_{chunk_id}",
            "text": current_chunk_text.strip(),
            "sentences": current_chunk_sentences,
            "metadata": current_chunk_metadata
        })
    
    return chunks