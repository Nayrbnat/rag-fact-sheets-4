"""
Populate the database with document chunks and their embeddings.
This script handles language detection and embedding generation.
"""
import os
import sys
import json
import re
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModel
from datetime import datetime  # Add this import
from tqdm import tqdm  # Import tqdm for progress bars

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from climate_policy_extractor.logging import get_logger

logger = get_logger(__name__)

# Define paths for models
MODELS_DIR = './local_models'
ENGLISH_MODEL_DIR = os.path.join(MODELS_DIR, 'distilroberta-base')
MULTILINGUAL_MODEL_DIR = os.path.join(MODELS_DIR, 'xlm-roberta-base')

# Define language code mappings for common countries and languages
COUNTRY_LANG_MAP = {
    'france': 'fr',
    'french': 'fr',
    'germany': 'de',
    'german': 'de',
    'spain': 'es',
    'spanish': 'es',
    'italy': 'it',
    'italian': 'it',
    'china': 'zh',
    'chinese': 'zh',
    'japan': 'ja',
    'japanese': 'ja',
    'russia': 'ru',
    'russian': 'ru',
    'brazil': 'pt',
    'portuguese': 'pt',
    'india': 'en',  # India often uses English for official documents
    'usa': 'en',
    'uk': 'en',
    'english': 'en',
    'mexico': 'es',
    'argentina': 'es',
    'colombia': 'es',
    'peru': 'es',
    'chile': 'es',
    'venezuela': 'es',
}

# Function to get language from filename or metadata
def determine_language(filename, metadata=None):
    """
    Determine the language from the filename or metadata.
    Falls back to language detection only when needed.
    
    Args:
        filename (str): The filename that might contain language information
        metadata (dict): Optional metadata that might contain language information
        
    Returns:
        str: Two-letter language code (defaults to 'en' when uncertain)
    """
    # First check if metadata already contains language information
    if metadata and 'language' in metadata and metadata['language']:
        return metadata['language'].lower()
        
    # Check for language code in filename (like "document_fr.pdf" or "fr_document.pdf")
    lang_pattern = r'[_\-\s\.](fr|es|de|it|zh|ja|ru|pt|en)[_\-\s\.]'
    lang_match = re.search(lang_pattern, filename.lower())
    if lang_match:
        return lang_match.group(1).lower()
        
    # Look for country names in filename or metadata that can indicate language
    if metadata and 'country' in metadata and metadata['country']:
        country = metadata['country'].lower()
        for country_key, lang_code in COUNTRY_LANG_MAP.items():
            if country_key in country:
                return lang_code
                
    # Check if filename contains country names
    for country_key, lang_code in COUNTRY_LANG_MAP.items():
        if country_key in filename.lower():
            return lang_code
            
    # If we still don't have a language, try to detect it from content
    # but we'll do this in the process_document_chunks function
    # to avoid circular imports
    
    # Default to English if we can't determine language
    return 'en'

def load_english_model():
    """
    Load the English language model (ClimateBERT) from local directory.
    If not available, attempt to download it from Hugging Face.
    Returns tokenizer and model as a tuple.
    """
    logger.info(f"Loading English model from {ENGLISH_MODEL_DIR}")
    try:
        # Check if local model exists
        if not os.path.exists(ENGLISH_MODEL_DIR):
            logger.warning(f"Local English model not found at {ENGLISH_MODEL_DIR}, attempting to download...")
            
            # Ensure the parent directory exists
            os.makedirs(os.path.dirname(ENGLISH_MODEL_DIR), exist_ok=True)
            
            # Download model from Hugging Face
            model_name = "distilroberta-base"
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=ENGLISH_MODEL_DIR)
            model = AutoModel.from_pretrained(model_name, cache_dir=ENGLISH_MODEL_DIR)
            
            logger.info(f"Successfully downloaded English model to {ENGLISH_MODEL_DIR}")
        else:
            # Load from local directory
            tokenizer = AutoTokenizer.from_pretrained(ENGLISH_MODEL_DIR, local_files_only=True)
            model = AutoModel.from_pretrained(ENGLISH_MODEL_DIR, local_files_only=True)
            
        return tokenizer, model
    except Exception as e:
        logger.error(f"Could not load or download English model: {e}")
        raise

def load_multilingual_model():
    """
    Load the multilingual model (XLM-RoBERTa) from local directory.
    If not available, attempt to download it from Hugging Face.
    Returns tokenizer and model as a tuple.
    """
    logger.info(f"Loading multilingual model from {MULTILINGUAL_MODEL_DIR}")
    try:
        # Check if local model exists
        if not os.path.exists(MULTILINGUAL_MODEL_DIR):
            logger.warning(f"Local multilingual model not found at {MULTILINGUAL_MODEL_DIR}, attempting to download...")
            
            # Ensure the parent directory exists
            os.makedirs(os.path.dirname(MULTILINGUAL_MODEL_DIR), exist_ok=True)
            
            # Download model from Hugging Face
            model_name = "xlm-roberta-base"
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MULTILINGUAL_MODEL_DIR)
            model = AutoModel.from_pretrained(model_name, cache_dir=MULTILINGUAL_MODEL_DIR)
            
            logger.info(f"Successfully downloaded multilingual model to {MULTILINGUAL_MODEL_DIR}")
        else:
            # Load from local directory
            tokenizer = AutoTokenizer.from_pretrained(MULTILINGUAL_MODEL_DIR, local_files_only=True)
            model = AutoModel.from_pretrained(MULTILINGUAL_MODEL_DIR, local_files_only=True)
            
        return tokenizer, model
    except Exception as e:
        logger.error(f"Could not load or download multilingual model: {e}")
        raise

def check_unprocessed_documents(engine):
    """
    Check if there are any documents in the database that haven't been processed yet.
    
    Args:
        engine: SQLAlchemy engine for database connection
        
    Returns:
        tuple: (has_unprocessed, total_docs, unprocessed_count)
            - has_unprocessed: Boolean indicating if there are unprocessed documents
            - total_docs: Total number of documents in the database
            - unprocessed_count: Number of unprocessed documents
    """
    try:
        with engine.connect() as conn:
            # Get total document count
            total_query = text("SELECT COUNT(*) FROM documents")
            total_result = conn.execute(total_query)
            total_docs = total_result.scalar()
            
            if total_docs == 0:
                logger.warning("No documents found in the database")
                return False, 0, 0
            
            # Get count of unprocessed documents (where processed_at is NULL)
            unprocessed_query = text("SELECT COUNT(*) FROM documents WHERE processed_at IS NULL")
            unprocessed_result = conn.execute(unprocessed_query)
            unprocessed_count = unprocessed_result.scalar()
            
            has_unprocessed = unprocessed_count > 0
            return has_unprocessed, total_docs, unprocessed_count
    except Exception as e:
        logger.error(f"Error checking for unprocessed documents: {e}")
        # If there's an error, assume there are unprocessed documents to be safe
        return True, 0, 0

def get_document_metadata(engine, doc_id):
    """
    Get document metadata from the database
    """
    try:
        with engine.connect() as conn:
            query = text("SELECT country, title, submission_date FROM documents WHERE doc_id = :doc_id")
            result = conn.execute(query, {"doc_id": doc_id})
            row = result.fetchone()
            
            if not row:
                return None
            
            submission_date = row[2]
            if submission_date and isinstance(submission_date, datetime):
                submission_date = submission_date.isoformat()
            elif submission_date:
                submission_date = str(submission_date)
            
            return {
                'country': row[0] or '',
                'document_title': row[1] or '',
                'submission_date': submission_date or ''
            }
    except Exception as e:
        logger.error(f"Error retrieving document metadata: {e}")
        return None

def safe_execute_sql(conn, sql_query, params=None):
    """
    Safely execute SQL with error handling.
    """
    try:
        if params:
            return conn.execute(sql_query, params)
        else:
            return conn.execute(sql_query)
    except Exception as e:
        logger.error(f"SQL Error: {e} for query {sql_query}")
        if params:
            logger.error(f"Parameters: {params}")
        raise

def generate_embedding(text, tokenizer, model):
    """
    Generate embeddings for text using the provided tokenizer and model.
    
    Args:
        text (str): The text to generate embeddings for
        tokenizer: The tokenizer for the model
        model: The transformer model
        
    Returns:
        list: Embedding vector (list of floats)
    """
    try:
        # Trim text if it's too long for the model
        max_length = tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') else 512
        if len(text) > max_length * 4:  # Rough character estimate
            logger.warning(f"Text too long ({len(text)} chars), truncating for embedding generation")
            text = text[:max_length * 4]  # Rough truncation before tokenization
            
        # Tokenize and prepare for model
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        
        # Move to same device as model
        if torch.cuda.is_available():
            inputs = {key: val.to('cuda') for key, val in inputs.items()}
            model = model.to('cuda')
        
        # Generate embeddings with no gradient calculation
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use CLS token embedding (first token) as the document embedding
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0].tolist()
        return embeddings
    
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        # Return a zero vector as fallback
        return [0.0] * 768  # Standard size for most transformer embeddings

def process_document_chunks(chunks_data, engine, english_tokenizer, english_model, multi_tokenizer, multi_model):
    """
    Process document chunks, determine language, generate embeddings, and store in the database.
    """
    total_chunks = len(chunks_data)
    logger.info(f"Processing {total_chunks} document chunks")
    
    # Process each chunk with tqdm progress bar
    for i, chunk in enumerate(tqdm(chunks_data, desc="Processing chunks", unit="chunk")):
        if i % 100 == 0:
            logger.info(f"Processing chunk {i+1}/{total_chunks}")
        
        # Extract chunk text and metadata
        chunk_text = chunk.get('text', '')
        metadata = chunk.get('metadata', {})
        
        # Skip empty chunks
        if not chunk_text.strip():
            logger.warning(f"Skipping empty chunk {i}")
            continue
        
        # Get filename from metadata if available
        filename = metadata.get('filename', 'unknown.txt')
        
        # Determine language from filename/metadata first
        language = determine_language(filename, metadata)
        
        # If we couldn't determine language from filename/metadata, detect from content
        if language == 'en' and len(chunk_text.strip()) > 50:
            try:
                import langdetect
                from langdetect.lang_detect_exception import LangDetectException
                
                # Only attempt detection if text has sufficient content
                try:
                    detected_lang = langdetect.detect(chunk_text)
                    if detected_lang and detected_lang != 'en':
                        logger.info(f"Language detected from content: {detected_lang}")
                        language = detected_lang
                except LangDetectException:
                    # If detection fails, stick with English
                    logger.debug("Language detection failed, using English")
            except ImportError:
                logger.warning("langdetect not available, using determined language")
        
        # Store the determined language in metadata
        metadata['document_language'] = language
        
        # Choose appropriate model based on language
        if language == 'en' and english_tokenizer is not None and english_model is not None:
            tokenizer = english_tokenizer
            model = english_model
        elif multi_tokenizer is not None and multi_model is not None:
            tokenizer = multi_tokenizer
            model = multi_model
        else:
            # Fallback to English if available, otherwise log and skip
            if english_tokenizer is not None and english_model is not None:
                logger.warning(f"No model available for language {language}, falling back to English")
                tokenizer = english_tokenizer
                model = english_model
            else:
                logger.error(f"No suitable model available for chunk {i}")
                continue
        
        # Generate embedding
        embedding = generate_embedding(chunk_text, tokenizer, model)
        
        # Extract paragraph information from metadata
        paragraph = None
        if 'paragraph_number' in metadata:
            paragraph = metadata['paragraph_number']
        elif 'paragraph_numbers' in metadata and metadata['paragraph_numbers']:
            paragraph = metadata['paragraph_numbers'][0] if isinstance(metadata['paragraph_numbers'], list) else metadata['paragraph_numbers']
        
        # Add to database
        try:
            # Extract metadata needed for document identification
            country = metadata.get('country', '')
            title = metadata.get('document_title', '')
            date = metadata.get('submission_date', None)
            
            # Extract doc_id from filename and remove file extensions
            base_doc_id = metadata.get('filename', '')
            # Remove file extensions like .pdf or .docx
            base_doc_id = re.sub(r'\.(pdf|docx?|txt|csv|xlsx?)$', '', base_doc_id, flags=re.IGNORECASE)
            
            with engine.connect() as conn:
                # First check if this document already exists
                check_existing_query = text("""
                    SELECT doc_id FROM documents 
                    WHERE doc_id = :base_doc_id
                    LIMIT 1
                """)
                
                existing_result = safe_execute_sql(conn, check_existing_query, {
                    "base_doc_id": base_doc_id
                })
                doc_row = existing_result.fetchone()
                
                database_doc_id = None
                
                # If document does not exist, create it
                if not doc_row:
                    table_check = text("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_name = 'documents'
                        )
                    """)
                    table_exists = safe_execute_sql(conn, table_check).fetchone()[0]
                    
                    if not table_exists:
                        logger.error("SQL table for Embeddings not found")
                        return
                    
                    unique_url = f"https://example.org/{country.lower().replace(' ', '_')}/{base_doc_id}"
                    
                    insert_doc = text("""
                        INSERT INTO documents (doc_id, language, title, country, submission_date, url)
                        VALUES (:doc_id, :language, :title, :country, :date, :url)
                        RETURNING doc_id
                    """)
                    
                    doc_result = conn.execute(insert_doc, {
                        "doc_id": base_doc_id,
                        "language": language,
                        "title": title,
                        "country": country,
                        "date": date,
                        "url": unique_url
                    })
                    
                    database_doc_id = doc_result.fetchone()[0]
                    logger.info(f"Created new document with ID: {database_doc_id}")
                else:
                    database_doc_id = doc_row[0]
                    logger.debug(f"Using existing document with ID: {database_doc_id}")
                
                insert_chunk = text("""
                    INSERT INTO doc_chunks (
                        doc_id, content, embedding, chunk_index, chunk_metadata, language, paragraph
                    )
                    VALUES (
                        :doc_id, :content, :embedding, :chunk_index, :chunk_metadata, :language, :paragraph
                    )
                """)
                
                embedding_string = f"[{','.join(map(str, embedding))}]"
                
                result = conn.execute(insert_chunk, {
                    "doc_id": database_doc_id,
                    "content": chunk_text,
                    "embedding": embedding_string,
                    "chunk_index": i,
                    "chunk_metadata": json.dumps(metadata),
                    "language": language,
                    "paragraph": paragraph
                })
                
                conn.commit()
        
        except Exception as e:
            logger.error(f"Error adding chunk to database: {e}")

def main():
    """Main function to populate the database with processed document chunks."""
    load_dotenv()
        
    # Load database connection
    db_url = os.getenv('DATABASE_URL', 'postgresql://climate:climate@localhost:5432/climate')
    engine = create_engine(db_url)
    
    # Check for unprocessed documents
    has_unprocessed, total_docs, unprocessed_count = check_unprocessed_documents(engine)
    
    if has_unprocessed:
        print(f"\nWARNING: {unprocessed_count} out of {total_docs} documents have not been processed.")
        print("These documents won't have chunks available for embedding.")
        
        # Prompt user for decision
        response = input("Do you want to continue with embedding anyway? (y/N): ").strip().lower()
        
        if response != 'y':
            print("Exiting. Please run process_documents.py to process all documents first.")
            return
        
        print(f"Continuing with embedding for {total_docs - unprocessed_count} processed documents.\n")
    
    # Load models
    try:
        english_tokenizer, english_model = load_english_model()
        multi_tokenizer, multi_model = load_multilingual_model()
        logger.info("Successfully loaded language models")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        print(f"ERROR: Could not load language models: {e}")
        return
    
    # Load document chunks from processed files
    data_dir = os.getenv('DATA_DIR', 'data')
    chunks_dir = os.path.join(data_dir, 'processed', 'chunks')
    if not os.path.exists(chunks_dir):
        logger.error(f"Chunks directory not found: {chunks_dir}")
        print(f"ERROR: Chunks directory not found. Run the document processing first.")
        return
    
    # Process all chunk files with tqdm progress bar
    json_files = [f for f in os.listdir(chunks_dir) if f.endswith('.json')]
    for filename in tqdm(json_files, desc="Processing JSON files", unit="file"):
        file_path = os.path.join(chunks_dir, filename)
        logger.info(f"Processing chunks file: {filename}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            process_document_chunks(
                chunks_data, 
                engine, 
                english_tokenizer, 
                english_model,
                multi_tokenizer,
                multi_model
            )
            logger.info(f"Completed processing {filename}")
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
    
    logger.info("Database population complete")
    print("Database population complete. The documents have been processed and stored with language detection.")

if __name__ == "__main__":
    main()
