"""
Perform similarity search with advanced filters to find information about emission reduction targets
"""
import os
import sys
import torch
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import langdetect
import json
from sklearn.metrics.pairwise import cosine_similarity
import re

# Add the parent directory to the path so we can import from the climate_policy_extractor package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from climate_policy_extractor.logging import get_logger
# Import model loading functions and embedding generation from populate_database.py
from populate_database import load_english_model, load_multilingual_model, generate_embedding

logger = get_logger(__name__)

# Constants for search parameters
SIMILARITY_THRESHOLD = 0.65
MAX_RESULTS_PER_COUNTRY = 20  # top 20 results per country
USE_MULTILINGUAL = True  # Set to True to enable multilingual search

def detect_language(text):
    """
    Detect the language of the provided text using langdetect
    Returns the ISO 639-1 language code (e.g. 'en', 'es', 'fr')
    Falls back to 'en' if detection fails
    """
    try:
        return langdetect.detect(text)
    except Exception as e:
        logger.warning(f"Language detection failed: {e}. Falling back to English.")
        return 'en'

def enhance_query_for_emissions_target(query, query_lang='en'):
    """
    Enhance the query to better target emissions reduction information
    Handles different languages by adding relevant keywords
    """
    base_query = query.strip()
    
    # Define multilingual keywords for emissions/climate terms
    multilingual_keywords = {
        'en': "emissions reduction targets NDCs 2030 percentage reductions baseline years conditional unconditional greenhouse gas CO2 carbon net-zero climate goals",
        'es': "reducción de emisiones objetivos NDC 2030 porcentaje reducciones año base condicional incondicional gases de efecto invernadero CO2 carbono neutralidad climática",
        'fr': "réduction des émissions objectifs CDN 2030 pourcentage réductions année de référence conditionnel inconditionnel gaz à effet de serre CO2 carbone neutralité climatique",
        # Add more languages as needed
    }
    
    # Get keywords for the detected language, or use English as fallback
    lang_keywords = multilingual_keywords.get(query_lang, multilingual_keywords['en'])
    
    # Create enhanced query with language-specific keywords
    enhanced_query = f"{base_query} {lang_keywords}"
    
    return enhanced_query

def safe_extract_target_percentage(text):

    """
    Safely extract percentage values related to emissions targets from text.
    
    Parameters:
    -----------
    text : str
        The text content to extract percentage values from
        
    Returns:
    --------
    str or None
        Extracted percentage value as a string, or None if no valid percentage found
    """
    if not isinstance(text, str) or not text:
        return None
        
    try:
        # Pattern 1: Match common percentage patterns related to emissions targets
        # Looks for numbers followed by % with potential keywords before/after
        percentage_patterns = [
            # Basic percentage pattern with target-related context
            r'(?:reduce|reduction|cut|decrease|lower|target|goal|pledge|commit|aim|by|of)\s+(?:emissions|emission|ghg|co2|carbon|greenhouse gas)?\s*(?:by|of)?\s*(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?%)',
            
            # Percentage at beginning of phrase
            r'(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?%)\s+(?:reduction|decrease|cut|below|compared)',
            
            # Range of percentages
            r'(?:between|from)\s+(\d+(?:\.\d+)?%?\s*(?:and|to)\s*\d+(?:\.\d+)?%)',
            
            # Percentage with 'by 2030' or similar context
            r'(\d+(?:\.\d+)?%)[^.]*?by\s+(?:20\d\d)',
            
            # Fallback: any percentage in context of emissions
            r'(?:emission|carbon|ghg)[^.]*?(\d+(?:\.\d+)?%)',
            
            # Last resort: any percentage pattern
            r'(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?%)'
        ]
        
        for pattern in percentage_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                # Clean up the extracted percentage
                percentage = matches[0].strip()
                # Handle cases where regex capture group might capture extra text
                if isinstance(percentage, tuple):
                    percentage = percentage[0]
                # Ensure it actually contains a % symbol
                if '%' in percentage:
                    return percentage
        
        # Special case: look for numbers followed by "percent" word
        percent_word_pattern = r'(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?)\s+percent'
        percent_matches = re.findall(percent_word_pattern, text.lower())
        if percent_matches:
            return f"{percent_matches[0]}%"
        
        # Special case: look for written percentages (e.g., "fifteen percent")
        written_numbers = {
            'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
            'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
            'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14', 'fifteen': '15',
            'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19', 'twenty': '20',
            'twenty-five': '25', 'thirty': '30', 'forty': '40', 'fifty': '50',
            'sixty': '60', 'seventy': '70', 'eighty': '80', 'ninety': '90'
        }
        
        written_pattern = '|'.join(written_numbers.keys())
        written_regex = fr'({written_pattern})\s+percent'
        written_matches = re.findall(written_regex, text.lower())
        
        if written_matches and written_matches[0] in written_numbers:
            return f"{written_numbers[written_matches[0]]}%"
            
        return None
        
    except Exception as e:
        # In case of any error, return None to ensure function doesn't crash
        print(f"Error extracting percentage: {e}")
        return None
    
def safe_extract_target_year(text):
    """
    Safely extract target years related to emissions goals from text.
    
    Parameters:
    -----------
    text : str
        The text content to extract year values from
        
    Returns:
    --------
    str or None
        Extracted year value as a string, or None if no valid year found
    """
    if not isinstance(text, str) or not text:
        return None
        
    try:
        # Pattern 1: Match target years in common climate policy contexts
        # Looks for year mentions that are likely associated with emissions targets
        year_patterns = [
            # Year after "by" in target context
            r'(?:by|until|in|before|from)\s+(?:the\s+year\s+)?(20[0-9]{2})(?:\s+level)?',
            
            # Year in reference to baseline
            r'(?:compared\s+to|relative\s+to|from|baseline|reference\s+year)\s+(?:the\s+year\s+)?((?:19|20)[0-9]{2})',
            
            # Year as target in NDC context
            r'(?:NDC|INDC|commitment|pledge|target|goal)s?\s+(?:for|by|in)\s+(?:the\s+year\s+)?(20[0-9]{2})',
            
            # Target/percentage with year
            r'(?:[0-9]+(?:\.[0-9]+)?%)[^.]*?(?:by|in|before)\s+(?:the\s+year\s+)?(20[0-9]{2})',
            
            # Net-zero by year
            r'(?:net[- ]zero|carbon[- ]neutral(?:ity)?|climate[- ]neutral(?:ity)?)[^.]*?(?:by|in|before)\s+(?:the\s+year\s+)?(20[0-9]{2})',
            
            # Years specifically mentioned with 2030 (common NDC target)
            r'(?:20[0-9]{2}(?:\s*-\s*|\s+and\s+|\s+to\s+))?2030',
            
            # Generic year pattern as fallback (bias toward future years)
            r'(?:20[2-9][0-9])'
        ]
        
        for pattern in year_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                # Clean up the extracted year
                year = matches[0].strip()
                if year.isdigit() and len(year) == 4 and year.startswith(('19', '20')):
                    return year
        
        # Special case for long-term goals
        long_term_patterns = [
            r'(?:long[- ]term|long[- ]range)\s+(?:goal|target|objective)[^.]*?(20[0-9]{2})',
            r'(?:mid[- ]century|by\s+(?:the\s+)?mid[- ]century)',
            r'(?:by\s+)?2050'
        ]
        
        for pattern in long_term_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                if matches[0].isdigit():
                    return matches[0]
                elif 'mid-century' in text.lower() or 'mid century' in text.lower():
                    return '2050'  # Standard interpretation of mid-century
        
        # Examine common explicit target years in climate policy
        common_years = ['2020', '2025', '2030', '2035', '2040', '2045', '2050', '2060', '2070', '2100']
        for year in common_years:
            if year in text:
                # Verify it's not part of a different number
                year_pattern = r'\b' + year + r'\b'
                if re.search(year_pattern, text):
                    return year
                    
        return None
        
    except Exception as e:
        # In case of any error, return None to ensure function doesn't crash
        print(f"Error extracting year: {e}")
        return None

def get_transformer_embeddings():
    """
    Retrieve transformer embeddings from the database
    """
    load_dotenv()
    db_url = os.getenv('DATABASE_URL', 'postgresql://climate:climate@localhost:5432/climate')
    engine = create_engine(db_url)
    
    with engine.connect() as conn:
        query = text("""
            SELECT d.country, dc.content, dc.embedding, dc.id, dc.chunk_metadata, dc.doc_id
            FROM doc_chunks dc
            JOIN documents d ON dc.doc_id = d.doc_id
            WHERE dc.embedding IS NOT NULL
        """)
        
        result = conn.execute(query)
        rows = result.fetchall()
    
    # Convert to DataFrame
    data = []
    for row in rows:
        # Extract embedding from PostgreSQL vector format
        embedding_str = row[2]
        # Convert embedding string to list of floats
        if embedding_str.startswith('[') and embedding_str.endswith(']'):
            embedding_str = embedding_str[1:-1]  # Remove brackets
            embedding = [float(x) for x in embedding_str.split(',')]
        else:
            # Default empty embedding if parsing fails
            embedding = []
        
        # Parse JSON metadata
        try:
            metadata = json.loads(row[4]) if row[4] else {}
        except (json.JSONDecodeError, TypeError):
            metadata = {}
        
        data.append({
            'country': row[0],
            'text': row[1],
            'embedding': embedding,
            'chunk_id': row[3],
            'metadata': metadata,
            'doc_id': row[5]
        })
    
    return pd.DataFrame(data)

def get_word2vec_embeddings():
    """
    Retrieve word2vec embeddings from the database with improved error handling
    and proper metadata extraction
    """
    load_dotenv()
    db_url = os.getenv('DATABASE_URL', 'postgresql://climate:climate@localhost:5432/climate')
    engine = create_engine(db_url)
    
    with engine.connect() as conn:
        query = text("""
            SELECT d.country, dc.content, dc.word2vec_embedding, dc.id, dc.chunk_metadata, dc.doc_id, 
                   d.language as doc_language, dc.language as chunk_language,
                   -- Calculate keyword boost
                   (CASE WHEN dc.content ~* 'emission|reduction|target|ndc|2030|co2|carbon|ghg' THEN 0.1 ELSE 0 END) +
                   (CASE WHEN dc.content ~* 'percent|baseline|conditional|unconditional' THEN 0.05 ELSE 0 END) AS keyword_boost,
                   -- Percentage pattern boost
                   (CASE WHEN dc.content ~ '[0-9]+([.][0-9]+)?%' THEN 0.15 ELSE 0 END) AS percentage_boost
            FROM doc_chunks dc
            JOIN documents d ON dc.doc_id = d.doc_id
            WHERE dc.word2vec_embedding IS NOT NULL
                 AND dc.content ~* 'target|emission|reduction|climate|goal|pledge|commitment|2030'
        """)
        
        try:
            result = conn.execute(query)
            rows = result.fetchall()
            if not rows:
                logger.warning("No word2vec embeddings found in database")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error querying word2vec embeddings: {e}")
            return pd.DataFrame()
    
    # Convert to DataFrame
    data = []
    for row in rows:
        # Extract embedding from PostgreSQL vector format
        embedding_str = row[2]
        
        # Parse JSON metadata with proper error handling
        chunk_metadata = row[4]
        try:
            if chunk_metadata and isinstance(chunk_metadata, str):
                metadata = json.loads(chunk_metadata)
            elif isinstance(chunk_metadata, dict):
                metadata = chunk_metadata
            else:
                metadata = {}
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Error parsing metadata JSON: {e}")
            metadata = {}
        
        # Extract individual metadata fields for direct access
        page_number = None
        element_type = None
        if isinstance(metadata, dict):
            page_number = metadata.get('page_number', None)
            element_type = metadata.get('element_type', None)
        
        data.append({
            'country': row[0],
            'content': row[1],
            'embedding_str': embedding_str,
            'chunk_id': row[3],
            'chunk_metadata': metadata,
            'doc_id': row[5],
            'doc_language': row[6],
            'chunk_language': row[7],
            'keyword_boost': float(row[8]) if row[8] is not None else 0.0,
            'percentage_boost': float(row[9]) if row[9] is not None else 0.0,
            'embedding_type': 'word2vec',
            'page_number': page_number,
            'element_type': element_type
        })
    
    df = pd.DataFrame(data)
    
    # Extract embeddings from PostgreSQL format
    def extract_embedding(embedding_str):
        if embedding_str and isinstance(embedding_str, str):
            try:
                # Handle both [1,2,3] format and {1,2,3} format
                if embedding_str.startswith('[') and embedding_str.endswith(']'):
                    embedding_str = embedding_str[1:-1]
                elif embedding_str.startswith('{') and embedding_str.endswith('}'):
                    embedding_str = embedding_str[1:-1]
                
                # Split by comma and convert to float
                return np.array([float(x) for x in embedding_str.split(',')])
            except (ValueError, TypeError) as e:
                logger.warning(f"Error parsing embedding: {e}")
                return None
        return None
        
    df['embedding'] = df['embedding_str'].apply(extract_embedding)
    
    # Remove rows with invalid embeddings
    df = df.dropna(subset=['embedding'])
    
    return df

def process_word2vec_search(query, similarity_threshold=SIMILARITY_THRESHOLD, max_results_per_country=MAX_RESULTS_PER_COUNTRY):
    """
    Process word2vec search using the improved methodology
    
    Parameters:
    -----------
    query : str
        Search query
    similarity_threshold : float
        Minimum similarity threshold to include in results
    max_results_per_country : int
        Maximum results to return per country
        
    Returns:
    --------
    pandas.DataFrame
        Search results
    """
    logger.info(f"Processing word2vec search with query: {query}")
    
    # Enhance the query for better targeting
    enhanced_query = query + " emissions reduction targets NDCs 2030 percentage reductions baseline years conditional unconditional greenhouse gas CO2 carbon net-zero climate goals"
    
    # Get all word2vec embeddings
    df = get_word2vec_embeddings()
    
    if df.empty:
        logger.warning("No word2vec embeddings found")
        return pd.DataFrame()
    
    # Generate query embedding from relevant document embeddings
    try:
        relevant_embeddings = df[df['keyword_boost'] > 0]['embedding'].tolist()
        if len(relevant_embeddings) >= 3:
            # Use several relevant embeddings
            query_embedding = np.mean(relevant_embeddings[:10], axis=0)
        else:
            # If not enough relevant embeddings, use mean of all
            all_embeddings = df['embedding'].tolist()
            query_embedding = np.mean(np.vstack(all_embeddings[:20]), axis=0)
        
        # Calculate similarity scores
        df['similarity_score'] = df['embedding'].apply(
            lambda x: float(cosine_similarity([query_embedding], [x])[0][0]) if x is not None else 0.0
        )
    except Exception as e:
        logger.error(f"Error calculating word2vec similarities: {e}")
        df['similarity_score'] = 0.5  # Fallback default score
    
    # Calculate total score
    df['total_score'] = df['similarity_score'] + df['keyword_boost'] + df['percentage_boost']
    
    # Filter by similarity threshold or boost
    filtered_df = df[(df['similarity_score'] >= similarity_threshold) | 
                    (df['keyword_boost'] > 0) | 
                    (df['percentage_boost'] > 0)].copy()
    
    if filtered_df.empty:
        logger.info("No word2vec results found above threshold")
        return pd.DataFrame()
    
    # Extract metadata fields
    def extract_metadata_field(metadata, field, default='Unknown'):
        if isinstance(metadata, dict):
            return metadata.get(field, default)
        return default
    
    # Fix: Change 'metadata' to 'chunk_metadata' to match the column name in the DataFrame
    filtered_df['page_number'] = filtered_df['chunk_metadata'].apply(
        lambda x: extract_metadata_field(x, 'page_number'))
    filtered_df['element_type'] = filtered_df['chunk_metadata'].apply(
        lambda x: extract_metadata_field(x, 'element_type'))
    
    # Extract target information
    filtered_df['extracted_percentage'] = filtered_df['content'].apply(safe_extract_target_percentage)
    filtered_df['extracted_year'] = filtered_df['content'].apply(safe_extract_target_year)
    filtered_df['target_text'] = filtered_df['content']
    
    # Keep only the top results per country
    top_results = []
    for country, group in filtered_df.groupby('country'):
        group = group.sort_values('total_score', ascending=False)
        top_results.append(group.head(max_results_per_country))
    
    # Combine results
    if top_results:
        final_df = pd.concat(top_results)
        final_df = final_df.sort_values(['country', 'total_score'], ascending=[True, False])
        
        # Clean up the DataFrame
        cols_to_drop = ['embedding_str', 'embedding']
        final_df = final_df.drop(columns=[c for c in cols_to_drop if c in final_df.columns])
        
        # Remove this line as it's unnecessary and would cause errors - chunk_metadata is already named correctly
        # final_df = final_df.rename(columns={'metadata': 'chunk_metadata'})
        
        return final_df
    else:
        return pd.DataFrame()

def get_emissions_targets(query=None, engine=None, tokenizer=None, model=None, use_word2vec=False):
    """
    Perform similarity search with advanced filters to find information about emission reduction targets.
    
    Parameters:
    -----------
    query : str, optional
        The search query for finding emissions targets
    engine : SQLAlchemy engine, optional
        Database connection engine
    tokenizer : transformer tokenizer, optional
        Tokenizer for transformer model
    model : transformer model, optional
        Transformer model for generating embeddings
    use_word2vec : bool, optional
        Whether to include word2vec embeddings in search (default: False)
    """
    if query is None:
        query = "What emissions reduction target is each country in the NDC registry aiming for by 2030?"
    
    print("INFO: Currently using transformer embeddings for similarity search.")
    if not use_word2vec:
        print("INFO: Word2vec embeddings are not being used. Set use_word2vec=True to include them.")
    else:
        print("INFO: Word2vec embeddings will also be used to enhance the search results.")
    
    # Create database connection if not provided
    if engine is None:
        load_dotenv()
        db_url = os.getenv('DATABASE_URL', 'postgresql://climate:climate@localhost:5432/climate')
        engine = create_engine(db_url)
    
    # Load models if not provided
    if tokenizer is None or model is None:
        tokenizer, model = load_english_model()

    # Get transformer embeddings results
    transformer_df = get_transformer_results(query, engine, tokenizer, model)
    
    # All results dataframe to store combined results from both embedding types
    all_results = transformer_df.copy() if not transformer_df.empty else pd.DataFrame()
    
    # If word2vec is requested, get word2vec results
    if use_word2vec:
        print("Processing word2vec search...")
        word2vec_df = process_word2vec_search(query)
        
        if not word2vec_df.empty:
            # Combine the results
            all_results = pd.concat([all_results, word2vec_df]) if not all_results.empty else word2vec_df
            
            # Remove duplicates by keeping the highest score when there are duplicates
            if not all_results.empty:
                all_results = all_results.sort_values('total_score', ascending=False)
                all_results = all_results.drop_duplicates(subset=['chunk_id'])
    
    # Keep only the top results per country
    if not all_results.empty:
        top_results = []
        for country, group in all_results.groupby('country'):
            group = group.sort_values('total_score', ascending=False)
            top_results.append(group.head(MAX_RESULTS_PER_COUNTRY))
        
        # Combine results
        if top_results:
            result_df = pd.concat(top_results)
            result_df = result_df.sort_values(['country', 'total_score'], ascending=[True, False])
            return result_df
    
    return all_results if not all_results.empty else pd.DataFrame()

def get_transformer_results(query, engine, tokenizer, model):
    """Get search results using transformer embeddings"""
    # Detect query language
    query_lang = detect_language(query)
    logger.info(f"Detected query language: {query_lang}")
    
    # Enhance the query for better targeting
    enhanced_query = enhance_query_for_emissions_target(query, query_lang)
    logger.info(f"Enhanced query: {enhanced_query}")
    
    # Generate embedding for the enhanced query
    query_embedding = generate_embedding(enhanced_query, tokenizer, model)
    
    # Convert embedding to PostgreSQL vector format - ensure proper formatting
    embedding_string = f"[{','.join(map(str, query_embedding))}]"
    
    try:
        # Get raw connection from SQLAlchemy engine
        conn = engine.raw_connection()
        cursor = conn.cursor()
        try:
            # SQL query with properly formatted parameters for pgvector
            sql_query = """
            WITH similarity_results AS (
                SELECT 
                    c.id,
                    c.doc_id,
                    d.country,
                    c.content,
                    c.chunk_metadata,
                    c.chunk_index,
                    d.language as doc_language,
                    c.language as chunk_language,
                    -- Calculate cosine similarity using pgvector's <=> operator
                    1 - (c.embedding <=> %s::vector) AS cosine_similarity,
                    
                    -- Keyword boost - simplified with regex
                    (CASE WHEN c.content ~* '\\y(emission|reduction|target|ndc|2030|co2|carbon|ghg)\\y' THEN 0.1 ELSE 0 END) +
                    (CASE WHEN c.content ~* '\\y(percent|baseline|conditional|unconditional)\\y' THEN 0.05 ELSE 0 END) AS keyword_boost,
                    
                    -- Percentage pattern boost
                    CASE WHEN c.content ~* E'\\d+(-\\d+)?%%' THEN 0.15 ELSE 0 END AS percentage_boost,
                    
                    'transformer' AS embedding_type
                FROM 
                    doc_chunks c
                JOIN
                    documents d ON c.doc_id = d.doc_id
                WHERE 
                    -- Use pgvector's operator for filtering with threshold
                    c.embedding IS NOT NULL AND (1 - (c.embedding <=> %s::vector)) > 0.4
            )
            
            SELECT 
                id as chunk_id,
                doc_id,
                country,
                content,
                chunk_metadata,
                chunk_index,
                doc_language,
                chunk_language,
                cosine_similarity as similarity_score,
                keyword_boost,
                percentage_boost,
                cosine_similarity + keyword_boost + percentage_boost AS total_score,
                embedding_type
            FROM 
                similarity_results
            WHERE
                cosine_similarity >= %s
                OR keyword_boost > 0
                OR percentage_boost > 0
            ORDER BY
                country,
                total_score DESC
            """
            
            # Execute with proper parameters list
            cursor.execute(sql_query, (
                embedding_string,  # For cosine_similarity calculation
                embedding_string,  # For WHERE clause filtering
                SIMILARITY_THRESHOLD  # For cosine_similarity threshold filter
            ))
            
            # Fetch all rows
            rows = cursor.fetchall()
            
            # Get column names from cursor description
            column_names = [desc[0] for desc in cursor.description]
            
            # Create DataFrame with proper column names
            transformer_df = pd.DataFrame(rows, columns=column_names)
            
            # Extract additional data
            if not transformer_df.empty:
                transformer_df['extracted_percentage'] = transformer_df['content'].apply(safe_extract_target_percentage)
                transformer_df['extracted_year'] = transformer_df['content'].apply(safe_extract_target_year)
                transformer_df['target_text'] = transformer_df['content']
            
            return transformer_df
            
        finally:
            # Ensure cursor is closed
            cursor.close()
            # Ensure connection is closed
            conn.close()
    except Exception as e:
        logger.error(f"Database query error: {e}")
        return pd.DataFrame()

def format_results(df):
    """Format the results for display with language information"""
    if df.empty:
        return "No results found."
    
    results = []
    
    for country, group in df.groupby('country'):
        results.append(f"\n{'='*80}\nCOUNTRY: {country.upper()} (Top {min(len(group), MAX_RESULTS_PER_COUNTRY)} Results)\n{'='*80}")
        
        for i, (_, row) in enumerate(group.iterrows(), 1):
            lang_info = f" [{row.get('doc_language', 'unknown')}]" if 'doc_language' in row else ""
            embedding_type = f" - {row.get('embedding_type', 'unknown')} embedding" if 'embedding_type' in row else ""
            results.append(f"\nResult {i}/{len(group)}{embedding_type}")
            results.append(f"Score: {row['total_score']:.4f}{lang_info}")
            results.append(f"Document: {row['doc_id']}")
            results.append(f"Page: {row.get('page_number', 'Unknown')}, Type: {row.get('element_type', 'Unknown')}")
            results.append(f"\nContent: {row['content']}")
            results.append("-" * 80)
    
    return "\n".join(results)

def load_model_safely(model_type='english'):
    """
    Load the model with improved path handling and error recovery
    
    Parameters:
    -----------
    model_type : str
        Type of model to load ('english' or 'multilingual')
    
    Returns:
    --------
    tuple
        Tokenizer and model, or (None, None) if loading fails
    """
    from transformers import AutoTokenizer, AutoModel
    
    # Use os.path to construct a proper path
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'local_models'))
    
    if model_type.lower() == 'english':
        model_name = 'distilroberta-base'
    else:  # multilingual
        model_name = 'xlm-roberta-base'
    
    model_dir = os.path.join(base_dir, model_name).replace('\\', '/')
    logger.info(f"Loading {model_type} model from {model_dir}")
    
    try:
        # First try to load locally
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModel.from_pretrained(model_dir, local_files_only=True)
        logger.info(f"Successfully loaded {model_type} model from local directory")
        return tokenizer, model
    except Exception as e:
        logger.warning(f"Could not load {model_type} model locally: {e}")
        
        # Try downloading from HuggingFace Hub as fallback
        try:
            hub_model_name = f"sentence-transformers/{model_name}" if model_type.lower() == 'multilingual' else model_name
            logger.info(f"Attempting to download {model_type} model from HuggingFace Hub: {hub_model_name}")
            tokenizer = AutoTokenizer.from_pretrained(hub_model_name)
            model = AutoModel.from_pretrained(hub_model_name)
            logger.info(f"Successfully downloaded and loaded {model_type} model")
            return tokenizer, model
        except Exception as download_error:
            logger.error(f"Failed to download {model_type} model: {download_error}")
            return None, None

def main():
    """Main function to perform similarity search with multiple queries"""
    load_dotenv()
    db_url = os.getenv('DATABASE_URL', 'postgresql://climate:climate@localhost:5432/climate')
    engine = create_engine(db_url)
    
    # Load transformer model for query embedding with improved path handling
    tokenizer, model = load_model_safely('english')
    
    # If loading fails, exit gracefully
    if tokenizer is None or model is None:
        logger.error("Failed to load models. Exiting.")
        print("Error: Could not load necessary models. Please check logs for details.")
        return
    
    # Ask user if they want to use word2vec embeddings
    use_word2vec = False
    try:
        user_input = input("Do you want to include word2vec embeddings in the search? (y/N): ").strip().lower()
        use_word2vec = user_input == 'y'
    except Exception as e:
        logger.error(f"Error getting user input: {e}")
        print("Using only transformer embeddings due to input error.")
    
    # Define standard queries
    standard_queries = [
        "What emissions reduction target is each country in the NDC registry aiming for by 2030?",
        "2030 GHG emissions %",
        "what is the commitment by 2035",
        "conditional and unconditional NDC targets"
    ]
    
    # Summarize the queries being used
    print(f"\nUsing {len(standard_queries)} queries for search:")
    for i, query in enumerate(standard_queries, 1):
        print(f"{i}. {query}")
    
    # Run search with selected queries
    print("\nSearching with selected queries. This may take some time...")
    results = get_emissions_targets(standard_queries[0], engine, tokenizer, model, use_word2vec=use_word2vec)
    
    # Print formatted results
    print(format_results(results))
    
    # Save results to CSV
    if not results.empty:
        results_dir = os.path.join('data', 'results')
        os.makedirs(results_dir, exist_ok=True)
        output_file = os.path.join(results_dir, 'similarity_search_results.csv')
        results.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
