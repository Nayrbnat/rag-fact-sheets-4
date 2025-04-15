"""
Utility functions for the climate policy extractor.
"""
# Standard library imports
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm  # Progress bar library

# Scientific and ML libraries
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer  # Hugging Face transformers for embeddings
from unstructured.partition.pdf import partition_pdf  # PDF processing library
from sqlalchemy import text, func  # SQLAlchemy for executing raw SQL queries


# Local application imports
from climate_policy_extractor.models import get_db_session, NDCDocumentModel, DocChunk
from climate_policy_extractor.utils import now_london_time  # Helper function for timestamps

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
    # logger.warning("Using starter code. Either edit this function or remove this warning if you are happy with the current implementation.")

    # Extract elements from the PDF
    try:
        # Use unstructured to extract elements from the PDF
        elements = partition_pdf(
            filename=pdf_path,
            strategy=unstructured_strategy,
            chunking_strategy="by_title",
            include_page_numbers=True,
            multipage_sections=True,             # Allow sections to span multiple pages
            overlap=50,
            overlap_all=True,
            max_characters=512,                  # Align with BERT's token limit
            new_after_n_chars=400,               # Soft limit to avoid hitting the hard max
            combine_text_under_n_chars=150,       # Combine small sections for better context
            )
        
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
                "page_number": element.metadata.page_number if hasattr(element, "metadata") and hasattr(element.metadata, "page_number") else 0,
                "metadata": {}
            }
            
            # # Add coordinates if available
            # if hasattr(element, "coordinates"):
            #     element_dict["metadata"]["coordinates"] = element.coordinates
                
            # Add any other metadata that might be useful
            if hasattr(element, "metadata"):
                for key, value in vars(element.metadata).items():
                    if key not in ["page_number", "coordinates", "_known_field_names"]:  # Skip already added metadata
                        element_dict["metadata"][key] = value
                
            result.append(element_dict)
        
        logger.info(f"Processed {len(result)} non-empty elements from {pdf_path}")
        return result
        
    except Exception as e:
        logger.error(f"Unable to extract text from PDF {pdf_path}: {e}")
        return []
    

def get_mean_embeddings(text, model, tokenizer) -> np.ndarray:
    """
    Extract the mean token embeddings for a given text using a tokenizer and model. This is more sophisticated than using the CLS token.
    It averages the embeddings of all tokens in the input text, making it better for comparing similarities.

    Args:
        text (str): The input text string.
        model_path (str): Path to the local model directory.

    Returns:
        numpy.ndarray: The mean token embeddings as a NumPy array.
    """

    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model = AutoModel.from_pretrained(model_path)

    # Tokenize the input string
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Pass the tokens through the model, without calculating gradients (since we only want embeddings)
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the mean token embeddings
    mean_embedding = outputs.last_hidden_state.mean(dim=1).detach()

    return mean_embedding.numpy()[0].tolist()  # Convert to list for JSON serialization

# Code to populate the chunks table in the database. 
# REFER TO MODELS.PY FOR THE TABLE SCHEMA

def bulk_process_and_store_chunks(json_file_path: Union[str, Path],
                              model: Any,
                              tokenizer: Any,
                              database_url: str,
                              batch_size: int = 1000) -> None:
    """
    Process document chunks from a JSON file and bulk insert them into the database.
    
    Args:
        json_file_path: Path to the JSON file containing document chunks
        model: The model to use for generating embeddings
        tokenizer: The tokenizer for the model
        database_url: Database connection URL
        batch_size: Number of records to insert in one batch
    """
    # Create database connection using the provided function
    session = get_db_session(database_url)
    
    # Load JSON data
    json_file_path = Path(json_file_path) if isinstance(json_file_path, str) else json_file_path
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Calculate total number of chunks for progress bar
    total_chunk_count = sum(len(chunks) for doc_id, chunks in data.items())
    
    # Set up progress bar
    progress_bar = tqdm(total=total_chunk_count, desc="Processing chunks", unit="chunk")
    
    # Process each document
    for doc_id, chunks in data.items():
        logger.info(f"Processing document: {doc_id}")
        
        # Check if document exists in the database
        document = session.query(NDCDocumentModel).filter_by(doc_id=doc_id).first()
        
        # If document doesn't exist, skip processing chunks and log a warning
        if not document:
            logger.warning(f"Document {doc_id} does not exist in the database. Skipping all associated chunks.")
            continue
        
        # Prepare batch of chunks for bulk insertion
        chunk_objects = []
        
        # Process each chunk
        for chunk in chunks:
            # Generate a unique ID for the chunk
            chunk_id = str(uuid.uuid4())
            
            # Get text content
            text = chunk["text"]
            
            # Generate embedding
            embedding = get_mean_embeddings(text, model, tokenizer)
            
            # Get page number if available
            page_number = chunk.get("page_number", 0)

            # Create metadata dict with available chunk info
            metadata = {
                "chunk_type": chunk.get("type"),
                "original_id": chunk.get("id"),
                "file_metadata": chunk.get("metadata", {})
            }
            
            # Create new chunk record
            doc_chunk = DocChunk(
                id=chunk_id,
                doc_id=doc_id,
                content=text,
                chunk_index=chunk["id"],  # Use the id from the JSON as chunk_index
                embedding=embedding,
                page_number=page_number,
                chunk_metadata=metadata,
                created_at=now_london_time(),
                updated_at=now_london_time()
            )
            
            # Add to batch
            chunk_objects.append(doc_chunk)
            progress_bar.update(1)
            
            # Commit in batches
            if len(chunk_objects) >= batch_size:
                session.bulk_save_objects(chunk_objects)
                session.commit()
                logger.info(f"Committed batch of {len(chunk_objects)} chunks")
                chunk_objects = []
        
        # Commit any remaining chunks in the last batch
        if chunk_objects:
            session.bulk_save_objects(chunk_objects)
            session.commit()
            logger.info(f"Committed final batch of {len(chunk_objects)} chunks for document {doc_id}")
    
    # Close progress bar and session
    progress_bar.close()
    logger.info(f"Total chunks processed and stored: {progress_bar.n}")
    session.close()

def populate_doc_chunks_from_json(
    json_file_path: Union[str, Path],      # Path to JSON file containing chunks
    db_session,
    delete_existing: bool = True,
    batch_size: int = 1000
) -> int:
    """
    Populate the doc_chunks table with data from a JSON file.

    Note: This function is a wrapper around populate_doc_chunks_from_dict.
    It loads the JSON file, processes it, and then calls the dict version.
    
    Args:
        json_file_path: Path to JSON file containing chunks
                        The JSON should be a list of dictionaries, each representing a chunk
                        Example: [{'id': 0, 'text': 'content', 'page_number': 1, 
                                  'metadata': {...}, 'doc_id': 'doc_id', 'embedding': [...]}, ...]
        db_session: SQLAlchemy database session
        delete_existing: Whether to delete existing chunks for documents
        batch_size: Number of chunks to insert in one batch
        
    Returns:
        Number of chunks inserted
    """
    
    logger = logging.getLogger(__name__)
    
    # Convert string path to Path object
    json_file_path = Path(json_file_path) if isinstance(json_file_path, str) else json_file_path
    
    # Load chunks from JSON file
    try:
        with open(json_file_path, 'r') as file:
            chunks_list = json.load(file)
        
        logger.info(f"Loaded {len(chunks_list)} chunks from {json_file_path}")
    except Exception as e:
        logger.error(f"Error loading JSON file {json_file_path}: {str(e)}")
        raise
    
    # Call the dict version with the loaded chunks
    return populate_doc_chunks_from_dict(
        chunks_list=chunks_list,
        db_session=db_session,
        delete_existing=delete_existing,
        batch_size=batch_size
    )

def populate_doc_chunks_from_dict(
        
    chunks_list: List[Dict[str, Any]],     # List of dictionaries, each representing a chunk
    db_session,
    delete_existing: bool = True,
    batch_size: int = 1000
) -> int:
    """
    Populate the doc_chunks table with data from a list of chunk dictionaries.
    
    Args:
        chunks_list: List of dictionaries, each containing a chunk's data
                     Example: [{'id': 0, 'text': 'content', 'page_number': 1, 
                               'metadata': {...}, 'doc_id': 'doc_id', 'embedding': [...]}, ...]
        db_session: SQLAlchemy database session
        delete_existing: Whether to delete existing chunks for documents
        batch_size: Number of chunks to insert in one batch
        
    Returns:
        Number of chunks inserted
    """
    logger = logging.getLogger(__name__)
    
    # Group chunks by doc_id for efficient processing
    chunks_by_doc_id = {}
    for chunk in chunks_list:
        doc_id = chunk.get("doc_id")
        if not doc_id:
            logger.warning(f"Skipping chunk with missing doc_id: {chunk.get('id')}")
            continue

        # Create new list for this doc_id if it doesn't exist    
        if doc_id not in chunks_by_doc_id:
            chunks_by_doc_id[doc_id] = []
        chunks_by_doc_id[doc_id].append(chunk)
    
    logger.info(f"Processing {len(chunks_list)} chunks for {len(chunks_by_doc_id)} documents")
    
    # Check if documents exist in the database
    existing_doc_ids = []
    for doc_id in chunks_by_doc_id:
        document = db_session.query(NDCDocumentModel).filter_by(doc_id=doc_id).first()
        if document:
            existing_doc_ids.append(doc_id)
        else:
            logger.warning(f"Document {doc_id} does not exist in the database. Skipping all associated chunks.")
    
    # Clean up existing chunks if requested
    if delete_existing and existing_doc_ids:    # Only delete if there are valid documents
        for doc_id in existing_doc_ids:
            deleted = db_session.query(DocChunk).filter(DocChunk.doc_id == doc_id).delete()
            logger.info(f"Deleted {deleted} existing chunks for document {doc_id}")
        db_session.commit()
    
    # Process and insert chunks
    inserted_count = 0
    try:
        # Process only chunks for existing documents
        doc_chunks = []
        
        for doc_id in existing_doc_ids:
            for chunk in chunks_by_doc_id[doc_id]:
                # Build metadata including page number
                metadata = {
                    "metadata": chunk.get("metadata", {}),
                    "type": chunk.get("type", "Unknown"),
                }
                
                # Create a new DocChunk object - use the chunk's id as chunk_index
                doc_chunk = DocChunk(
                    id=str(uuid.uuid4()),  # Generate a new UUID for the database ID
                    doc_id=doc_id,
                    content=chunk.get("text", ""),
                    chunk_index=int(chunk.get("id", 0)),  # Use the chunk's id as chunk_index
                    embedding=chunk.get("embedding"),     # Use the provided embedding
                    page_number=chunk.get("page_number", 0),  # Use the provided page number
                    chunk_metadata=metadata,
                    created_at=now_london_time(),
                    updated_at=now_london_time()
                )
                
                doc_chunks.append(doc_chunk)
                inserted_count += 1
                
                # Process in batches
                if len(doc_chunks) >= batch_size:
                    db_session.bulk_save_objects(doc_chunks)
                    db_session.commit()
                    logger.info(f"Inserted batch of {len(doc_chunks)} chunks")
                    doc_chunks = []
        
        # Insert any remaining chunks
        if doc_chunks:
            db_session.bulk_save_objects(doc_chunks)
            db_session.commit()
            logger.info(f"Inserted final batch of {len(doc_chunks)} chunks")
        
        logger.info(f"Successfully inserted {inserted_count} chunks")
        
        # Update the processed_at timestamp for all affected documents
        for doc_id in existing_doc_ids:
            # Use text() to explicitly mark SQL as text
            db_session.execute(
                text("UPDATE documents SET processed_at = :now WHERE doc_id = :doc_id"),
                {"now": now_london_time(), "doc_id": doc_id}
            )
        db_session.commit()
        
        return inserted_count
        
    except Exception as e:
        db_session.rollback()
        logger.error(f"Error inserting chunks into database: {str(e)}")
        raise

def count_chunks_for_doc(doc_id, db_session):
    """
    Count the number of rows in the doc_chunks table related to a specific doc_id.
    
    Args:
        doc_id (str): The document ID to query.
        db_session: SQLAlchemy database session.
    
    Returns:
        int: The number of related rows in the doc_chunks table.
    """
    chunk_count = db_session.query(func.count(DocChunk.id)).filter(DocChunk.doc_id == doc_id).scalar()
    return chunk_count

# Vector similarity search
def search_top_chunks_per_document(query_embedding, n_per_doc=10, min_similarity=0.7, db_session=None):
    """
    Search for the top N most similar chunks for each document.
    
    Args:
        query_embedding: The embedding vector for your query
        n_per_doc: Number of chunks to return per document
        min_similarity: Minimum similarity threshold (0-1)
        db_session: SQLAlchemy database session
        
    Returns:
        List of rows containing doc_id, chunk details, and similarity score
    """
    if db_session is None:
        raise ValueError("db_session must be provided")
    
    # Convert embedding to JSON for safe parameter passing
    embedding_json = json.dumps(query_embedding)
    
    # SQL query using window functions to get top N per document
    sql = text("""
        WITH similarity_results AS (
            SELECT 
                id,
                doc_id,
                content,
                chunk_index,
                page_number,
                chunk_metadata,
                1 - (embedding <=> :embedding) AS similarity,
                ROW_NUMBER() OVER (PARTITION BY doc_id ORDER BY embedding <=> :embedding) as rank
            FROM doc_chunks
            WHERE 1 - (embedding <=> :embedding) >= :min_similarity
        )
        SELECT 
            id,
            doc_id,
            content,
            chunk_index,
            page_number,
            chunk_metadata,
            similarity
        FROM similarity_results
        WHERE rank <= :n_per_doc
        ORDER BY doc_id, similarity DESC;
    """)
    
    # Execute query
    result = db_session.execute(
        sql, 
        {
            "embedding": embedding_json, 
            "n_per_doc": n_per_doc, 
            "min_similarity": min_similarity
        }
    )
    return result.fetchall()

# Get top N chunks for a specific country
def get_top_chunks_for_country(
        query_embedding, 
        country: str,
        n_per_doc=10,
        min_similarity=0.7,
        db_session=None
        ):
    """
    Search for the top N most similar chunks for documents from a specific country.
    
    Args:
        query_embedding: The embedding vector for your query
        country: Country name to filter documents by
        n_per_doc: Number of chunks to return per document
        min_similarity: Minimum similarity threshold (0-1)
        db_session: SQLAlchemy database session
        
    Returns:
        List of rows containing doc_id, chunk details, and similarity score
    """
    if db_session is None:
        raise ValueError("db_session must be provided")
    
    # Convert embedding to JSON for safe parameter passing
    embedding_json = json.dumps(query_embedding)
    
    # SQL query using window functions to get top N per document for the specific country
    # Join doc_chunks with documents table to filter by country
    sql = text("""
        WITH similarity_results AS (
            SELECT 
                c.id,
                c.doc_id,
                c.content,
                c.chunk_index,
                c.page_number,
                c.chunk_metadata,
                d.country,
                1 - (c.embedding <=> :embedding) AS similarity,
                ROW_NUMBER() OVER (PARTITION BY c.doc_id ORDER BY c.embedding <=> :embedding) as rank
            FROM doc_chunks c
            JOIN documents d ON c.doc_id = d.doc_id
            WHERE 1 - (c.embedding <=> :embedding) >= :min_similarity
            AND LOWER(d.country) = LOWER(:country)
        )
        SELECT 
            id,
            doc_id,
            content,
            chunk_index,
            page_number,
            chunk_metadata,
            country,
            similarity
        FROM similarity_results
        WHERE rank <= :n_per_doc
        ORDER BY similarity DESC;
    """)
    
    # Execute query
    result = db_session.execute(
        sql, 
        {
            "embedding": embedding_json, 
            "country": country,
            "n_per_doc": n_per_doc, 
            "min_similarity": min_similarity
        }
    )
    
    return result.all()

def get_top_chunks_for_query(query_embedding: List[float], db_session, country: str, n_per_doc: int = 20, min_similarity: float = 0.8):
    """
    Retrieve the top chunks for a given query embedding and country.

    Args:
        query_embedding (List[float]): The embedding vector for the query.
        db_session: SQLAlchemy database session.
        country (str): The country name to filter documents by.
        n_per_doc (int): Number of chunks to retrieve per document. Default is 20.
        min_similarity (float): Minimum similarity threshold. Default is 0.8.

    Returns:
        List: A list of results containing the top chunks for the query and country.
    """
    # Get top chunks for the specified country
    results = get_top_chunks_for_country(
        query_embedding=query_embedding,
        country=country,
        n_per_doc=n_per_doc,
        min_similarity=min_similarity,
        db_session=db_session
    )
    
    return results


# LLM functions

# Format retrieval results into prompt
def format_context(results):
    context = []
    for row in results:
        context.append(f"[Doc ID: {row.doc_id}, Page: {row.page_number}, Chunk ID: {row.chunk_index}], Content: {row.content}\n]")
    return "\n\n".join(context)

# Generate LLM response based on country.
def get_country_specific_response(client, query: str, context: str, country: str) -> str:
    """
    Generate a response from the LLM based on a prompt and a specific country.

    Args:
        client: The OpenAI client instance.
        prompt (str): The input prompt for the LLM.
        country (str): The country to tailor the response to.

    Returns:
        str: The response message from the LLM.
    """
    # Combine the prompt with the country context

    system_prompt = """
    You are an expert in climate policy analysis.
    Answer the question SOLELY based on the provided document chunks.
    Do not make any assumptions or rely on external knowledge.
    Sources come in the form of [Doc ID: <id>, Page: <page>, Chunk ID: <id>, Content: <chunk>].
    Every time you use a source, cite it using the format [Doc ID: <id>, Chunk ID: <id>, Page: <page>].
    """

    full_prompt = f"{system_prompt}\n\n{context}\n\n{query}\n\nPlease tailor your response to the context of {country}."

    # Generate the completion
    completion = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        messages=[
            {
                "role": "user",
                "content": full_prompt
            }
        ],
        temperature=0.6,
    )

    # Parse and return the response
    response = json.loads(completion.to_json())
    return response['choices'][0]['message']['content']