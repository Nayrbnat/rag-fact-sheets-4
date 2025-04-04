"""
Process downloaded documents, extract text, and create chunks
"""
import os
import sys
import json
import glob
import logging
import threading
import concurrent.futures
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import signal

# Add the parent directory to the path so we can import from the climate_policy_extractor package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from climate_policy_extractor.models import NDCDocumentModel
from climate_policy_extractor.logging import get_logger
# Import only the specific functions we need from utils
from notebooks.utils import extract_text_from_pdf, extract_text_from_docx, chunk_document_by_sentences

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = get_logger(__name__)
logger.info("Starting document processing script")

# Define a custom JSON encoder to handle non-serializable types
class ExtractedDataEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle CoordinatesMetadata specifically
        if obj.__class__.__name__ == 'CoordinatesMetadata':
            return {
                'points': getattr(obj, 'points', []),
                'system': getattr(obj, 'system', ''),
                'x1': getattr(obj, 'x1', 0),
                'y1': getattr(obj, 'y1', 0),
                'x2': getattr(obj, 'x2', 0),
                'width': getattr(obj, 'width', 0),
                'height': getattr(obj, 'height', 0)
            }
        
        # Handle other types
        if hasattr(obj, '__dict__'): return obj.__dict__
        try: return dict(obj)
        except: pass
        try: return list(obj)
        except: pass
        return str(obj)

def get_document_metadata(engine, doc_id):
    """Get document metadata from database"""
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

def update_document_processed(engine, doc_id, chunks=None):
    """Mark document as processed in the database"""
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        doc = session.query(NDCDocumentModel).filter(NDCDocumentModel.doc_id == doc_id).first()
        if doc:
            doc.processed_at = datetime.now()
            if chunks:
                try:
                    doc.chunks = json.dumps(chunks, cls=ExtractedDataEncoder)
                except Exception as e:
                    logger.error(f"Error serializing chunks for database: {e}")
            session.commit()
            logger.info(f"Successfully updated database for document: {doc_id}")
            return True
        else:
            logger.warning(f"Document not found in database: {doc_id}")
    except Exception as e:
        logger.error(f"Database update error: {e}")
        session.rollback()
    finally:
        session.close()
    
    return False

def with_timeout(func, timeout=20):
    """Execute function with timeout using a daemon thread that won't block execution"""
    import threading
    result = [None]
    exception = [None]
    completed = [False]
    
    def worker():
        try:
            result[0] = func()
            completed[0] = True
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=worker)
    thread.daemon = True  # Set as daemon so it won't block program exit
    thread.start()
    thread.join(timeout)
    
    if completed[0]:
        return result[0]
    elif exception[0]:
        logger.error(f"Error in processing: {exception[0]}")
        return {'status': 'failed', 'reason': str(exception[0])}
    else:
        logger.warning(f"Processing timed out after {timeout} seconds - skipping document")
        return {'status': 'skipped', 'reason': f'timeout_after_{timeout}_seconds'}

def merge_short_chunks(elements, min_length=20):
    """Merge chunks that are shorter than the minimum length threshold."""
    if not elements:
        return elements
    
    merged_elements = []
    current_chunk = None
    
    for element in elements:
        text = element.get('text', '')
        
        if current_chunk is None:
            current_chunk = element
        elif len(current_chunk.get('text', '')) < min_length:
            # Merge with current chunk
            current_chunk['text'] = current_chunk.get('text', '') + ' ' + text
            # Keep the metadata from the first chunk
        else:
            # Current chunk is long enough, add to results
            merged_elements.append(current_chunk)
            current_chunk = element
    
    # Don't forget the last chunk
    if current_chunk is not None:
        merged_elements.append(current_chunk)
    
    return merged_elements

def process_document(document_path, output_dir, engine=None, metadata=None, chunk_size=512, overlap=2, timeout=20, ocr_timeout=60):
    """Process a document file and extract text chunks with timeout"""
    def extract_and_process():
        try:
            # Get paths
            base_filename = os.path.basename(document_path).split('.')[0]
            json_output_path = os.path.join(output_dir, 'json', f"{base_filename}_text.json")
            chunks_output_path = os.path.join(output_dir, 'chunks', f"{base_filename}_chunks.json")
            
            # Skip if already processed
            if os.path.exists(json_output_path) and os.path.exists(chunks_output_path):
                return {'status': 'skipped', 'reason': 'already_processed', 'paths': [json_output_path, chunks_output_path]}
            
            # Extract text based on file type - always use fast strategy 
            file_ext = os.path.splitext(document_path)[1].lower()
            elements = None
            extraction_strategy = "fast"  # Track the strategy that succeeded
            
            # Helper function for extraction with timeout
            def extract_with_strategy(strategy, custom_timeout):
                nonlocal elements
                logger.info(f"Attempting to extract text from {document_path} using {strategy} strategy (timeout: {custom_timeout}s)")
                
                def extraction_func():
                    try:
                        if file_ext == '.pdf':
                            if strategy == "ocr_only":
                                logger.info(f"Starting OCR process with ocr_only strategy for {document_path}")
                                start_time = datetime.now()
                                result = extract_text_from_pdf(document_path, strategy=strategy)
                                end_time = datetime.now()
                                duration = (end_time - start_time).total_seconds()
                                page_count = len(set([elem.get('metadata', {}).get('page_number', 0) for elem in result])) if result else 0
                                logger.info(f"OCR ocr_only completed in {duration:.2f}s. Extracted {len(result) if result else 0} elements across {page_count} pages")
                                return result
                            else:
                                return extract_text_from_pdf(document_path, strategy=strategy)
                        elif file_ext == '.docx':
                            return extract_text_from_docx(document_path)
                        else:
                            logger.warning(f"Unsupported file type: {file_ext}, attempting to process as PDF with {strategy} strategy")
                            return extract_text_from_pdf(document_path, strategy=strategy)
                    except Exception as e:
                        if strategy == "ocr_only":
                            logger.error(f"OCR ocr_only extraction error: {str(e)}", exc_info=True)
                        else:
                            logger.error(f"Error in {strategy} extraction: {str(e)}", exc_info=True)
                        return None
                
                return with_timeout(extraction_func, timeout=custom_timeout)
            
            # First try fast extraction
            elements = extract_with_strategy("fast", timeout)
            
            # If extraction failed or timed out, retry with ocr_only strategy
            if not elements or isinstance(elements, dict) and elements.get('status') in ('failed', 'skipped'):
                if isinstance(elements, dict):
                    logger.warning(f"Fast extraction {elements.get('status', 'failed')}: {elements.get('reason', 'unknown reason')}")
                else:
                    logger.warning(f"Fast extraction returned no elements, attempting ocr_only strategy")
                
                # Use longer timeout for ocr_only extraction
                logger.info(f"Retrying with ocr_only strategy using extended timeout of {ocr_timeout}s")
                elements = extract_with_strategy("ocr_only", ocr_timeout)
                extraction_strategy = "ocr_only"
                
                if not elements:
                    logger.error(f"OCR extraction returned no elements")
                    return {'status': 'failed', 'reason': 'no_text_extracted', 'extraction_strategy': 'ocr_only_failed'}
                elif isinstance(elements, dict) and elements.get('status') in ('failed', 'skipped'):
                    logger.error(f"OCR extraction {elements.get('status')}: {elements.get('reason')}")
                    return {'status': 'failed', 'reason': f"ocr_only_extraction_{elements.get('reason', 'failed')}", 'extraction_details': elements}
                else:
                    logger.info(f"OCR extraction successful with {len(elements)} elements")
                    # Add additional debugging information about extracted content
                    if elements:
                        page_count = len(set([elem.get('metadata', {}).get('page_number', 0) for elem in elements]))
                        avg_text_length = sum(len(elem.get('text', '')) for elem in elements) / len(elements) if elements else 0
                        logger.info(f"OCR stats: {page_count} pages, avg element length: {avg_text_length:.1f} chars")
            else:
                logger.info(f"Fast extraction successful with {len(elements)} elements")
            
            # Get metadata if needed
            doc_metadata = metadata
            if engine and not doc_metadata:
                doc_metadata = get_document_metadata(engine, base_filename)
            
            # Merge short chunks that don't constitute full sentences (using 20 chars as minimum threshold)
            min_sentence_length = 20
            elements = merge_short_chunks(elements, min_length=min_sentence_length)
            
            # Chunk text using the function from utils.py - updated parameter name to match function signature
            chunks = chunk_document_by_sentences(elements, max_chunk_size=chunk_size, overlap=overlap)
            if not chunks:
                return {'status': 'failed', 'reason': 'no_chunks_created', 'extraction_strategy': extraction_strategy}
            
            # Add metadata to elements
            for item in elements:
                if 'metadata' not in item:
                    item['metadata'] = {}
                
                # Add document metadata from database
                if doc_metadata:
                    item['metadata'].update(doc_metadata)
                    
                # Add filename
                item['metadata']['filename'] = os.path.basename(document_path)
            
            # Add metadata to chunks while preserving paragraph info
            for item in chunks:
                if 'metadata' not in item:
                    item['metadata'] = {}
                
                # Add document metadata first
                if doc_metadata:
                    item['metadata'].update(doc_metadata)
                
                # Add filename
                item['metadata']['filename'] = os.path.basename(document_path)

                # Preserve source paragraph information that came from chunking
                for key in ['paragraph_number', 'page_number', 'element_types']:
                    if key in item.get('metadata', {}):
                        item['metadata'][key] = item['metadata'][key]
            
            # Save output
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(elements, f, cls=ExtractedDataEncoder, ensure_ascii=False, indent=2)
                
            with open(chunks_output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, cls=ExtractedDataEncoder, ensure_ascii=False, indent=2)
            
            # Update database
            db_updated = False
            if engine:
                logger.info(f"Updating database for document: {base_filename}")
                db_updated = update_document_processed(engine, base_filename, chunks)
                if not db_updated:
                    logger.warning(f"Failed to update database for document: {base_filename}")
            
            return {
                'status': 'success',
                'paths': [json_output_path, chunks_output_path],
                'chunk_count': len(chunks),
                'db_updated': db_updated,
                'extraction_strategy': extraction_strategy
            }
            
        except Exception as e:
            logger.error(f"Error processing document {document_path}: {str(e)}", exc_info=True)
            return {'status': 'failed', 'reason': str(e)}
    
    return with_timeout(extract_and_process, timeout=max(timeout, ocr_timeout) + 10)  # Add extra time for the overall process

def main():
    """Main function to process documents"""
    load_dotenv()
    db_url = os.getenv('DATABASE_URL', 'postgresql://climate:climate@localhost:5432/climate')
    data_dir = os.path.abspath(os.getenv('DATA_DIR', 'data'))
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Process downloaded documents")
    parser.add_argument("--chunk-size", type=int, default=512, help="Maximum chunk size in characters")
    parser.add_argument("--overlap", type=int, default=2, help="Number of sentences to overlap between chunks")
    parser.add_argument("--output", "-o", help="Output directory for JSON files")
    parser.add_argument("--pdfs-dir", help="Directory containing PDF files")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds for processing each document with fast strategy (default: 20)")
    parser.add_argument("--ocr-timeout", type=int, default=60, help="Timeout in seconds for ocr_only extraction (default: 60)")
    args = parser.parse_args()
    
    try:
        # Create the database engine
        engine = create_engine(db_url)
        
        # Set up directories
        output_dir = args.output or os.path.join(data_dir, 'processed')
        pdfs_dir = args.pdfs_dir or os.path.join(data_dir, 'pdfs')
        
        # Ensure output directories exist
        for subdir in ['json', 'chunks']:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
        
        # Find PDF files
        pdf_files = []
        search_dirs = [
            pdfs_dir,
            os.path.join(data_dir, 'pdfs'),
            './data/pdfs',
            '../data/pdfs'
        ]
        for directory in search_dirs:
            if os.path.exists(directory):
                pdf_files = glob.glob(os.path.join(directory, '*.pdf'))
                if pdf_files:
                    pdfs_dir = directory
                    print(f"Found {len(pdf_files)} PDFs in: {directory}")
                    break
        if not pdf_files:
            print("ERROR: No PDF files found in any search location")
            return
            
        # Track processing results and all document IDs
        results = {
            "success": 0,  
            "skipped": [], 
            "failed": [], 
            "processed_doc_ids": set()
        }
        
        # Process each PDF
        for i, pdf_path in enumerate(tqdm(pdf_files, desc="Processing PDFs")):
            doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
            print(f"\nProcessing document {i+1}/{len(pdf_files)}: {doc_id}")
            
            # Track that we attempted to process this document
            results["processed_doc_ids"].add(doc_id)
            try:
                # Get metadata
                metadata = get_document_metadata(engine, doc_id)
                
                # Process the document with separate timeouts for fast and ocr_only strategies
                result = process_document(
                    document_path=pdf_path,
                    output_dir=output_dir,
                    engine=engine,
                    metadata=metadata,
                    chunk_size=args.chunk_size,
                    overlap=args.overlap,
                    timeout=args.timeout,
                    ocr_timeout=args.ocr_timeout
                )
                
                # Track result
                if result:
                    if result['status'] == 'success':
                        results["success"] += 1
                        db_status = "DB updated" if result.get('db_updated') else "DB update failed"
                        strategy = result.get('extraction_strategy', 'unknown')
                        print(f"Successfully processed with {result.get('chunk_count', 0)} chunks using {strategy} strategy. {db_status}")
                    elif result['status'] == 'skipped':
                        results["skipped"].append((doc_id, result['reason']))
                        print(f"Skipped ({result['reason']})")
                    else:  # failed
                        results["failed"].append((doc_id, result.get('reason', 'unknown error')))
                        print(f"Failed: {result.get('reason', 'unknown error')}")
                else:
                    results["failed"].append((doc_id, "No result returned"))
                    print("Failed: No result returned")
            except Exception as e:
                logger.error(f"Uncaught exception processing {doc_id}: {str(e)}", exc_info=True)
                results["failed"].append((doc_id, f"Uncaught exception: {str(e)}"))
                print(f"Failed with uncaught exception: {str(e)}")
        
        # Check if all expected output files were created
        expected_doc_ids = set([os.path.splitext(os.path.basename(path))[0] for path in pdf_files])
        json_dir = os.path.join(output_dir, 'json')
        chunks_dir = os.path.join(output_dir, 'chunks')
        json_files = set([f.split('_text.json')[0] for f in os.listdir(json_dir) if f.endswith('_text.json')])
        chunk_files = set([f.split('_chunks.json')[0] for f in os.listdir(chunks_dir) if f.endswith('_chunks.json')])
        
        # Find documents that have no json or chunks output
        missing_json = expected_doc_ids - json_files
        missing_chunks = expected_doc_ids - chunk_files
        
        # Add these to the failed list if they weren't already tracked
        for doc_id in missing_json:
            if doc_id not in [d for d, _ in results["failed"]] and doc_id not in [d for d, _ in results["skipped"]]:
                results["failed"].append((doc_id, "No JSON output was generated"))
        for doc_id in missing_chunks:
            if doc_id not in [d for d, _ in results["failed"]] and doc_id not in [d for d, _ in results["skipped"]]:
                results["failed"].append((doc_id, "No chunks output was generated"))
        
        # Print summary
        print("\n" + "="*50)
        print("DOCUMENT PROCESSING SUMMARY")
        print("="*50)
        print(f"Total: {len(pdf_files)} | Success: {results['success']} | " +
              f"Skipped: {len(results['skipped'])} | Failed: {len(results['failed'])}")
        
        # Log all failures and skips
        failure_log_path = os.path.join(output_dir, 'processing_failures.json')
        with open(failure_log_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_pdfs': len(pdf_files),
                'successful': results['success'],
                'skipped': [{'doc_id': doc_id, 'reason': reason} for doc_id, reason in results["skipped"]],
                'failed': [{'doc_id': doc_id, 'reason': reason} for doc_id, reason in results["failed"]],
                'missing_json': list(missing_json),
                'missing_chunks': list(missing_chunks)
            }, f, indent=2, ensure_ascii=False)
        print(f"\nProcessing summary written to: {failure_log_path}")
        
        if results["failed"] or results["skipped"]:
            print(f"Failed documents: {len(results['failed'])}")
            print(f"Skipped documents: {len(results['skipped'])}")
            print(f"Details written to: {failure_log_path}")
        
        # Check for unexpected discrepancies
        total_accounted = results['success'] + len(results['failed']) + len(results['skipped'])
        if total_accounted != len(pdf_files):
            print(f"\nWARNING: Discrepancy in totals - PDF files: {len(pdf_files)}, Accounted for: {total_accounted}")
            print(f"Missing: {len(pdf_files) - total_accounted}")
    except Exception as e:
        import traceback
        print(f"ERROR: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()