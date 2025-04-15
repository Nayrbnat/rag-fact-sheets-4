"""
Common management tasks for the climate policy extractor.
"""
import os
import click
import subprocess
import shutil
import glob
import json
from tqdm import tqdm
import numpy as np

from pathlib import Path
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from climate_policy_extractor.models import Base, get_db_session, NDCDocumentModel, DocChunk
from climate_policy_extractor.spiders.ndc_spider import NDCSpider
from climate_policy_extractor.downloaders import process_downloads
from climate_policy_extractor.utils import now_london_time

# Chunking and embedding utils
from notebooks.utils import extract_text_data_from_pdf, get_mean_embeddings, populate_doc_chunks_from_json

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')
DOWNLOAD_DIR = os.getenv('DOWNLOAD_DIR', 'data/pdfs')

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in environment variables")

@click.group()
def cli():
    """Management commands for the climate policy extractor."""
    pass

def create_engine_and_extension():
    """Create the database engine and vector extension."""
    engine = create_engine(DATABASE_URL)
    click.echo("Creating vector extension...")
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
    return engine

@cli.command()
@click.option('--download/--no-download', default=False, 
              help='Whether to download PDFs (default: False)')
def crawl(download):
    """Run the NDC spider to collect document metadata."""
    settings = get_project_settings()
    
    if not download:
        # Remove the DocumentDownloadPipeline if --no-download
        settings['ITEM_PIPELINES'] = {
            'climate_policy_extractor.pipelines.PostgreSQLPipeline': 300
        }
        click.echo("Running crawler (metadata only, no downloads)...")
    else:
        click.echo("Running crawler with document downloads...")
    process = CrawlerProcess(settings)
    process.crawl(NDCSpider)
    process.start()

@cli.command()
@click.option('--tail', '-t', is_flag=True, help='Tail the log file')
@click.option('--clear', '-c', is_flag=True, help='Clear the log file')
@click.option('--lines', '-n', default=10, help='Number of lines to show')
def logs(tail, clear, lines):
    """View or manage the scrapy log file."""
    settings = get_project_settings()
    log_file = settings.get('LOG_FILE')
    
    if not log_file or not os.path.exists(log_file):
        click.echo("No log file found.")
        return
    
    if clear:
        if click.confirm('Are you sure you want to clear the log file?'):
            open(log_file, 'w').close()
            click.echo("Log file cleared.")
        return
    
    if tail:
        # Use subprocess to tail the file
        click.echo(f"Tailing log file (Ctrl+C to stop)...")
        try:
            subprocess.run(['tail', '-f', log_file])
        except KeyboardInterrupt:
            click.echo("\nStopped tailing log file.")
        except FileNotFoundError:
            # For Windows systems where tail isn't available
            click.echo("Tail command not available. Showing last few lines instead:")
            with open(log_file, 'r') as f:
                click.echo(''.join(f.readlines()[-lines:]))
    else:
        # Show last N lines
        with open(log_file, 'r') as f:
            click.echo(''.join(f.readlines()[-lines:]))

@cli.command()
def recreate_db():
    """Recreate the database from scratch (WARNING: destructive operation)."""
    engine = create_engine_and_extension()
    
    # Drop all tables
    click.echo("Dropping all tables...")
    Base.metadata.drop_all(engine)
    
    # Recreate all tables
    click.echo("Recreating all tables...")
    Base.metadata.create_all(engine)
    
    click.echo("Database recreated successfully!")

@cli.command()
def init_db():
    """Initialize the database (safe operation, won't drop existing tables)."""
    engine = create_engine_and_extension()
    
    # Create tables if they don't exist
    click.echo("Creating tables if they don't exist...")
    Base.metadata.create_all(engine)
    
    click.echo("Database initialized successfully!")

@cli.command()
def drop_db():
    """Drop all tables from the database (WARNING: destructive operation)."""
    if click.confirm('Are you sure you want to drop all tables? This cannot be undone!'):
        engine = create_engine(DATABASE_URL)
        
        # Drop all tables
        click.echo("Dropping all tables...")
        Base.metadata.drop_all(engine)
        
        click.echo("Database tables dropped successfully!")
    else:
        click.echo("Operation cancelled.")

@cli.command()
def list_tables():
    """List all tables in the database."""
    engine = create_engine(DATABASE_URL)
    
    # Get all table names
    with engine.connect() as conn:
        tables = Base.metadata.tables.keys()
        
        if not tables:
            click.echo("No tables found in the database.")
            return
        
        click.echo("\nTables in the database:")
        for table in tables:
            # Get row count
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = result.scalar()
            click.echo(f"- {table} ({count} rows)")

@cli.command()
@click.option('--force/--no-force', default=False, 
              help='Force download even if file exists')
def download(force):
    """Download PDF documents for records in the database."""
    session = get_db_session(DATABASE_URL)
    
    try:
        click.echo("Starting download process...")
        total, successful = process_downloads(session, DOWNLOAD_DIR)    # downloaders.py function
        
        if total == 0:
            click.echo("No new documents to download.")
            return
        
        # Add a newline after progress bars
        click.echo("\nDownload summary:")
        click.echo(f"- Total documents processed: {total}")
        click.echo(f"- Successfully downloaded: {successful}")
        if total - successful > 0:
            click.echo(f"- Failed downloads: {total - successful}")
        
    except Exception as e:
        click.echo(f"\nError processing downloads: {e}")
        raise
    finally:
        session.close()

@cli.command()
@click.argument('pdf_dir', type=click.Path(exists=True))
def import_pdfs(pdf_dir):
    """Import manually downloaded PDFs from a directory.
    
    The PDFs should be named with their doc_ids (e.g., 'liberia_english_20220601.pdf').
    """
    session = get_db_session(DATABASE_URL)
    imported = 0
    skipped = 0
    
    try:
        # Convert paths to absolute paths for accurate comparison
        pdf_dir = os.path.abspath(pdf_dir)
        download_dir = os.path.abspath(DOWNLOAD_DIR)
        
        # Get all PDF files in the directory
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
        if not pdf_files:
            click.echo("No PDF files found in the specified directory.")
            return
            
        # Check if directories are the same
        same_directory = pdf_dir == download_dir
        if same_directory:
            click.echo(f"Source directory is the same as target directory ({pdf_dir}). Files will not be copied.")
        
        with click.progressbar(pdf_files, label='Importing PDFs') as files:
            for filename in files:
                # Use a more robust way to extract doc_id
                doc_id = os.path.splitext(filename)[0]
                # Find corresponding document in database
                doc = session.query(NDCDocumentModel).filter_by(doc_id=doc_id).first()
                
                if not doc:
                    click.echo(f"\nSkipping {filename}: No matching doc_id in database")
                    skipped += 1
                    continue
                
                if doc.downloaded_at:
                    click.echo(f"\nSkipping {filename}: Already marked as downloaded")
                    skipped += 1
                    continue
                
                # Get full paths
                src_path = os.path.join(pdf_dir, filename)
                dst_path = os.path.join(download_dir, filename)
                
                # Ensure download directory exists
                os.makedirs(download_dir, exist_ok=True)
                
                # Copy the file only if directories are different
                if not same_directory:
                    shutil.copy2(src_path, dst_path)
                    file_path = dst_path
                else:
                    # Use the source path directly if directories are the same
                    file_path = src_path
                
                # Normalize file path for storage (use forward slashes)
                normalized_path = file_path.replace('\\', '/')
                
                # Update document record
                doc.file_path = normalized_path
                doc.file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
                doc.downloaded_at = now_london_time()
                doc.download_error = None
                
                session.commit()
                imported += 1
        
        click.echo(f"\nImport complete:")
        click.echo(f"- Successfully imported: {imported}")
        if skipped > 0:
            click.echo(f"- Skipped: {skipped}")
            
    except Exception as e:
        click.echo(f"\nError importing PDFs: {e}")
        raise
    finally:
        session.close()

@cli.command()
def download_status():
    """Show status of document downloads."""
    session = get_db_session(DATABASE_URL)
    
    try:
        total = session.query(NDCDocumentModel).count()
        downloaded = session.query(NDCDocumentModel).filter(
            NDCDocumentModel.downloaded_at.isnot(None)
        ).count()
        failed = session.query(NDCDocumentModel).filter(
            NDCDocumentModel.download_attempts >= 3,
            NDCDocumentModel.downloaded_at.is_(None)
        ).count()
        pending = total - downloaded - failed
        
        click.echo("\nDownload Status:")
        click.echo(f"- Total documents: {total}")
        click.echo(f"- Successfully downloaded: {downloaded}")
        click.echo(f"- Failed (max attempts): {failed}")
        click.echo(f"- Pending download: {pending}")
        
        if failed > 0:
            click.echo("\nFailed documents:")
            failed_docs = session.query(NDCDocumentModel).filter(
                NDCDocumentModel.download_attempts >= 3,
                NDCDocumentModel.downloaded_at.is_(None)
            ).all()
            for doc in failed_docs:
                click.echo(f"- {doc.doc_id}: {doc.download_error}")
                
    finally:
        session.close()

# Text extraction and chunking functions here:
@cli.command()
@click.argument('pdf_dir',
                type=click.Path(exists=True),
                )
@click.option('--first-n-pdfs', '-n', type=int, default=0, 
                help='Process only the first N PDF files (default: process all files)')
def extract_text_and_chunk_pdfs(pdf_dir, first_n_pdfs):
    """
    Extracts text and chunks content of PDFs in the specified directory.
    Saves the extracted text in a JSON file.
    This function is run after the PDFs have been downloaded.
    
    Warning: WILL TAKE A LONG TIME!
    """

    files = glob.glob(os.path.join(pdf_dir, '*'))
    len(files)

    # For debugging: only add the first N files to reduce memory and runtime.
    if first_n_pdfs > 0:
        files = files[:first_n_pdfs]

    # TODO: You might want to do this in batches if you have too many files, or it will take a long time.
    doc_text = {os.path.splitext(os.path.basename(file))[0]: extract_text_data_from_pdf(file, unstructured_strategy="fast")
                for file in tqdm(files)}

    len(doc_text)

    with open('../data/json/all_chunks_bert.json', 'w') as f:
        json.dump(doc_text, f, indent=4)


# # Postgres updating here:
# @cli.command()
# @click.argument('doc_id', type=str)
# @click.option('--chunk-size', '-c', default=500, help='Size of document chunks')
# @click.option('--chunk-overlap', '-o', default=50, help='Overlap between chunks')
# def update_doc_chunks(doc_id, chunk_size, chunk_overlap):
#     """
#     Update a document in the database with document chunks.
#     This updates the chunks column, which is a JSONB field.
    
#     """
    
#     session = get_db_session(DATABASE_URL)
    
#     try:
#         # Fetch the document
#         doc = session.query(NDCDocumentModel).filter_by(doc_id=doc_id).first()
        
#         if not doc:
#             click.echo(f"Document with ID '{doc_id}' not found.")
#             return
            
#         if not doc.downloaded_at or not doc.file_path:
#             click.echo(f"Document '{doc_id}' has not been downloaded yet.")
#             return
            
#         click.echo(f"Processing document: {doc.doc_id}")
        
#         # Delete existing chunks if any
#         with session.begin():
#             if doc.chunks:
#                 click.echo(f"Removing {len(doc.chunks)} existing chunks...")
#                 for chunk in doc.chunks:
#                     session.delete(chunk)
#                 session.commit()
        
#         # Chunk the document and add to database
#         click.echo(f"Creating new chunks (size={chunk_size}, overlap={chunk_overlap})...")
#         # Change chunk document function.
#         chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
#         if not chunks:
#             click.echo("No chunks were created. Check if the document is readable.")
#             return
            
#         with click.progressbar(chunks, label='Saving chunks') as chunks_iter:
#             for chunk in chunks_iter:
#                 session.add(chunk)
#             session.commit()
            
#         click.echo(f"Successfully created {len(chunks)} chunks for document '{doc_id}'")
        
#     except Exception as e:
#         click.echo(f"Error updating document chunks: {e}")
#         raise
#     finally:
#         session.close()

if __name__ == '__main__':
    cli()