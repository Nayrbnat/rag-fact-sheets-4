"""
Common management tasks for the climate policy extractor.
"""
import os
import click
import subprocess
import shutil
import sys

from pathlib import Path
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect

from climate_policy_extractor.models import Base, get_db_session, NDCDocumentModel, DocChunk, Vector
from climate_policy_extractor.spiders.ndc_spider import NDCSpider
from climate_policy_extractor.downloaders import process_downloads
from climate_policy_extractor.utils import now_london_time
from climate_policy_extractor.logging import get_logger

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')
DOWNLOAD_DIR = os.getenv('DOWNLOAD_DIR', 'data/pdfs')

# Initialize logger
logger = get_logger(__name__)

# Register the Vector type with SQLAlchemy
from sqlalchemy.dialects import postgresql
postgresql.base.ischema_names['vector'] = Vector

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
@click.option('--operation', '-o', type=click.Choice(['init', 'recreate', 'drop', 'drop_chunks', 'update_chunks']), 
              required=True, help='Database operation to perform')
@click.option('--force/--no-force', default=False, help='Skip confirmation for destructive operations')
def manage_db(operation, force):
    """Manage database operations with a single command.
    
    Operations:
    - init: Initialize the database (create tables if they don't exist)
    - recreate: Drop all tables and recreate them
    - drop: Drop all tables
    - drop_chunks: Drop only the doc_chunks table
    - update_chunks: Update doc_chunks table to match current model definition
    """
    # Import Base at the beginning of the function
    from climate_policy_extractor.models import Base
    engine = create_engine_and_extension()
    
    # Handle different operations
    if operation == 'init':
        click.echo("Creating tables if they don't exist...")
        Base.metadata.create_all(engine)
        click.echo("Database initialized successfully!")
        
    elif operation == 'recreate':
        if force or click.confirm('Are you sure you want to recreate the database? This will delete ALL data!'):
            click.echo("Dropping all tables...")
            Base.metadata.drop_all(engine)
            click.echo("Recreating all tables...")
            Base.metadata.create_all(engine)
            click.echo("Database recreated successfully!")
        else:
            click.echo("Operation cancelled.")
            
    elif operation == 'drop':
        if force or click.confirm('Are you sure you want to drop all tables? This cannot be undone!'):
            click.echo("Dropping all tables...")
            Base.metadata.drop_all(engine)
            click.echo("Database tables dropped successfully!")
        else:
            click.echo("Operation cancelled.")
            
    elif operation == 'drop_chunks':
        if force or click.confirm('Are you sure you want to drop the doc_chunks table? This will delete all stored chunks!'):
            with engine.connect() as conn:
                conn.execute(text("DROP TABLE IF EXISTS doc_chunks;"))
                conn.commit()
            click.echo("doc_chunks table dropped successfully!")
            
            if click.confirm('Do you want to recreate the empty doc_chunks table?'):
                from climate_policy_extractor.models import Base
                # Get only the doc_chunks table and create it
                doc_chunks_table = Base.metadata.tables['doc_chunks']
                doc_chunks_table.create(engine)
                click.echo("doc_chunks table recreated!")
        else:
            click.echo("Operation cancelled.")
            
    elif operation == 'update_chunks':
        if force or click.confirm('Are you sure you want to drop and recreate the doc_chunks table?'):
            with engine.connect() as conn:
                conn.execute(text("DROP TABLE IF EXISTS doc_chunks;"))
                conn.commit()
            
            # Create the table with the new structure
            from climate_policy_extractor.models import DocChunk
            DocChunk.__table__.create(engine)
            
            # Add vector indexes only - columns are already created by the model
            with engine.connect() as conn:
                # Create the indexes for vector columns
                conn.execute(text("CREATE INDEX ON doc_chunks USING ivfflat (embedding vector_cosine_ops);"))
                conn.execute(text("CREATE INDEX ON doc_chunks USING ivfflat (word2vec_embedding vector_cosine_ops);"))
                conn.commit()
            
            click.echo("doc_chunks table updated successfully!")
        else:
            click.echo("Operation cancelled.")

@cli.command()
def recreate_db():
    """Recreate the database from scratch (WARNING: destructive operation)."""
    ctx = click.get_current_context()
    ctx.invoke(manage_db, operation='recreate', force=False)

@cli.command()
def init_db():
    """Initialize the database (safe operation, won't drop existing tables)."""
    ctx = click.get_current_context()
    ctx.invoke(manage_db, operation='init', force=True)

@cli.command()
def drop_db():
    """Drop all tables from the database (WARNING: destructive operation)."""
    ctx = click.get_current_context()
    ctx.invoke(manage_db, operation='drop', force=False)

@cli.command()
def drop_doc_chunks():
    """Drop only the doc_chunks table from the database."""
    ctx = click.get_current_context()
    ctx.invoke(manage_db, operation='drop_chunks', force=False)

@cli.command()
def update_doc_chunks_table():
    """Update doc_chunks table to match models.py definition."""
    ctx = click.get_current_context()
    ctx.invoke(manage_db, operation='update_chunks', force=False)

@cli.command()
def setup_db():
    """Set up the database with pgVector extension and required tables."""
    engine = create_engine(DATABASE_URL)
    
    click.echo(f"Connecting to database: {DATABASE_URL}")
    
    # Create the pgvector extension
    click.echo("Creating pgvector extension if it doesn't exist")
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
    
    # Check if doc_chunks table exists and has correct column types
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    # If doc_chunks exists, check column types
    table_needs_recreation = False
    if 'doc_chunks' in tables:
        columns = {c['name']: c['type'] for c in inspector.get_columns('doc_chunks')}
        
        # Check if embedding column exists and is not vector type
        if 'embedding' in columns:
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'doc_chunks' AND column_name = 'embedding';
                """))
                column_type = result.scalar()
                
                if column_type != 'USER-DEFINED' and 'vector' not in str(column_type).lower():
                    click.echo(f"Found embedding column with incorrect type: {column_type}")
                    table_needs_recreation = True
    
    # Drop and recreate the doc_chunks table if needed
    if table_needs_recreation and 'doc_chunks' in tables:
        click.echo("The doc_chunks table has incorrect column types. It needs to be dropped and recreated.")
        if click.confirm('Do you want to drop and recreate the doc_chunks table? This will delete all existing chunk data.'):
            with engine.connect() as conn:
                click.echo("Dropping doc_chunks table...")
                conn.execute(text("DROP TABLE doc_chunks;"))
                conn.commit()
            # Remove from the list of existing tables so it will be recreated
            tables.remove('doc_chunks')
            click.echo("Table dropped successfully.")
        else:
            click.echo("Cannot continue setup without fixing table structure. Exiting.")
            return
    
    # Create tables that don't exist yet
    click.echo("Creating missing database tables...")
    Base.metadata.create_all(engine, checkfirst=True)
    
    # Create indices for vector columns
    with engine.connect() as conn:
        click.echo("Creating indices for vector columns if needed")
        
        # Check if transformer embedding index exists
        result = conn.execute(text("""
        SELECT indexname FROM pg_indexes 
        WHERE tablename = 'doc_chunks' AND indexdef LIKE '%embedding vector_cosine_ops%';
        """))
        
        if not result.fetchone():
            click.echo("Creating index on transformer embedding column")
            try:
                conn.execute(text("CREATE INDEX ON doc_chunks USING ivfflat (embedding vector_cosine_ops);"))
            except Exception as e:
                click.echo(f"Failed to create index on embedding column: {str(e)}")
        
        # Check if word2vec embedding index exists
        result = conn.execute(text("""
        SELECT indexname FROM pg_indexes 
        WHERE tablename = 'doc_chunks' AND indexdef LIKE '%word2vec_embedding vector_cosine_ops%';
        """))
        
        if not result.fetchone():
            click.echo("Creating index on Word2Vec embedding column")
            try:
                conn.execute(text("CREATE INDEX ON doc_chunks USING ivfflat (word2vec_embedding vector_cosine_ops);"))
            except Exception as e:
                click.echo(f"Failed to create index on word2vec_embedding column: {str(e)}")
        
        # Commit all changes
        conn.commit()
        
        click.echo("Vector column setup complete")
    
    click.echo("Database setup complete")

@cli.command()
def reset_processed_status():
    """Reset the processed_at column to NULL for all documents.
    
    This allows documents to be processed again by the processing scripts,
    without needing to re-download them.
    """
    if click.confirm('Are you sure you want to reset processing status for all documents? This will mark all documents as unprocessed.'):
        session = get_db_session(DATABASE_URL)
        
        try:
            # Query to count documents with processed_at set
            processed_count = session.query(NDCDocumentModel).filter(
                NDCDocumentModel.processed_at.isnot(None)
            ).count()
            
            if processed_count == 0:
                click.echo("No documents found with processed_at set. Nothing to reset.")
                return
            
            # Update all documents to set processed_at to NULL
            updated = session.query(NDCDocumentModel).filter(
                NDCDocumentModel.processed_at.isnot(None)
            ).update({"processed_at": None})
            
            session.commit()
            click.echo(f"Reset processing status for {updated} documents.")
            
            # Option to also clear doc_chunks table
            if click.confirm('Do you also want to clear the doc_chunks table?'):
                engine = create_engine(DATABASE_URL)
                with engine.connect() as conn:
                    # We use truncate instead of drop to maintain table structure
                    conn.execute(text("TRUNCATE TABLE doc_chunks;"))
                    conn.commit()
                click.echo("doc_chunks table has been cleared.")
                
        except Exception as e:
            session.rollback()
            click.echo(f"Error resetting processing status: {str(e)}")
        finally:
            session.close()
    else:
        click.echo("Operation cancelled.")

@cli.command()
@click.argument('country', required=False)
def reset_processed_by_country(country=None):
    """Reset the processed_at column for documents from a specific country.
    
    If no country is provided, it will show a list of countries to choose from.
    """
    session = get_db_session(DATABASE_URL)
    
    try:
        if not country:
            # Get list of countries that have processed documents
            countries = session.query(NDCDocumentModel.country).filter(
                NDCDocumentModel.processed_at.isnot(None)
            ).distinct().order_by(NDCDocumentModel.country).all()
            
            if not countries:
                click.echo("No processed documents found for any country.")
                return
                
            # Display countries for selection
            click.echo("Select a country to reset processing status:")
            for i, (c,) in enumerate(countries, 1):
                click.echo(f"{i}. {c}")
                
            # Get user selection
            selection = click.prompt("Enter country number", type=int, default=1)
            if selection < 1 or selection > len(countries):
                click.echo("Invalid selection.")
                return
                
            country = countries[selection-1][0]
        
        # Confirm with user
        if not click.confirm(f'Reset processing status for documents from {country}?'):
            click.echo("Operation cancelled.")
            return
            
        # Update documents for selected country
        updated = session.query(NDCDocumentModel).filter(
            NDCDocumentModel.country == country,
            NDCDocumentModel.processed_at.isnot(None)
        ).update({"processed_at": None})
        
        session.commit()
        click.echo(f"Reset processing status for {updated} documents from {country}.")
        
        # Option to also remove chunks for this country from doc_chunks
        if updated > 0 and click.confirm('Do you also want to remove chunks for this country from doc_chunks table?'):
            engine = create_engine(DATABASE_URL)
            with engine.connect() as conn:
                result = conn.execute(
                    text("DELETE FROM doc_chunks WHERE metadata->>'country' = :country"),
                    {"country": country}
                )
                conn.commit()
                click.echo(f"Removed {result.rowcount} chunks for {country} from doc_chunks table.")
                
    except Exception as e:
        session.rollback()
        click.echo(f"Error resetting processing status: {str(e)}")
    finally:
        session.close()

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
        total, successful = process_downloads(session, DOWNLOAD_DIR)
        
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
        # Get all PDF files in the directory
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
        
        if not pdf_files:
            click.echo("No PDF files found in the specified directory.")
            return
            
        with click.progressbar(pdf_files, label='Importing PDFs') as files:
            for filename in files:
                doc_id = filename[:-4]  # Remove .pdf extension
                
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
                dst_path = os.path.join(DOWNLOAD_DIR, filename)
                
                # Ensure download directory exists
                os.makedirs(DOWNLOAD_DIR, exist_ok=True)
                
                # Copy file to download directory
                shutil.copy2(src_path, dst_path)
                
                # Update document record
                doc.file_path = dst_path
                doc.file_size = os.path.getsize(dst_path) / (1024 * 1024)  # Convert to MB
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

@cli.command()
@click.option('--country', '-c', help='Filter by country name')
@click.option('--output', '-o', help='Output file path')
@click.option('--limit', '-l', type=int, help='Limit the number of documents to process')
def find_emissions_targets(country, output, limit):
    """Run the emissions_target_search.py script to find emissions targets in documents."""
    script_path = 'scripts/emissions_target_search.py'
    
    args = [sys.executable, script_path]
    if country:
        args.extend(['--country', country])
    if output:
        args.extend(['--output', output])
    if limit:
        args.extend(['--limit', str(limit)])
    
    click.echo(f"Running {script_path}...")
    subprocess.run(args)

@cli.command()
@click.option('--country', '-c', help='Filter by country name')
@click.option('--output', '-o', help='Output file path')
def extract_targets(country, output):
    """Run the extract_target_summary.py script to extract and summarize targets."""
    script_path = 'scripts/extract_target_summary.py'
    
    args = [sys.executable, script_path]
    if country:
        args.extend(['--country', country])
    if output:
        args.extend(['--output', output])
    
    click.echo(f"Running {script_path}...")
    subprocess.run(args)

@cli.command()
@click.option('--reset', '-r', is_flag=True, help='Reset the database before populating')
@click.option('--source', '-s', help='Source file for data import')
def populate_db(reset, source):
    """Run the populate_database.py script to populate the database with data."""
    script_path = 'scripts/populate_database.py'
    
    args = [sys.executable, script_path]
    if reset:
        args.append('--reset')
    if source:
        args.extend(['--source', source])
    
    click.echo(f"Running {script_path}...")
    subprocess.run(args)

@cli.command()
@click.option('--country', '-c', help='Filter by country name')
@click.option('--limit', '-l', type=int, help='Limit the number of documents to process')
@click.option('--reprocess', is_flag=True, help='Reprocess already processed documents')

def process_docs(country, limit, reprocess):
    """Run the process_documents.py script to process downloaded documents."""
    script_path = 'scripts/process_documents.py'
    
    args = [sys.executable, script_path]
    if country:
        args.extend(['--country', country])
    if limit:
        args.extend(['--limit', str(limit)])
    if reprocess:
        args.append('--reprocess')
    
    click.echo(f"Running {script_path}...")
    subprocess.run(args)

@cli.command()
@click.option('--image', '-i', help='Path to test image')
def test_ocr(image):
    """Run the test_tesseract.py script to test OCR functionality."""
    script_path = 'scripts/test_tesseract.py'
    
    args = [sys.executable, script_path]
    if image:
        args.extend(['--image', image])
    
    click.echo(f"Running {script_path}...")
    subprocess.run(args)

if __name__ == '__main__':
    cli()