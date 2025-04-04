"""
Item pipelines for the climate policy extractor.
"""
import os
from datetime import datetime, date
from dotenv import load_dotenv
from itemadapter import ItemAdapter

from scrapy.exceptions import DropItem

from .models import NDCDocumentModel, init_db, get_db_session
from .utils import now_london_time

def generate_doc_id(item):
    """Generate a document ID from item metadata."""
    country = item.get('country', 'unknown').lower().replace(" ", "_")
    lang = item.get('language', 'unknown').lower().replace(" ", "_")
    try:
        # Ensure we're using just the date part for the ID
        submission_date = item.get('submission_date')
        if isinstance(submission_date, datetime):
            date_str = submission_date.date().strftime('%Y%m%d')
        elif isinstance(submission_date, date):
            date_str = submission_date.strftime('%Y%m%d')
        else:
            date_str = 'unknown_date'
    except:
        date_str = 'unknown_date'
    
    return f"{country}_{lang}_{date_str}"

class PostgreSQLPipeline:
    """Pipeline for storing NDC documents in PostgreSQL."""

    def __init__(self, db_url=None):
        # Load environment variables
        load_dotenv()
        self.db_url = db_url or os.getenv('DATABASE_URL')
        if not self.db_url:
            raise ValueError("DATABASE_URL not found in environment variables")

    @classmethod
    def from_crawler(cls, crawler):
        return cls()

    def open_spider(self, spider):
        """Initialize database connection when spider opens."""
        self.logger = spider.logger
        init_db(self.db_url)  # Create tables if they don't exist
        self.session = get_db_session(self.db_url)

    def close_spider(self, spider):
        """Close database connection when spider closes."""
        self.session.close()

    def process_item(self, item, spider):
        """Process scraped item and store in PostgreSQL."""
        adapter = ItemAdapter(item)
        
        # Convert submission_date to date if it's a datetime
        if 'submission_date' in item:
            submission_date = item['submission_date']
            if isinstance(submission_date, datetime):
                item['submission_date'] = submission_date.date()
        
        self.logger.debug(f"Processing item: {item}")

        # Generate doc_id from metadata (same as future file name)
        doc_id = generate_doc_id(item)
        self.logger.debug(f"Generated doc_id: {doc_id}")

        # Create or update document record
        self.logger.debug(f"Querying database for document with doc_id: {doc_id}")
        doc = self.session.query(NDCDocumentModel).filter_by(doc_id=doc_id).first()

        if doc:
            log_msg = (
                f"Document found in database: {doc}. "
                "Checking if it has already been fully processed..."
            )
            self.logger.debug(log_msg)

            retrieved_doc_as_dict = adapter.asdict()
            
            # Check if any data has changed, excluding timestamps we don't want to modify
            has_changes = False
            changes = []
            
            for key, value in retrieved_doc_as_dict.items():
                # Skip downloaded_at and processed_at to preserve their values
                if key in ['downloaded_at', 'processed_at', 'scraped_at']:
                    continue

                if hasattr(doc, key):
                    current_value = getattr(doc, key)
                    
                    if current_value != value:
                        changes.append(f"{key}: {current_value} -> {value}")
                        has_changes = True
                        setattr(doc, key, value)
            
            if has_changes:
                # Always update scraped_at when we see the document
                doc.scraped_at = now_london_time()
                self.logger.info(f"Updating document {doc_id} with changes: {', '.join(changes)}")
            else:
                raise DropItem(f"No changes detected for document {doc_id}, skipping update")
        else:
            log_msg = (
                f"Document not found in database. "
                "Inserting new document."
            )
            self.logger.debug(log_msg)

            doc = NDCDocumentModel(
                doc_id=doc_id,
                country=adapter.get('country'),
                title=adapter.get('title'),
                url=adapter.get('url'),
                language=adapter.get('language'),
                submission_date=adapter.get('submission_date'),
                file_path=None,       # Will be set by download pipeline (outside scrapy)
                file_size=None,       # Will be set by download pipeline (outside scrapy)
                extracted_text=None,  # Will be set by processing pipeline (outside scrapy)
                chunks=None,          # Will be set by processing pipeline (outside scrapy)
                downloaded_at=None,   # Will be set by download pipeline (outside scrapy)
                processed_at=None     # Will be set by processing pipeline (outside scrapy)
            )
            self.logger.debug(f"Adding document to PostgreSQL: {doc}")
            try:
                self.session.add(doc)
            except Exception as e:
                single_line_msg = str(e).replace("\n", " ")
                self.logger.error(f"Error adding document to PostgreSQL: {single_line_msg}")
                raise DropItem(f"Failed to add document to PostgreSQL: {single_line_msg}")
        
        try:
            self.session.commit()
            # Add doc_id back to the item for downstream processing
            item['doc_id'] = doc_id
            self.logger.debug(f"Stored item in PostgreSQL: {item}")
        except Exception as e:
            self.session.rollback()
            self.logger.error(f"Error storing item in PostgreSQL: {e}")
            raise DropItem(f"Failed to store item in PostgreSQL: {e}")
        
        return item