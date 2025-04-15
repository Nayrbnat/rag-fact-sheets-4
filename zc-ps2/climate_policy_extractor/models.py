"""
SQLAlchemy models for the climate policy extractor.
"""
import zoneinfo
from datetime import datetime, date

from sqlalchemy import create_engine, Column, String, Integer, DateTime, Float, Text, Date, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

from .logging import get_logger
from .utils import now_london_time

Base = declarative_base()
logger = get_logger(__name__)

class NDCDocumentModel(Base):
    """SQLAlchemy model for NDC documents."""
    __tablename__ = 'documents'

    doc_id = Column(String, primary_key=True)

    # Metadata fields
    scraped_at = Column(DateTime, default=now_london_time)
    downloaded_at = Column(DateTime, nullable=True)
    processed_at = Column(DateTime, nullable=True)
    
    # Download attempt tracking
    last_download_attempt = Column(DateTime, nullable=True)
    download_error = Column(String, nullable=True)  # Store error message if download failed
    download_attempts = Column(Integer, default=0)  # Count number of attempts

    country = Column(String, nullable=False)
    title = Column(String)
    url = Column(String, unique=True, nullable=False)
    language = Column(String)
    submission_date = Column(Date)
    file_path = Column(String)
    file_size = Column(Float)
    # extracted_text = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), default=now_london_time)
    updated_at = Column(DateTime(timezone=True), 
                        default=now_london_time,
                        onupdate=now_london_time)
    
    # Relationship to chunks (will be populated by standalone processing)
    doc_chunks = relationship("DocChunk", back_populates="document")
                        
    def __repr__(self):
        obj = (
            f"NDCDocumentModel(doc_id={self.doc_id}, "
            f"country={self.country}, "
            f"title={self.title}, "
            f"url={self.url}, "
            f"language={self.language}, "
            f"submission_date={self.submission_date}, "
            f"scraped_at={self.scraped_at})"
        )
        return obj

class DocChunk(Base):
    """Model for document chunks created during processing."""
    __tablename__ = 'doc_chunks'

    id = Column(String, primary_key=True)  # UUID or similar
    doc_id = Column(String, ForeignKey('documents.doc_id'), nullable=False)
    
    # Chunk content and metadata
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)  # Order in document
    embedding = Column('embedding', ARRAY(Float))  # Store embeddings as an array of floats, for pgvector
    page_number = Column(Integer, nullable=False)  # Page number in the original document
    # Optional metadata about the chunk
    chunk_metadata = Column(JSONB, nullable=True)  # For flexible metadata storage
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=now_london_time)
    updated_at = Column(DateTime(timezone=True), 
                       default=now_london_time,
                       onupdate=now_london_time)
    
    # Relationship back to document
    document = relationship("NDCDocumentModel", back_populates="doc_chunks")

    def __repr__(self):
        return f"DocChunk(id={self.id}, doc_id={self.doc_id}, chunk_index={self.chunk_index})"


def get_db_session(database_url):
    """Create database session."""
    logger.debug(f"Creating database session for {database_url}")
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    return Session()

def init_db(database_url):
    """Initialize database tables."""
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)

