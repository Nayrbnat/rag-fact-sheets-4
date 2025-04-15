"""
Standalone document downloaders for the climate policy extractor.

References:
    - File downloading approach: https://stackoverflow.com/a/39217788
"""
import os
import time
import random
import requests
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import NDCDocumentModel
from .logging import get_logger
from .utils import now_london_time

logger = get_logger(__name__)

def create_session() -> requests.Session:
    """Create a requests session with retries and browser-like headers."""
    session = requests.Session()
    
    # Configure retries
    retries = Retry(
        total=5,  # number of retries
        backoff_factor=1,  # wait 1, 2, 4, 8, 16 seconds between retries
        status_forcelist=[429, 500, 502, 503, 504],  # retry on these status codes
    )
    
    # Add retry adapter to session
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    # Browser-like headers
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/pdf,application/x-pdf,*/*',
        # The 'Accept-Language' header indicates the preferred language for the response.
        # It helps the server understand the language in which the client wants the content.
        # More info: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Accept-Language
        'Accept-Language': 'en-US,en;q=0.5',
        
        # The 'Connection' header controls whether the network connection stays open after the current transaction.
        # 'keep-alive' allows the connection to remain open for further requests, improving performance.
        # More info: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Connection
        'Connection': 'keep-alive',
        
        # The 'Upgrade-Insecure-Requests' header is a signal to the server that the client prefers an encrypted and authenticated response.
        # It indicates that the client would like to upgrade to HTTPS if possible.
        # More info: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Upgrade-Insecure-Requests
        'Upgrade-Insecure-Requests': '1',
        
        # The 'Sec-Fetch-Dest' header indicates the request's destination, which helps the server understand the context of the request.
        # In this case, 'document' indicates that the request is for a document.
        # More info: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Sec-Fetch-Dest
        'Sec-Fetch-Dest': 'document',
        
        # The 'Sec-Fetch-Mode' header indicates the mode of the request, which can affect how the server processes it.
        # 'navigate' indicates that the request is for navigation, such as loading a new page.
        # More info: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Sec-Fetch-Mode
        'Sec-Fetch-Mode': 'navigate',
        
        # The 'Sec-Fetch-Site' header indicates the relationship between the origin of the request and the origin of the document.
        # 'none' indicates that the request is not initiated by a top-level navigation.
        # More info: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Sec-Fetch-Site
        'Sec-Fetch-Site': 'none',
        
        # The 'Sec-Fetch-User' header indicates whether the request was initiated by a user action.
        # '?1' indicates that the request was made as a result of a user action.
        # More info: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Sec-Fetch-User
        'Sec-Fetch-User': '?1',
        
        # The 'DNT' (Do Not Track) header indicates the user's preference regarding tracking.
        # '1' means the user does not want to be tracked.
        # More info: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/DNT
        'DNT': '1',
    })
    
    return session

def download_file(session: requests.Session, url: str, save_path: str, 
                 pbar: Optional[tqdm] = None, retry_count: int = 0) -> Tuple[bool, Optional[float]]:
    """
    Download a file from a URL and save it to the specified path.
    
    Args:
        session: Requests session to use
        url: The URL to download from
        save_path: The full path where to save the file
        pbar: Optional progress bar to update during download
        retry_count: Current retry attempt (for internal use)
        
    Returns:
        tuple: (success: bool, file_size: Optional[float])
    """
    try:
        # Add a random delay between 2-5 seconds
        # We don't want to DDOS the server!
        time.sleep(random.uniform(2, 5))
        
        # First make a HEAD request to check content type
        head_response = session.head(url, allow_redirects=True)
        content_type = head_response.headers.get('Content-Type', '').lower()
        
        if 'pdf' not in content_type and retry_count < 3:
            logger.warning(f"Unexpected content type {content_type}, retrying...")
            return download_file(session, url, save_path, pbar, retry_count + 1)
        
        # Now get the actual file
        with session.get(url, stream=True) as r:
            r.raise_for_status()
            
            # Update progress bar description if provided
            if pbar:
                pbar.set_description(f"Downloading {Path(save_path).name}")
            
            with open(save_path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
            
            # Get file size in MB
            size_mb = Path(save_path).stat().st_size / (1024 * 1024)
            
            # Verify we got a PDF, MAY WANT TO CHANGE LATER TO ACCOMMODATE OTHER FILE TYPES
            with open(save_path, 'rb') as f:
                header = f.read(4)
                if not header.startswith(b'%PDF'):
                    if retry_count < 3:
                        logger.warning(f"Invalid PDF file, retrying... (attempt {retry_count + 1})")
                        os.remove(save_path)
                        # Wait longer before retry
                        time.sleep(random.uniform(5, 10))
                        return download_file(session, url, save_path, pbar, retry_count + 1)
                    else:
                        logger.error(f"Downloaded file is not a valid PDF after {retry_count} retries")
                        os.remove(save_path)
                        return False, None
                
            return True, size_mb
            
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        if retry_count < 3:
            logger.warning(f"Retrying download... (attempt {retry_count + 1})")
            time.sleep(random.uniform(5, 10))
            return download_file(session, url, save_path, pbar, retry_count + 1)
        return False, None

def process_downloads(session: Session, download_dir: str, limit: Optional[int] = None) -> Tuple[int, int]:
    """
    Process all undownloaded documents in the database.
    Downloads files and updates the database with download status, file path and size.
    
    Args:
        session: SQLAlchemy session
        download_dir: Directory to save downloaded files
        limit: Optional limit on number of documents to process
        
    Returns:
        tuple: (num_processed: int, num_successful: int)
    """
    # Ensure download directory exists
    os.makedirs(download_dir, exist_ok=True)
    
    # Query for undownloaded documents that haven't exceeded max attempts
    query = select(NDCDocumentModel).where(
        NDCDocumentModel.downloaded_at.is_(None),
        NDCDocumentModel.download_attempts < 3  # Max 3 attempts
    )
    
    undownloaded = session.execute(query).scalars().all()   # SQLAlchemy get all undownloaded documents
    if limit:
        undownloaded = undownloaded[:limit]
    
    successful = 0
    
    logger.info(f"Found {len(undownloaded)} documents to download")
    
    # Create a requests session for all downloads
    http_session = create_session()
    
    # Create progress bar for overall progress
    with tqdm(total=len(undownloaded), desc="Overall progress") as pbar:
        for doc in undownloaded:
            # Generate save path using doc_id
            save_path = os.path.join(download_dir, f"{doc.doc_id}.pdf")
            
            # Update attempt tracking
            doc.last_download_attempt = now_london_time()
            doc.download_attempts += 1
            doc.download_error = None  # Clear previous error
            
            logger.debug(f"Downloading {doc.url} to {save_path} (attempt {doc.download_attempts})")
            success, file_size = download_file(http_session, doc.url, save_path, pbar)
            
            if success:
                # Update document record
                doc.file_path = save_path
                doc.file_size = file_size
                doc.downloaded_at = now_london_time()
                successful += 1
                logger.info(f"Successfully downloaded {doc.doc_id} ({file_size:.2f}MB)")
            else:
                doc.download_error = f"Failed to download after {doc.download_attempts} attempts"
                logger.warning(f"Failed to download {doc.doc_id} (attempt {doc.download_attempts})")
            
            session.commit()
            pbar.update(1)
            
            # Add a delay between documents
            time.sleep(random.uniform(3, 7))
    
    return len(undownloaded), successful