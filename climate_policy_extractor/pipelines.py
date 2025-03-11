"""
Item pipelines for the climate policy extractor.
"""
import os
import logging

from scrapy.pipelines.files import FilesPipeline
from scrapy.exceptions import DropItem
from scrapy import Request

# # from .logging import setup_colored_logging

logger = logging.getLogger(__name__)

class DocumentDownloadPipeline(FilesPipeline):
    """
    Pipeline for downloading NDC documents.
    
    This pipeline:
    1. Downloads PDF documents from URLs
    2. Stores them with appropriate filenames
    3. Updates the item with file paths
    """
    
    def __init__(self, store_uri, download_func=None, settings=None):
        super().__init__(store_uri, download_func, settings)
    
    def get_media_requests(self, item, info):
        """
        Generate requests for downloading files.
        
        Args:
            item: The item containing file URLs
            info: The media pipeline info
            
        Yields:
            Requests for downloading files
        """
        if 'url' in item:
            logger.info(f"Requesting download for: {item['url']}")
            yield Request(
                item['url'],
                meta={'item': item}  # Pass the item in request meta for use in file_path
            )
    
    def file_path(self, request, response=None, info=None, *, item=None):
        """
        Generate file path for storing the document.
        
        This is called after the file has been downloaded and 
        is how the Scrapy documentation recommends doing this.
        
        Args:
            request: The download request
            response: The download response
            info: The media pipeline info
            item: The item being processed
            
        Returns:
            Path where the file should be stored
        """
        # Use item from request.meta if not provided directly
        if item is None:
            item = request.meta.get('item')
            if item is None:
                raise DropItem(f"No item provided in request meta for {request.url}")
        
        # Use item['country'] and item['submission_date'] to compose the filename
        country = item.get('country', 'unknown').lower().replace(" ", "_")
        lang = item.get('language', 'unknown').lower().replace(" ", "_")
        
        try:
            date_str = item.get('submission_date').strftime('%Y%m%d')
        except:
            date_str = 'unknown_date'
        
        # Format the filename to include country and submission date
        file_name = f"{country}_{lang}_{date_str}.pdf"
        
        logger.info(f"Saving file as: {file_name}")
        return file_name
    
    def item_completed(self, results, item, info):
        """
        Called when all file downloads for an item have completed.
        
        Args:
            results: List of (success, file_info_or_error) tuples
            item: The item being processed
            info: The media pipeline info
            
        Returns:
            The updated item
        """
        # Filter out failed downloads
        file_paths = [x['path'] for ok, x in results if ok]
        
        if not file_paths:
            logger.error(f"Failed to download file for {item.get('country', 'unknown')}")
            raise DropItem(f"Failed to download file for {item.get('country', 'unknown')}")
        
        # Update the item with the file path
        item['file_path'] = os.path.join(info.spider.settings.get('FILES_STORE', ''), file_paths[0])
        
        # Get file size
        try:
            item['file_size'] = os.path.getsize(item['file_path'])
        except:
            item['file_size'] = 0
            
        logger.info(f"Successfully downloaded file to: {item['file_path']}")
        return item