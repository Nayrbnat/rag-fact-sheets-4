"""
Spider for scraping NDC (Nationally Determined Contributions) documents from the UNFCCC NDC registry.
"""
import scrapy

from tqdm import tqdm
from datetime import datetime
from dateutil.parser import parse

from ..items import NDCDocument

class NDCSpider(scrapy.Spider):
    """
    Spider for scraping NDC (Nationally Determined Contributions) documents.
    
    This spider crawls the UNFCCC NDC registry to find and download
    NDC documents for various countries.
    """
    name = "ndc_spider"
    allowed_domains = ["unfccc.int"]
    start_urls = ["https://unfccc.int/NDCREG"]

    def start_requests(self):
        """
        Start requests method to add more logging.
        """
        print("Starting requests...")
        self.logger.info("Starting requests method called")

        for url in self.start_urls:
            self.logger.info(f"Sending request to {url}")
            yield scrapy.Request(url=url, callback=self.parse)
    
    def parse(self, response):
        """
        Parse the main NDC registry page to find links to country NDC documents.
        
        The UNFCCC NDC Registry page contains a table with countries and their NDC submissions.
        This function extracts the relevant information from this table.
        
        Args:
            response: The HTTP response object
            
        Yields:
            Requests to download PDF documents
        """
        print("Parse method called!")
        self.logger.info(f"Parsing NDC registry page: {response.url}")

        # Find the table and all rows
        rows = response.xpath('//table[contains(@class, "table-hover")]//tr')

        # TODO: Remove max number of files to process all of them!
        MAX_FILES = 5
        for row in tqdm(rows[1:], desc="Processing rows"):  # Skip table header
            # Extract data from columns
            cols = row.xpath('./td')
            country = cols[0].xpath('normalize-space(.)').get()
            langs = cols[1].xpath('normalize-space(.//span)').getall()
            docs = cols[1].xpath('.//a/@href').getall()
            doc_titles = cols[1].xpath('normalize-space(.//a)').getall()
            upload_dates = [date.split("/")[-2] for date in docs]
            backup_date = cols[6].xpath('normalize-space(.)').get()
            
            for lang, doc, date, doc_title in zip(langs, docs, upload_dates, doc_titles):
                # Create item directly instead of using metadata and a separate callback
                item = NDCDocument()
                
                item["country"] = country
                item["url"] = doc
                item["title"] = doc_title
                item["language"] = lang

                item["download_date"] = datetime.now().date()
                try: 
                    item["submission_date"] = parse(date + "/01").date()
                except: 
                    item["submission_date"] = parse(backup_date).date()

                # Note: the remaining fields are populated in the pipeline
                # file_path, file_size
                # Check pipelines.py to see how the fields are populated

                yield item