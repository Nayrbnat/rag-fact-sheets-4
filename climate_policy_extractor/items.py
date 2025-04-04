"""
Define data models for scraped items.
"""
import scrapy


class NDCDocument(scrapy.Item):
    """
    Data model for NDC (Nationally Determined Contributions) documents.
    """

    doc_id = scrapy.Field()

    # Fields that are populated directly by the spider
    country = scrapy.Field()
    title = scrapy.Field()
    url = scrapy.Field()
    language = scrapy.Field()
    submission_date = scrapy.Field()
    download_date = scrapy.Field()

    # Fields that are populated after extraction
    # Under the pipelines.py file
    file_path = scrapy.Field()
    file_size = scrapy.Field()

    # TODO: This is up to you to decide if you want to use as part of your pipeline
    #       or if you just want to use a separate notebook/script for it
    extracted_text = scrapy.Field()
    chunks = scrapy.Field()

    def __repr__(self):
        obj = (
            f"NDCDocument(doc_id={self.get('doc_id')}, "
            f"country={self.get('country')}, "
            f"title={self.get('title')}, "
            f"url={self.get('url')}, "
            f"language={self.get('language')}, "
        )
        return obj
