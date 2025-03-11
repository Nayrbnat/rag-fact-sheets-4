"""
Scrapy settings for climate_policy_extractor project.
"""
import os
from pathlib import Path

BOT_NAME = "climate_policy_extractor"

SPIDER_MODULES = ["climate_policy_extractor.spiders"]
NEWSPIDER_MODULE = "climate_policy_extractor.spiders"

# Crawl responsibly by identifying yourself on the user-agent
# USER_AGENT = "climate_policy_extractor (+https://lse-dsi.github.io/DS205/)"

ITEM_PIPELINES = {
    'climate_policy_extractor.pipelines.DocumentDownloadPipeline': 100,
    # TODO: You can choose to add another pipeline here to do the 
    #       text extraction and chunking whenever you run the crawler
    # 'climate_policy_extractor.pipelines.PDFTextExtractionPipeline': 200
}

AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 3
AUTOTHROTTLE_MAX_DELAY = 10

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Set settings whose default value is deprecated to a future-proof value
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
FEED_EXPORT_ENCODING = "utf-8"

# Custom settings for the project
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
DOWNLOAD_DIRECTORY = os.path.join(DATA_DIR, 'pdfs')
PROCESSED_DIRECTORY = os.path.join(DATA_DIR, 'processed')
LOG_LEVEL = "DEBUG"

FEEDS = {
    'data/output.jsonl': {
        'format': 'jsonlines',
        'encoding': 'utf8',
        'store_empty': True,
        'overwrite': True,
    },
}

# Files Pipeline settings
FILES_STORE = DOWNLOAD_DIRECTORY
FILES_EXPIRES = 365  # Files will not expire for 1 year
MEDIA_ALLOW_REDIRECTS = True

# Create necessary directories
for directory in [DATA_DIR, DOWNLOAD_DIRECTORY, PROCESSED_DIRECTORY]:
    os.makedirs(directory, exist_ok=True) 