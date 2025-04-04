"""
Scrapy settings for climate_policy_extractor project.
"""
import os
from dotenv import load_dotenv
from pathlib import Path
from .logging import setup_colored_logging

load_dotenv()

# In settings.py
DATABASE_URL = os.getenv('DATABASE_URL')
# If not set in environment, use a default
if not DATABASE_URL:
    DATABASE_URL = 'postgresql://climate:climate@localhost:5432/climate'

BOT_NAME = "climate_policy_extractor"

SPIDER_MODULES = ["climate_policy_extractor.spiders"]
NEWSPIDER_MODULE = "climate_policy_extractor.spiders"

# Crawl responsibly by identifying yourself on the user-agent
# USER_AGENT = "climate_policy_extractor (+https://lse-dsi.github.io/DS205/)"

ITEM_PIPELINES = {
    'climate_policy_extractor.pipelines.PostgreSQLPipeline': 300,
    # 'climate_policy_extractor.pipelines.DocumentDownloadPipeline': 400,  # Enable document download
    # 'climate_policy_extractor.pipelines.PDFTextExtractionPipeline': 500,  # Enable text extraction
}

AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 3
AUTOTHROTTLE_MAX_DELAY = 10

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Documents processing settings
MAX_CHUNK_SIZE = 512  # Maximum size of text chunks in characters
CHUNK_OVERLAP = 2     # Number of sentences to overlap between chunks

# Set settings whose default value is deprecated to a future-proof value
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
FEED_EXPORT_ENCODING = "utf-8"

# Custom settings for the project
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
DOWNLOAD_DIRECTORY = os.path.join(DATA_DIR, 'pdfs')
PROCESSED_DIRECTORY = os.path.join(DATA_DIR, 'processed')

# Add accepted file types
ACCEPTED_FILE_TYPES = ['pdf', 'docx']

# Logging settings
LOG_FILE = os.path.join(PROJECT_ROOT, 'scrapy.log')
LOG_ENABLED = True
LOG_LEVEL = 'DEBUG'
LOG_STDOUT = False  # Don't log stdout
LOG_STDERR = False  # Don't log stderr
LOG_FILE_APPEND = True  # Start fresh log each time
LOG_FORMAT = '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
LOG_DATEFORMAT = '%Y-%m-%d %H:%M:%S'

# Create necessary directories
os.makedirs(DOWNLOAD_DIRECTORY, exist_ok=True)
os.makedirs(PROCESSED_DIRECTORY, exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DIRECTORY, 'json'), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DIRECTORY, 'chunks'), exist_ok=True)