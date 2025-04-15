"""
Scrapy settings for climate_policy_extractor project.
"""
import os
from dotenv import load_dotenv
from pathlib import Path
from .logging import setup_colored_logging

load_dotenv()

BOT_NAME = "climate_policy_extractor"

SPIDER_MODULES = ["climate_policy_extractor.spiders"]
NEWSPIDER_MODULE = "climate_policy_extractor.spiders"

# Crawl responsibly by identifying yourself on the user-agent
# USER_AGENT = "climate_policy_extractor (+https://lse-dsi.github.io/DS205/)"

ITEM_PIPELINES = {
    'climate_policy_extractor.pipelines.PostgreSQLPipeline': 300,
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
for directory in [DATA_DIR, DOWNLOAD_DIRECTORY, PROCESSED_DIRECTORY]:
    os.makedirs(directory, exist_ok=True)

# Database settings are loaded from .env file
# See CONTRIBUTING.md for setup instructions