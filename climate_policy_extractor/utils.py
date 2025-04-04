"""
Utility functions for the climate policy extractor.
"""
import zoneinfo
from datetime import datetime

def now_london_time():
    """Get current time in London timezone."""
    london_tz = zoneinfo.ZoneInfo('Europe/London')
    return datetime.now(london_tz)