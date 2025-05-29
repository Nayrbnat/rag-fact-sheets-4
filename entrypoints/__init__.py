"""
Entrypoints module - exposes all run_script functions for API consumption.

This module provides clean imports for all entrypoint functionality,
making it easy for API routes to access individual pipeline stages.
"""

import importlib

# Import modules (required for numeric prefixes)
scrape_module = importlib.import_module('entrypoints.1_scrape')
chunk_module = importlib.import_module('entrypoints.2_chunk')
embed_module = importlib.import_module('entrypoints.3_embed')
retrieve_module = importlib.import_module('entrypoints.4_retrieve')
llm_module = importlib.import_module('entrypoints.5_llm_response')
output_module = importlib.import_module('entrypoints.6_output')
send_email_module = importlib.import_module('entrypoints.7_send_email')

# Clean wrapper functions
def run_scrape():
    """Run the scraping/detection process."""
    return scrape_module.main()

async def run_chunk(force_reprocess: bool = False):
    """Run the document chunking pipeline."""
    return await chunk_module.run_script(force_reprocess=force_reprocess)

async def run_embed(force_reembed: bool = False):
    """Run the embedding generation pipeline."""
    return await embed_module.run_script(force_reembed=force_reembed)

def run_retrieve(prompt: str = None, question_number: int = None, country: str = None, use_hop_retrieval: bool = False):
    """Run the retrieval process to find relevant chunks."""
    return retrieve_module.run_script(question_number=question_number, country=country, use_hop_retrieval=use_hop_retrieval)

def run_llm_response():
    """Generate LLM response based on chunks and prompt."""
    return llm_module.main()

def run_output():
    """Run the output processing."""
    return output_module.main()

def run_send_email():
    """Run the email sending process."""
    return send_email_module.main()

# Expose all functions
__all__ = [
    'run_scrape',
    'run_chunk',
    'run_embed', 
    'run_retrieve',
    'run_llm_response',
    'run_output',
    'run_send_email'
]