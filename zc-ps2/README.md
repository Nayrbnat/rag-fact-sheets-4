# Climate Policy Extractor (DS205 - Problem Set 2)

<figure><img src="./figures/DS205_2024_25_icon_200px.png" role="presentation" style="object-fit: cover;width:5em;height:5em;border-radius: 1em;margin:1em 0em;"/></figure>

A tool for extracting climate policy information from National Determined Contributions (NDC) documents.

[(Part of [DS205 - Advanced Data Manipulation](https://lse-dsi.github.io/DS205))]{style="color:grey;font-size:0.9em;display:block;"}

## Overview

This project provides a framework for analyzing climate policy information from NDC documents submitted to the UNFCCC. It is designed as part of the DS205 course at the London School of Economics and Political Science.

The Climate Policy Extractor helps you:

1.  **Collect National Determined Contributions (NDC) documents** from the [UNFCCC registry](https://unfccc.int/NDCREG)
2.  **Extract and process text** from these documents using NLP techniques
3.  **Generate embeddings** for document chunks to enable semantic search
4.  **Build an information retrieval system** to extract specific climate policy information, such as emissions reduction targets

This tool aims to assist analysts in quickly finding and extracting relevant climate policy information from lengthy policy documents, making the assessment of climate commitments more efficient and accurate.

## Project Structure

```         
climate-policy-extractor/
├── climate_policy_extractor/
│   ├── __init__.py
│   ├── items.py                       # Data models for scraped items
│   ├── downloaders.py                 # Custom downloader functions
│   ├── models.py                      # SQLAlchemy models for database
│   ├── settings.py                    # Scrapy settings
│   ├── pipelines.py                   # Processing pipelines
│   ├── spiders/
│   │   ├── __init__.py
│   │   └── ndc_spider.py              # Spider for scraping NDC documents
│   └── utils.py                       # Utility functions
├── notebooks/                         # Jupyter notebooks
├── data/
│   ├── pdfs/                          # Downloaded PDF documents
│   └── processed/                     # Processed document data
├── README.md                          # This file
├── CONTRIBUTING.md                    # Setup and contribution guidelines
├── REPORT.md                          # Technical report template
├── requirements.txt                   # Project dependencies
└── scrapy.cfg                         # Scrapy configuration
```

## Getting Started

For detailed setup instructions, including environment setup, database configuration, and troubleshooting tips, please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## Workflow

The system comprises 4 main components:

1. Scraping and file download
2. Document text extraction and chunking
3. Embedding generation
4. Information extraction

More details on this are provided in the [REPORT.md](REPORT.md) file.

## 1.  **Scraping and File Download**: Collect NDC documents and data from the UNFCCC registry

To download the documents, run the `ndc_spider.py` spider with the following command:

``` bash
python tasks.py crawl       # Update database with document data, including links to PDFs
python tasks.py download    # Download all documents from links stored in database

# OPTIONAL: If you have already downloaded the PDFs and want to import them into the database, run:
python tasks.py import-pdfs <path_to_pdfs> 
```

The spider will save the downloaded PDFs in the `data/pdfs/` directory.

## 2.  Document Text Extraction and Chunking: Extract text from PDFs and chunk it into smaller segments

To extract text and chunks from the downloaded PDFs, run [NB01-pdf-extractor.ipynb](notebooks/NB01-pdf-extractor.ipynb).

## 3.  **Embedding Generation**: Create vector representations of text chunks

To generate embeddings for the extracted text chunks, run [NB02-embedding-generation.ipynb](notebooks/NB02-embedding-generator.ipynb).

This notebook generates embeddings for all the text chunks and populates the `doc_chunks` table with the generated embeddings, as well as a vector similarity index for each chunk.


## 4.  **Information Retrieval**: Extract specific climate policy information with an LLM

To run the pgvector and LLM-based info retrieval, run [NB03-llm-rag.ipynb](notebooks/NB03-llm-rag.ipynb).

ACTION NEEDED: Set the country you wish to retrieve information for in the notebook itself. Currently, it is set to Australia.

Note: The LLM used is LLama-3.1-70B-Instruct, which will require an API key and endpoint. I used Nebius as introduced in the lecture.


## ⚠️ Known Issues

The unstructured library may not be able to parse some pdfs. Running `extract_text_from_pdf()` on all countries in [NB01-pdf-extractor.ipynb](notebooks/NB01-pdf-extractor.ipynb) raised 5 errors of the following format:

``` python
No /Root object! - Is this really a PDF?
...
PDF text extraction failed, skip text extraction...
```

On manual inspection, the countries with this issue are:
- Eswatini (DOCX, not PDF)
- Israel (DOCX, not PDF)
- Bahamas (Unknown error, maybe increase download attempts)
- Mauritius (DOCX, not PDF)
- Iraq (DOCX, not PDF)

## License

This current project is to be kept private and intended for educational purposes only. In the Spring Term 2024/25, we will evaluate contributions and select the best ones to create a public repository. All other participants will have the opportunity to contribute through pull requests (PRs) based on the feedback received on the individual submissions, which may also earn you additional marks.