# Climate Policy Extractor (DS205 - Problem Set 2)

<figure>
    <img src="./figures/DS205_2024_25_icon_200px.png" role="presentation" style="object-fit: cover;width:5em;height:5em;border-radius: 1em;margin:1em 0em;">
</figure>

A tool for extracting climate policy information from National Determined Contributions (NDC) documents.

<span style="color:grey;font-size:0.9em;display:block;">(Part of [DS205 - Advanced Data Manipulation](https://lse-dsi.github.io/DS205))</span>


## Overview

This project provides a framework for analyzing climate policy information from NDC documents submitted to the UNFCCC. It is designed as part of the DS205 course at the London School of Economics and Political Science.

The Climate Policy Extractor helps you:

1. **Collect National Determined Contributions (NDC) documents** from the [UNFCCC registry](https://unfccc.int/NDCREG)
2. **Extract and process text** from these documents using NLP techniques
3. **Generate embeddings** for document chunks to enable semantic search
4. **Build an information retrieval system** to extract specific climate policy information, such as emissions reduction targets

This tool aims to assist analysts in quickly finding and extracting relevant climate policy information from lengthy policy documents, making the assessment of climate commitments more efficient and accurate.

## Project Structure

```
climate-policy-extractor/
├── climate_policy_extractor/
│   ├── __init__.py
│   ├── .env                          # Environment variables configuration
│   ├── downloaders.py                # Document download functionality
│   ├── items.py                      # Data models for scraped items
│   ├── logging.py                    # Custom logging configuration
│   ├── models.py                     # SQLAlchemy models for database
│   ├── pipelines.py                  # Processing pipelines
│   ├── settings.py                   # Scrapy settings with DB configuration
│   ├── utils.py                      # Utility functions
│   └── spiders/
│       ├── __init__.py
│       └── ndc_spider.py             # Spider for scraping NDC documents
├── notebooks/                        # Jupyter notebooks for exploration and analysis
│   ├── NB01-pdf-extractor.ipynb      # PDF extraction and processing
│   ├── NB02-embedding-comparison.ipynb # Comparing embedding methods
│   ├── NB03-information-retrieval.ipynb # Semantic search implementation
│   ├── NB04-evaluation.ipynb         # System evaluation and results
│   └── utils.py                      # Notebook-specific utilities
├── scripts/
│   ├── process_documents.py          # Extract and process text from PDFs
│   ├── populate_database.py          # Populate database with processed data
│   ├── emissions_target_search.py    # Search for emissions targets
│   ├── extract_target_summary.py     # Extract summary of targets
│   └── test_tesseract.py             # Testing OCR capabilities
├── data/                             # [gitignored] Data storage directory
│   ├── pdfs/                         # Downloaded PDF documents
│   └── processed/                    # Processed document data
│       ├── json/                     # Full text JSON from processing
│       ├── chunks/                   # JSON files with document chunks
│       └── results/                  # Results from extraction and analysis
├── REPORT.ipynb                      # Detailed project report with code examples
├── README.md                         # Project overview and usage instructions
├── CONTRIBUTING.md                   # Setup and contribution guidelines
├── .env.sample                       # Example environment variables template
├── requirements.txt                  # Project dependencies
├── tasks.py                          # Command-line tools for project management
├── .gitignore                        # Git ignore patterns for project files
└── scrapy.cfg                        # Scrapy configurationn
```

## Project Pipeline Visualisation

![Climate Policy Extractor Workflow]
<figure>
    <img src="./figures/DS205 Problem Set 2 Workflow.png" role="presentation" style="object-fit: cover;width:5em;height:5em;border-radius: 1em;margin:1em 0em;">
</figure>


*Overview of the complete document processing and information extraction pipeline*

## Getting Started

For detailed setup instructions, including environment setup, database configuration, and troubleshooting tips, please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## Workflow

The project follows this detailed workflow:deac

1. **Setup and Configuration**:
   - Create environment variables using `.env` file (copy from `.env.sample`)
   - Make sure to update the .env file with your API key from the bottom of this page (refer to the API Key section)
   - Refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for detailed setup instructions
   - In the event that you face errors with setting up `venv` , just default to the `venv_working` virtual environment that I have uploaded. Make sure to be on Python 3.10.11 to guarantee that this works
   - Initialize PostgreSQL database using `python tasks.py init-db`
   - Database schema is defined in `models.py` using SQLAlchemy ORM


2. **Data Collection**:
   - Scrape NDC documents from the UNFCCC registry
   - Downloaded PDFs are stored in `data/pdfs/` directory

3. **Document Processing** (`process_documents.py`):
   - Extract text and metadata from PDFs
   - Process documents into manageable chunks
   - Generate JSON outputs in `data/chunks/` and `data/json/` directories
   - OCR support via Tesseract for image-based PDFs

4. **Database Population** (`populate_database.py`):
   - Load processed document chunks into PostgreSQL database
   - Update database columns with extracted metadata
   - Create relationships between documents and chunks

5. **Text Analysis and Embedding Generation**:
   - Analyze document structure to identify key sections
   - Create vector representations of text chunks using pre-trained models
   - Store embeddings in the database for efficient retrieval

6. **Information Retrieval**:
   - Use `emissions_target_search.py` to locate climate commitments
   - Extract specific climate policy information like emissions targets
   - Generate summaries using `extract_target_summary.py`

7. **Evaluation and Reporting**:
   - Assess extraction accuracy and system effectiveness
   - Generate reports on climate commitments by country

The included notebooks guide you through each step of this process with examples and visualizations.


## Using the tasks.py file

### Managing the POSTGRESQL database
```bash
python tasks.py manage-db --operation <operation_name>
```
Available Operations
The manage_db function supports the following operations:
- init: Initialize the database by creating tables if they don't exist
- recreate: Drop all tables and recreate them (destructive)
- drop: Drop all tables (destructive)
- drop_chunks: Drop only the doc_chunks table (destructive)
- update_chunks: Update doc_chunks table to match current model definition (destructive)

### Step by step process on how to use this repository



# 0. Test Tesseract OCR

- Run this code to check if you have tesseract properly installed.
```bash
python tasks.py test-ocr
```

# 1. Initialize database
```bash
python tasks.py manage-db --operation init
```
- Make sure you use setup-db to initialize the Database with PGvector

# 1a Scrap data from NDC
```bash
python tasks.py crawl
```

# 1b Download data from NDC
```bash
python tasks.py download
```

# 2. Process documents
```bash
python tasks.py process-docs
```

# 3. Populate database with embeddings
```bash
python tasks.py populate-db
```

# 4. Search for emissions targets
```bash
python tasks.py find-emissions-targets
```

# 5. Extract and summarize targets
```bash
python tasks.py extract-targets
```

## API Key (Very Important for NB04)

- Under NB04, you will see that there are 2 options to be used for the LLM. Either you uncomment the code and download a local LLM or (if your computer has no CUDA support like me) you can use the API to access open sourced models. I will write the API key here and you have to edit the .env.sample to include the following API key
- API KEY: "4ZYotjAApOrVK1MEjunlagS1pr8AYpkO"
- Add this to the .env file under API_KEY

## Notebooks

- I do not suggest running all the scripts and then looking through the notebooks. The notebooks are meant to tell a story for the project. I would recommand adopting the following structure:

1. Run the 1a,1b and 2 scripts to download and process the data. Now read NB01 to understand what this does.

2. Run 3 (`tasks.py populate-db`) to populate the database and read NB02 to understand what this does.

3. Run 4 (`find-emissions-targets`) to do a similarity search and read NB03 to understand what this does. If you DON'T run this script first, you will get an error for NB03!

4. Run 5 (`extract-targets`) to extract the emissions targets and read NB04 to understand what this does. If you DON'T run this script first, you will get an error for NB04!

## License

This current project is to be kept private and intended for educational purposes only. In the Spring Term 2024/25, we will evaluate contributions and select the best ones to create a public repository. All other participants will have the opportunity to contribute through pull requests (PRs) based on the feedback received on the individual submissions, which may also earn you additional marks.