# Contributing to Climate Policy Extractor

This guide will help you set up your development environment to work on the Climate Policy Extractor project.

## 1. Install Python dependencies

First, clone the repository and navigate to the project directory:

```bash
# Calling it ds205-ps2 (or the like) will make it easier for you
git clone your-unique-repo-url ds205-ps2
cd ds205-ps2
```

Create a virtual environment using Python's built-in `venv` module:

```bash
# Create a virtual environment. 
# This time, let's just simply call it venv to avoid confusion
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

Verify that you're using the Python interpreter from the virtual environment:

```bash
which python  # On macOS/Linux
where python  # On Windows
```

This should output a path that includes your virtual environment directory. **If it doesn't, contact instructors via Slack. Something is wrong.**

Install the required dependencies:

```bash
pip install -r requirements.txt
```

**Importantly**, install the `NLTK` downloads:

```bash
python -m nltk.downloader stopwords punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng
```

## 2. Running the Spider

To run the NDC spider and collect documents:

```bash
# Make sure you're in the project root directory
cd climate_policy_extractor
# Run the spider
scrapy crawl ndc_spider
```

## 3. Set up the Database

The project uses PostgreSQL with the pgvector extension for vector search capabilities. You have two options for setting up the database:

#### 3.1 Install Docker

If you don't have Docker installed, you can download it from the [Docker website](https://www.docker.com/products/docker-desktop/).


#### 3.2 Run PostgreSQL with pgvector

Start a PostgreSQL container with the pgvector extension:

```bash
docker run -itd --name climate-policy-postgres   -v postgres_climate_policy_data:/var/lib/postgresql/data   -p 5432:5432   -e POSTGRES_PASSWORD=climate   -e POSTGRES_USER=climate   -e POSTGRES_DB=climate   pgvector/pgvector:0.7.1-pg16
```

Verify that the container is running:

```bash
docker ps -a
```


The STATUS of the container should be "Up".

#### 3.2.3 Create database and user

```bash
# Connect to PostgreSQL
sudo -u postgres psql

# Create user and database
CREATE USER climate WITH PASSWORD 'climate';
CREATE DATABASE climate OWNER climate;

# Connect to the climate database
\c climate

# Enable pgvector extension
CREATE EXTENSION vector;

# Exit PostgreSQL
\q
```

## 4. Create a `.env` file

Create a `.env` file in the project root directory:

```bash
cp .env.sample .env
```

Edit the `.env` file to include your database connection details:

```
# Database settings
DB_HOST=localhost
DB_PORT=5432
DB_NAME=climate
DB_USER=climate
DB_PASSWORD=climate

# Embedding model settings
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
```

## 5. Working with the Database

After opening docker desktop, run the following command to start the PostgreSQL container:

```bash
docker start climate-policy-postgres
```
To initiate docker you have to run the Docker Desktop application.

### 5.1 Viewing PostgreSQL tables

You can connect to the database to view tables and run queries:

#### Using Docker:

```bash
docker exec -it climate-policy-postgres psql -U climate -d climate
```

### 5.2 Database Schema

The project uses the following database schema:

#### Table `documents`

| Column | Type | Description |
|--------|------|-------------|
| doc_id | text | Primary key |
| country | text | Country name |
| title | text | Document title |
| url | text | Source URL |
| submission_date | timestamp | Date of submission |
| file_path | text | Path to the local file |

#### Table `chunks`

| Column | Type | Description |
|--------|------|-------------|
| chunk_id | serial | Primary key |
| doc_id | text | Foreign key to documents |
| text | text | Chunk text content |
| line_number | integer | Line number representing this chunk |
| embedding | vector(768) | Text embedding vector |


## 7. Troubleshooting

### Docker Issues

If you encounter issues with Docker:

1. Make sure Docker is running
2. Try restarting the container:
   ```bash
   docker restart climate-policy-postgres
   ```
3. Check container logs:
   ```bash
   docker logs climate-policy-postgres
   ```

### Database Connection Issues

If you have trouble connecting to the database:

1. Verify your `.env` file has the correct connection details
2. Check that the PostgreSQL service is running
3. Ensure the pgvector extension is installed correctly

### PDF Extraction Issues

If you encounter issues with PDF extraction:

1. Make sure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```
2. For OCR functionality, ensure Tesseract is installed:
   - macOS: `brew install tesseract`
   - Ubuntu/Debian: `sudo apt install tesseract-ocr`
   - Windows: Download from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki) 