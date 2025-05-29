# Embed Module Documentation

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Core Functions](#core-functions)
  - [run_script](#run_script)
  - [chunk_file_many](#chunk_file_many)
  - [chunk_file_one](#chunk_file_one)
  - [get_file_paths](#get_file_paths)
- [Text Extraction Pipeline](#text-extraction-pipeline)
- [Chunking Process](#chunking-process)
- [Cleaning Strategy](#cleaning-strategy)
- [Database Operations](#database-operations)
- [Error Handling](#error-handling)
- [Usage](#usage)
- [Performance Considerations](#performance-considerations)
- [Contributing](#contributing)
- [Appendix](#appendix)

---

## Overview

The `2_chunk.py` module is responsible for transforming raw NDC PDF documents into structured, semantically coherent text chunks. These chunks are the foundational units used in our RAG (Retrieval-Augmented Generation) pipeline. The script extracts, cleans, segments, and uploads chunks into the database for downstream vector embedding.

**Entry Point:**
```python
asyncio.run(run_script(force_reprocess=args.force))
```

**System Flow:**
```text
PDF File
  ↓
extract_text_from_pdf()
  ↓
chunk_document_by_sentences()
  ↓
cleaning_function()
  ↓
Database: doc_chunks table
```

...

**End of Documentation**
