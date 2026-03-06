"""
ingest.py
---------
First step of the RAG pipeline.

This file:
1. Reads the EU AI Act PDF from the data/ folder
2. Cleans and chunks the text into smaller pieces
3. Sends each chunk to AzureOpenAI to generate embeddings
4. Stores the chunks + embeddings in PostgreSQL (with pgvector)

Run this script ONCE to populate your database.
After that, use query.py to ask questions.
"""

import os
import time
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
import pdfplumber
from openai import AzureOpenAI
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.chunking import chunk_recursively

# ─── 1. Load environment variables from .env ───────────────────────────────
load_dotenv(".env")

AZURE_ENDPOINT = os.environ["AZURE_ENDPOINT"]
AZURE_API_KEY = os.environ["AZURE_API_KEY"]
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-01")
DEPLOY_LARGE = os.environ["DEPLOY_LARGE"]  # text-embedding-3-large → 3072 dimensions

PGUSER = os.environ["PGUSER"]
PGPASSWORD = os.environ["PGPASSWORD"]
PGHOST = os.environ["PGHOST"]
PGPORT = os.getenv("PGPORT", "5433")
PGDATABASE = os.environ["PGDATABASE"]

PDF_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "euaiact.pdf")
SOURCE_NAME = "euaiact.pdf"

# How many tokens per chunk. 500 is a good balance for GPT-4.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# How many chunks to send to the DB at once (batching is more efficient)
DB_BATCH_SIZE = 50

# Azure OpenAI rate limits mean we need a small pause between embedding calls
EMBED_BATCH_SIZE = 16  # how many chunks to embed at once
EMBED_PAUSE = 0.5      # seconds to wait between batches


# ─── 2. Connect to PostgreSQL ───────────────────────────────────────────────
def get_connection():
    conn = psycopg2.connect(
        host=PGHOST,
        port=PGPORT,
        user=PGUSER,
        password=PGPASSWORD,
        dbname=PGDATABASE
    )
    return conn


# ─── 3. Create the table if it doesn't exist ────────────────────────────────
def setup_table(conn):
    with conn.cursor() as cur:
        register_vector(cur)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id SERIAL PRIMARY KEY,
                source TEXT NOT NULL,
                content TEXT,
                embedding VECTOR(3072),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
    print("Success - table created")


#─── 4. Extract text from the PDF ───────────────────────────────────────────
def extract_text_from_pdf(pdf_path):
    print(f"Reading PDF: {pdf_path}")
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    print(f"Extracted {len(text):,} characters from PDF.")
    return text


#─── 5. Chunk the text ──────────────────────────────────────────────────────
def chunk_text(text):
    print(f"Chunking text (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks = chunk_recursively(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"Created {len(chunks)} chunks")
    return chunks


#─── 6. Embed chunks using Azure OpenAI ─────────────────────────────────────
def embed_chunks(chunks):
    print(f"Embedding {len(chunks)} chunks using Azure OpenAI ({DEPLOY_LARGE})...")

    client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
    )

    all_embeddings = []

    # Process in batches to avoid hitting rate limits
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        print(f"  Embedding batch {i // EMBED_BATCH_SIZE + 1} / {(len(chunks) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE}...")

        response = client.embeddings.create(
            input=batch,
            model=DEPLOY_LARGE
        )

        # Extract the embedding vectors from the response
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

        # Pause briefly to avoid rate limit errors
        time.sleep(EMBED_PAUSE)

    print(f"Generated {len(all_embeddings)} embeddings (each {len(all_embeddings[0])} dimensions).")
    return all_embeddings


#─── 7. Store chunks + embeddings in PostgreSQL ─────────────────────────────
def store_in_db(conn, chunks, embeddings):
    print(f"Storing {len(chunks)} chunks in PostgreSQL...")

    with conn.cursor() as cur:
        register_vector(cur)

        # Build rows as (source, content, embedding)
        rows = [
            (SOURCE_NAME, chunk, embedding)
            for chunk, embedding in zip(chunks, embeddings)
        ]

        # Insert in batches
        for i in range(0, len(rows), DB_BATCH_SIZE):
            batch = rows[i : i + DB_BATCH_SIZE]
            execute_values(
                cur,
                "INSERT INTO knowledge_base (source, content, embedding) VALUES %s",
                batch
            )
            conn.commit()
            print(f"  Stored batch {i // DB_BATCH_SIZE + 1} / {(len(rows) + DB_BATCH_SIZE - 1) // DB_BATCH_SIZE}")

    print(f"All chunks stored")


#─── 8. Main ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nStarting ingestion pipeline...\n")

    # Connect to DB
    conn = get_connection()

    # Set up table
    setup_table(conn)

    # Extract text from PDF
    text = extract_text_from_pdf(PDF_PATH)

    # Chunk the text
    chunks = chunk_text(text)

    # Embed the chunks
    embeddings = embed_chunks(chunks)

    # Store in DB
    store_in_db(conn, chunks, embeddings)

    conn.close()
    print("\nIngestion complete! You can now run query.py to ask questions.\n")