### Utility libraries
import argparse
import os
import sys
import time
from dotenv import load_dotenv
import requests

### PostgreSQL adapter for Python
import psycopg

### PyPDF for text extraction
from PyPDF2 import PdfReader

### Improved chunking strategies
from rag_demo.chunking import ChunkingConfig, ChunkingStrategy, chunk_text

### Reranking for two-stage retrieval
from rag_demo.reranker import Reranker

### Query router for malicious intent detection
from rag_demo.router import QueryRouter, is_query_safe

### Constants
load_dotenv()
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
CHUNK_SIZE = 1024  # Reduced from 2048 for better granularity with overlap
CHUNK_OVERLAP = 200  # Overlap to preserve context between chunks
EMBEDDINGS_API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-small-en-v1.5"
MODEL_API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
RERANKER_MODEL = "BAAI/bge-reranker-base"
hf_api_key = os.environ.get("HF_API_KEY")
HEADERS = {
    "Authorization": f"""Bearer {hf_api_key}""",
    "Content-Type": "application/json",
    "x-wait-for-model": "true",
}

### Argument parser
parser = argparse.ArgumentParser(description="RAG Demo")
parser.add_argument(
    "--skip-embedding-step",
    action="store_true",
    help="Skip the embedding step and use the existing embeddings if this flag is provided.",
)
parser.add_argument(
    "--chunking-strategy",
    type=str,
    choices=["recursive_character", "sentence_transformer", "naive"],
    default="recursive_character",
    help="Chunking strategy to use (default: recursive_character)",
)
parser.add_argument(
    "--chunk-size",
    type=int,
    default=CHUNK_SIZE,
    help=f"Target chunk size in characters (default: {CHUNK_SIZE})",
)
parser.add_argument(
    "--chunk-overlap",
    type=int,
    default=CHUNK_OVERLAP,
    help=f"Overlap between chunks in characters (default: {CHUNK_OVERLAP})",
)
parser.add_argument(
    "--use-reranker",
    action="store_true",
    help="Enable two-stage retrieval with reranking (improves retrieval quality)",
)
parser.add_argument(
    "--retrieval-top-k",
    type=int,
    default=25,
    help="Number of documents to retrieve in first stage (vector search) (default: 25)",
)
parser.add_argument(
    "--rerank-top-n",
    type=int,
    default=5,
    help="Number of top documents after reranking (default: 5)",
)
parser.add_argument(
    "--reranker-model",
    type=str,
    default=RERANKER_MODEL,
    help=f"Reranker model to use (default: {RERANKER_MODEL})",
)
parser.add_argument(
    "--disable-query-router",
    action="store_true",
    help="Disable the malicious intent query router guardrail",
)
args = parser.parse_args()

### Useful functions [can go to a utils.py file]
def get_embedding(payload):
    response = requests.post(
        EMBEDDINGS_API_URL,
        headers=HEADERS,
        json=payload,
    )
    return response.json()

def get_answer(payload):
    response = requests.post(
        MODEL_API_URL,
        headers=HEADERS,
        json=payload,
    )
    return response.json()


### PostgreSQL database url and connection
database_url = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:6432/rag_demo"
)
db = psycopg.Connection.connect(database_url)


# Initialize chunking configuration
chunking_config = ChunkingConfig(
    chunk_size=args.chunk_size,
    chunk_overlap=args.chunk_overlap,
    strategy=ChunkingStrategy(args.chunking_strategy),
)

print(f"Using chunking strategy: {chunking_config.strategy.value}")
print(f"Chunk size: {chunking_config.chunk_size}, Overlap: {chunking_config.chunk_overlap}")

# Initialize query router unless disabled
query_router: QueryRouter | None = None
if args.disable_query_router:
    print("Query router disabled (processing all queries)")
else:
    query_router = QueryRouter()
    print("Query router enabled: malicious intent checks active")

# Initialize reranker if enabled
reranker = None
if args.use_reranker:
    reranker = Reranker(model_name=args.reranker_model, api_key=hf_api_key)
    print(f"Reranking enabled: {args.reranker_model}")
    print(f"  Retrieval top_k: {args.retrieval_top_k}, Rerank top_n: {args.rerank_top_n}")
else:
    print("Reranking disabled (using single-stage retrieval)")

# Loop through chunks from the pdf and create embeddings in the database

if not args.skip_embedding_step:
    print("Cleaning database...")
    db.execute("TRUNCATE TABLE chunks")

    tic = time.perf_counter()
    total_chunks = 0
    
    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)
        print(f"\nProcessing file: {filename}")

        reader = PdfReader(file_path)
        content = ""
        for page in reader.pages:
            content += page.extract_text()

        # Use improved chunking strategy
        chunks = chunk_text(content, chunking_config)
        print(f"  Created {len(chunks)} chunks from {filename}")
        total_chunks += len(chunks)

        for chunk in chunks:
            print(f"  Creating embedding for chunk: {chunk[:50]}...")
            
            db.execute(
                "INSERT INTO chunks (embedding, chunk) VALUES (%s, %s)",
                [str(get_embedding(chunk)), chunk],
            )

    print(f"\nTotal chunks created: {total_chunks}")
    print(f"Total index time: {time.perf_counter() - tic:.2f}s")
    db.commit()

question = input("\nEnter question: ")

if query_router is not None:
    is_allowed, message = is_query_safe(question, query_router)
    print(f"Query router decision: {message}")
    if not is_allowed:
        print(
            "This query was blocked because it may contain malicious intent."
            " Please rephrase your request with benign language."
        )
        sys.exit(0)

# Create embedding from question.  Many RAG applications use a query rewriter before querying
# the vector database.  For more information on query rewriting, see this whitepaper:
#    https://arxiv.org/abs/2305.14283
question_embedding = get_embedding(question)

# Stage 1: Vector search - retrieve more documents than needed
# This maximizes retrieval recall
retrieval_k = args.retrieval_top_k if args.use_reranker else args.rerank_top_n
# Use parameterized query to prevent SQL injection
result = db.execute(
    "SELECT (embedding <=> %s::vector)*100 as score, chunk FROM chunks ORDER BY score DESC LIMIT %s", 
    (question_embedding, retrieval_k)
)

rows = list(result)

# Stage 2: Reranking (if enabled)
if args.use_reranker and reranker:
    print(f"\nStage 1 (Vector Search): Retrieved {len(rows)} documents")
    print("Vector search scores:", [f"{row[0]:.2f}" for row in rows])
    
    # Extract document texts for reranking
    documents = [row[1] for row in rows]
    
    # Rerank documents
    print(f"\nStage 2 (Reranking): Reranking {len(documents)} documents...")
    reranked_results = reranker.rerank(question, documents, top_n=args.rerank_top_n)
    
    # Update rows with reranked results
    rows = [(0.0, doc) for doc, score in reranked_results]  # Score not used after reranking
    print(f"Reranked to top {len(rows)} documents")
    print("Reranked scores:", [f"{score:.4f}" for _, score in reranked_results])
else:
    print(f"\nRetrieved {len(rows)} documents (single-stage retrieval)")
    print("Vector search scores:", [f"{row[0]:.2f}" for row in rows])

context = "\n\n".join([row[1] for row in rows])

prompt = f"""
Answer the question using only the following context:

{context}

Question: {question}
"""

answer = get_answer(
    {
        "inputs": {
            "question": question,
            "context": context,
        }
    })

print(f"\nUsing {len(rows)} chunks in answer. Answer:\n")
print(answer["answer"])

view_prompt = input("\nWould you like to see the raw prompt? [Y/N] ")
if view_prompt == "Y":
    print("\n" + prompt)