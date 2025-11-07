# rag-demo

A bare bones RAG application for educational purposes.

DISCLAIMER: There are several concepts in this repository that can be implemented in much better ways.  The point of this repository is to remove unfamiliar terms and abstractions as much as possible to demonstrate the essential concepts of a RAG application.

You should get acquainted first with RAG and [when to use it and when not to.](https://www.anthropic.com/news/contextual-retrieval)
Also, feel free to check out the [BGE family of models](https://huggingface.co/BAAI/bge-small-en-v1.5) a series of API accessible models for many RAG pieces such as embeddings, retrieval, reranking, etc. 

## Prerequisites

- **Python 3.12 or higher**: Ensure you have Python 3.12 or a later version installed.
- **Poetry**: Install Poetry on your machine for dependency management.
- **Docker**: Ensure Docker is installed in your WSL environment.
- **SSH Key**: Configure your SSH key with your GitHub account if you haven't already.

Links:
- [Updating Python to 3.12 in WSL](https://stackoverflow.com/questions/78284506/how-to-update-python-to-the-latest-version-3-12-2-in-wsl2)
- [Installing Poetry with Official installer](https://python-poetry.org/docs/#installing-with-the-official-installer)
- [Generating a new SSH key and adding it to the ssh-agent](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)


## Sample output
```
Cleaning database...
Creating embedding for chunk: ize
human priors to ...
Creating embedding for chunk: odifications, ensuri...
Creating embedding for chunk: u et al., 2023; Qian...
Creating embedding for chunk: ned agents and meta-...
Creating embedding for chunk: n Hendrycks, Collin ...

Total index time: 11.419859999790788ms

Enter question: what are the main contributions from the godel agents paper?
scores:  [46.2205144035501, 45.83206449938195, 44.91440161297333, 44.8048227792767, 44.57226661278396]

Using 5 chunks in answer. Answer:

Compiling declarative language model calls into self-improving pipelines
```

## Setup:

This setup works for my WSL installation of Ubuntu with root access.

1. **Clone the repository**:
```bash
git clone git@github.com:smferro54/rag-workshop-from-scratch.git
cd rag-workshop-from-scratch/
```

2. **Start a pgvector docker container**:

```bash
docker run -p 6432:5432  --name pgvector -e POSTGRES_PASSWORD=postgres -d pgvector/pgvector:pg17
```

Note: The host port for the docker container is 6432 instead of the normal 5432 to avoid port collisions

3. **Setup the database: Ensure you are in the directory where the repository was cloned.**
```bash
psql -h localhost -p 6432 -U postgres -c "CREATE DATABASE rag_demo;"
psql -h localhost -p 6432 -U postgres rag_demo < schema.sql
```
Note: All psql commands will prompt for a password, which is **"postgres"**.

*Troubleshoot*: 
```bash
sudo apt update
sudo apt install postgresql-client-common
sudo apt install postgresql-client
```

4. **Create the .env file:**
```bash
touch .env
```
Note: You can use the example_env.txt as a reference for the required environment variables.

5. **Export your Hugging Face API Key environment variable:**
```
source .env
```

6. **Install dependencies with Poetry:**
```
poetry install
```
*Troubleshoot*: 
```
sudo apt update
curl -fsSL https://pyenv.run | bash
pyenv install 3.12.0
pyenv local 3.12.0
```

### Running

`poetry run python -m rag_demo`

You can skip the embedding step if you already have a database and want to experiment with different models. 
`poetry run python -m rag_demo --skip-embedding-step`

#### Chunking Strategy Options

The improved chunking implementation supports multiple strategies:

- **recursive_character** (default): Uses langchain's RecursiveCharacterTextSplitter which respects document structure (paragraphs, sentences, words) and includes overlap between chunks for better context preservation.

- **sentence_transformer**: Uses SentenceTransformersTokenTextSplitter for semantic-aware chunking based on token boundaries.

- **naive**: Original simple character-based splitting (kept for comparison, not recommended for production).

Example usage:
```bash
# Use recursive character splitter with custom chunk size and overlap
poetry run python -m rag_demo --chunking-strategy recursive_character --chunk-size 1024 --chunk-overlap 200

# Use sentence transformer strategy
poetry run python -m rag_demo --chunking-strategy sentence_transformer

# Compare with original naive approach
poetry run python -m rag_demo --chunking-strategy naive --chunk-size 2048
```

#### Two-Stage Retrieval with Reranking

The RAG demo now supports **two-stage retrieval with reranking** to improve retrieval quality. This follows the approach described in [Pinecone's Rerankers Guide](https://www.pinecone.io/learn/series/rag/rerankers/).

**How it works:**
1. **Stage 1 (Vector Search)**: Retrieve a larger set of documents (e.g., top 25) using fast vector similarity search
2. **Stage 2 (Reranking)**: Use a cross-encoder reranker model to rerank the retrieved documents and select the most relevant ones (e.g., top 5)

**Why use reranking?**
- **Better accuracy**: Rerankers process query-document pairs together, avoiding information loss from vector compression
- **Maximizes recall**: Retrieve more documents initially, then rerank to get the best ones
- **Improves LLM performance**: Better context leads to better answers

**Usage:**
```bash
# Enable reranking (recommended for better results)
poetry run python -m rag_demo --use-reranker

# Customize retrieval and reranking parameters
poetry run python -m rag_demo --use-reranker --retrieval-top-k 30 --rerank-top-n 5

# Use a different reranker model
poetry run python -m rag_demo --use-reranker --reranker-model "BAAI/bge-reranker-large"

# Without reranking (single-stage retrieval, faster but less accurate)
poetry run python -m rag_demo
```

**Reranking Options:**
- `--use-reranker`: Enable two-stage retrieval with reranking
- `--retrieval-top-k`: Number of documents to retrieve in first stage (default: 25)
- `--rerank-top-n`: Number of top documents after reranking (default: 5)
- `--reranker-model`: Reranker model to use (default: BAAI/bge-reranker-base)

**Example Output with Reranking:**
```
Stage 1 (Vector Search): Retrieved 25 documents
Vector search scores: ['45.23', '44.12', '43.89', ...]

Stage 2 (Reranking): Reranking 25 documents...
Reranked to top 5 documents
Reranked scores: ['0.9234', '0.8912', '0.8765', '0.8456', '0.8123']
```

#### Query Router Guardrail

Following the guidance in the [LangChain overview](https://docs.langchain.com/oss/python/langchain/overview), the demo now includes a lightweight **query router** to catch obvious malicious intent (prompt-injection phrases, destructive shell commands, credential harvesting attempts) before any retrieval or model call happens.

- Enabled by default; disable with `--disable-query-router`
- Checks for suspicious keywords and rejects empty/oversized queries
- Blocks the request and prompts the user to rephrase if the query is unsafe

Example guardrail output:
```
Query router decision: Query rejected: detected potentially malicious intent (contains disallowed instructions).
This query was blocked because it may contain malicious intent. Please rephrase your request with benign language.
```

### Check chunks table first 5 rows
```
psql -h localhost -p 6432 -U postgres rag_demo -c "SELECT * FROM chunks LIMIT 5;"
```
