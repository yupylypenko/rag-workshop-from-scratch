"""
Reranking module for two-stage retrieval in RAG applications.

Implements reranking using cross-encoder models to improve retrieval quality
by reranking initial vector search results.
"""

from __future__ import annotations

from typing import List, Optional

import requests


class Reranker:
    """
    Reranker for two-stage retrieval.

    Uses cross-encoder models to rerank documents based on query relevance.
    This improves retrieval quality by processing query-document pairs together.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize reranker.

        Args:
            model_name: Hugging Face model name for reranking
            api_url: Optional custom API URL (defaults to Hugging Face Inference API)
            api_key: Optional API key (uses HF_API_KEY env var if not provided)
        """
        self.model_name = model_name
        self.api_url = api_url or f"https://api-inference.huggingface.co/models/{model_name}"
        self.api_key = api_key

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[tuple[str, float]]:
        """
        Rerank documents based on query relevance.

        Args:
            query: User query
            documents: List of document texts to rerank
            top_n: Number of top documents to return (None returns all)

        Returns:
            List of tuples (document_text, relevance_score) sorted by relevance
        """
        if not documents:
            return []

        # Prepare input pairs for reranking
        # Format: [query, document] pairs
        inputs = [[query, doc] for doc in documents]

        headers = {
            "Authorization": f"Bearer {self.api_key or ''}",
            "Content-Type": "application/json",
            "x-wait-for-model": "true",
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json={"inputs": inputs},
                timeout=30,
            )
            response.raise_for_status()
            scores = response.json()

            # Handle different response formats
            if isinstance(scores, list) and len(scores) > 0:
                if isinstance(scores[0], list):
                    # Format: [[score1], [score2], ...]
                    scores = [s[0] if isinstance(s, list) else s for s in scores]
                elif isinstance(scores[0], dict):
                    # Format: [{"score": ...}, ...]
                    scores = [s.get("score", 0.0) for s in scores]

            # Pair documents with scores and sort by score (descending)
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)

            # Return top_n if specified
            if top_n is not None:
                return doc_scores[:top_n]

            return doc_scores

        except requests.exceptions.RequestException as e:
            print(f"Warning: Reranking failed: {e}")
            print("Falling back to original order without reranking.")
            # Return documents in original order with neutral scores
            return [(doc, 0.5) for doc in documents]

    def rerank_with_metadata(
        self,
        query: str,
        documents: List[tuple[str, float]],
        top_n: Optional[int] = None,
    ) -> List[tuple[str, float]]:
        """
        Rerank documents that already have scores (e.g., from vector search).

        Args:
            query: User query
            documents: List of tuples (document_text, initial_score)
            top_n: Number of top documents to return (None returns all)

        Returns:
            List of tuples (document_text, reranked_score) sorted by relevance
        """
        # Extract just the document texts for reranking
        doc_texts = [doc[0] for doc in documents]
        return self.rerank(query, doc_texts, top_n=top_n)

