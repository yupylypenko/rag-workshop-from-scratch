"""
Improved chunking strategies for RAG applications.

This module provides multiple chunking strategies that respect sentence boundaries,
support overlap, and use proven algorithms from langchain.
"""

from __future__ import annotations

from enum import Enum
from typing import List

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    RECURSIVE_CHARACTER = "recursive_character"
    SENTENCE_TRANSFORMER = "sentence_transformer"
    NAIVE = "naive"  # Original naive implementation for comparison


class ChunkingConfig:
    """
    Configuration for chunking strategies.

    Attributes:
        chunk_size: Target size of chunks (in characters or tokens)
        chunk_overlap: Overlap between chunks to preserve context
        strategy: Chunking strategy to use
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE_CHARACTER,
    ):
        """
        Initialize chunking configuration.

        Args:
            chunk_size: Target size of chunks
            chunk_overlap: Overlap between chunks
            strategy: Chunking strategy to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy


def chunk_text_naive(text: str, chunk_size: int) -> List[str]:
    """
    Naive chunking by character length (original implementation).

    This is kept for comparison but should not be used in production.

    Args:
        text: Text to chunk
        chunk_size: Size of each chunk

    Returns:
        List of text chunks
    """
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def chunk_text_recursive_character(
    text: str, chunk_size: int, chunk_overlap: int
) -> List[str]:
    """
    Recursive character text splitter with overlap.

    This strategy tries to split on paragraph boundaries first, then sentences,
    then words, and finally characters if needed. It respects document structure
    and includes overlap to preserve context.

    Args:
        text: Text to chunk
        chunk_size: Target size of chunks (in characters)
        chunk_overlap: Overlap between chunks (in characters)

    Returns:
        List of text chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def chunk_text_sentence_transformer(
    text: str, chunk_size: int, chunk_overlap: int
) -> List[str]:
    """
    Sentence transformer token-based text splitter.

    This strategy uses a sentence transformer model to split text based on
    semantic boundaries. It's more sophisticated but requires more resources.

    Args:
        text: Text to chunk
        chunk_size: Target size of chunks (in tokens)
        chunk_overlap: Overlap between chunks (in tokens)

    Returns:
        List of text chunks
    """
    splitter = SentenceTransformersTokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model_name="BAAI/bge-small-en-v1.5",  # Same model used for embeddings
    )
    return splitter.split_text(text)


def chunk_text(text: str, config: ChunkingConfig) -> List[str]:
    """
    Chunk text using the specified strategy.

    Args:
        text: Text to chunk
        config: Chunking configuration

    Returns:
        List of text chunks

    Raises:
        ValueError: If an unknown strategy is specified
    """
    if config.strategy == ChunkingStrategy.NAIVE:
        return chunk_text_naive(text, config.chunk_size)
    elif config.strategy == ChunkingStrategy.RECURSIVE_CHARACTER:
        return chunk_text_recursive_character(
            text, config.chunk_size, config.chunk_overlap
        )
    elif config.strategy == ChunkingStrategy.SENTENCE_TRANSFORMER:
        return chunk_text_sentence_transformer(
            text, config.chunk_size, config.chunk_overlap
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {config.strategy}")
