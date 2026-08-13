"""Stages 4 and 5: local embeddings, FAISS index, and grounded retrieval."""

from .index import IndexResult, build_index, chunk_corpus
from .retrieve import LocalRAG, RagHit

__all__ = ["IndexResult", "build_index", "chunk_corpus", "LocalRAG", "RagHit"]
