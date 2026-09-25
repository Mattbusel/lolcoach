"""lolcoach: an end-to-end League of Legends coaching model pipeline.

The package is organised as eight stages that mirror the lifecycle of the
system:

============  ==========================================================
Module        Responsibility
============  ==========================================================
``sources``   Stage 1. Legally-available data acquisition + provenance.
``ingest``    Stage 2a. Raw payloads -> normalised relational tables.
``features``  Stage 2b. Game-state inference (waves, jungle, macro...).
``dataset``   Stage 2c/3. Structured + synthetic instruction examples.
``rag``       Stage 4/5. Embeddings, FAISS vector store, retrieval.
``train``     Stage 6. LoRA / QLoRA supervised fine-tuning.
``eval``      Stage 7. Benchmarking against frontier APIs.
``cli``       Stage 8. Operator entry point.
============  ==========================================================

Every stage is independently runnable and writes its outputs under the data
root (see :mod:`lolcoach.paths`), so a pipeline can be resumed from any point.
"""

__version__ = "1.1.0"

__all__ = ["__version__"]
