"""Provider-optional RAG building blocks.

The package intentionally keeps retrieval, ranking, context formatting and
citation validation separate.  Each module can therefore be tested with plain
Python data and the application graph can record a structured result without
reaching back into the legacy recall service.
"""

from app.services.materials.rag.citations import validate_citations
from app.services.materials.rag.context import build_context
from app.services.materials.rag.query_rewrite import rewrite_queries
from app.services.materials.rag.ranking import fuse_candidates, govern_candidates, rerank_candidates
from app.services.materials.rag.retrievers import dual_retrieve

__all__ = [
    "build_context",
    "dual_retrieve",
    "fuse_candidates",
    "govern_candidates",
    "rerank_candidates",
    "rewrite_queries",
    "validate_citations",
]
