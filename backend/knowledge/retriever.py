"""
Knowledge Retriever

Combines VectorStore and KnowledgeGraph for unified contextual retrieval.
Supports local (vector), global (graph), and hybrid retrieval modes.
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from backend.knowledge.graph import KnowledgeGraph
from backend.knowledge.vectors import VectorStore

logger = logging.getLogger(__name__)


class RetrievalMode(str, Enum):
    """Retrieval modes for context gathering."""

    LOCAL = "local"  # Vector search only (semantic similarity)
    GLOBAL = "global"  # Graph traversal only (structural relationships)
    HYBRID = "hybrid"  # Both vector and graph (combined with RRF)


class RetrievedItem(BaseModel):
    """A single retrieved item with metadata."""

    datapoint_id: str = Field(..., description="DataPoint identifier")
    score: float = Field(..., description="Relevance score (0-1, higher is better)")
    source: str = Field(..., description="Retrieval source (vector/graph/hybrid)")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    content: Optional[str] = Field(None, description="Retrieved content/document")


class RetrievalResult(BaseModel):
    """Result from retrieval operation."""

    items: List[RetrievedItem] = Field(
        default_factory=list, description="Retrieved items ranked by relevance"
    )
    total_count: int = Field(..., description="Total number of items retrieved")
    mode: RetrievalMode = Field(..., description="Retrieval mode used")
    query: str = Field(..., description="Original query")


class RetrieverError(Exception):
    """Raised when retrieval operations fail."""

    pass


class Retriever:
    """
    Unified retriever combining vector search and graph traversal.

    Supports three retrieval modes:
    - Local: Semantic search via vector embeddings
    - Global: Structural search via knowledge graph
    - Hybrid: Combined search with RRF ranking

    Usage:
        retriever = Retriever(vector_store, knowledge_graph)

        # Local mode (vector search)
        result = await retriever.retrieve("revenue metrics", mode="local", top_k=5)

        # Global mode (graph traversal)
        result = await retriever.retrieve(
            "metric_revenue_001",
            mode="global",
            top_k=10
        )

        # Hybrid mode (combined)
        result = await retriever.retrieve(
            "sales data analysis",
            mode="hybrid",
            top_k=10
        )
    """

    def __init__(
        self,
        vector_store: VectorStore,
        knowledge_graph: KnowledgeGraph,
        rrf_k: int = 60,
    ):
        """
        Initialize the retriever.

        Args:
            vector_store: VectorStore instance for semantic search
            knowledge_graph: KnowledgeGraph instance for structural search
            rrf_k: RRF constant (default: 60, standard value from literature)
        """
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.rrf_k = rrf_k

        logger.info(f"Retriever initialized with RRF k={rrf_k}")

    async def retrieve(
        self,
        query: str,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        top_k: int = 10,
        vector_top_k: Optional[int] = None,
        graph_top_k: Optional[int] = None,
        graph_max_depth: int = 2,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant DataPoints based on query.

        Args:
            query: Search query (text for local/hybrid, node ID for global)
            mode: Retrieval mode (local/global/hybrid)
            top_k: Number of results to return
            vector_top_k: Top-k for vector search (default: top_k * 2 for hybrid)
            graph_top_k: Top-k for graph search (default: top_k * 2 for hybrid)
            graph_max_depth: Max graph traversal depth (default: 2)
            metadata_filter: Optional metadata filter for vector search

        Returns:
            RetrievalResult with ranked items

        Raises:
            RetrieverError: If retrieval fails
        """
        try:
            if mode == RetrievalMode.LOCAL:
                items = await self._retrieve_local(query, top_k, metadata_filter)
            elif mode == RetrievalMode.GLOBAL:
                items = await self._retrieve_global(query, top_k, graph_max_depth)
            elif mode == RetrievalMode.HYBRID:
                items = await self._retrieve_hybrid(
                    query,
                    top_k,
                    vector_top_k or top_k * 2,
                    graph_top_k or top_k * 2,
                    graph_max_depth,
                    metadata_filter,
                )
            else:
                raise RetrieverError(f"Unknown retrieval mode: {mode}")

            logger.info(
                f"Retrieved {len(items)} items using {mode} mode for query: '{query[:50]}...'"
            )

            return RetrievalResult(
                items=items, total_count=len(items), mode=mode, query=query
            )

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise RetrieverError(f"Retrieval failed: {e}") from e

    async def _retrieve_local(
        self, query: str, top_k: int, metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedItem]:
        """Retrieve using vector search only."""
        vector_results = await self.vector_store.search(
            query, top_k=top_k, filter_metadata=metadata_filter
        )

        items = []
        for result in vector_results:
            # Convert distance to similarity score (cosine distance: lower is better)
            # Score = 1 / (1 + distance), normalized to 0-1 range
            distance = result["distance"]
            score = 1.0 / (1.0 + abs(distance))

            items.append(
                RetrievedItem(
                    datapoint_id=result["datapoint_id"],
                    score=score,
                    source="vector",
                    metadata=result["metadata"],
                    content=result.get("document"),
                )
            )

        logger.debug(f"Vector search returned {len(items)} items")
        return items

    async def _retrieve_global(
        self, node_id: str, top_k: int, max_depth: int
    ) -> List[RetrievedItem]:
        """Retrieve using graph traversal only."""
        # Get related nodes from graph
        related_nodes = self.knowledge_graph.get_related(
            node_id, max_depth=max_depth
        )

        items = []
        for node in related_nodes[:top_k]:
            # Convert distance to score (graph distance: 1 is closest)
            # Score = 1 / distance, normalized
            distance = node["distance"]
            score = 1.0 / distance if distance > 0 else 1.0

            items.append(
                RetrievedItem(
                    datapoint_id=node["node_id"],
                    score=score,
                    source="graph",
                    metadata=node["metadata"],
                    content=None,  # Graph doesn't have document content
                )
            )

        logger.debug(f"Graph traversal returned {len(items)} items")
        return items

    async def _retrieve_hybrid(
        self,
        query: str,
        top_k: int,
        vector_top_k: int,
        graph_top_k: int,
        graph_max_depth: int,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedItem]:
        """Retrieve using both vector and graph, combined with RRF."""
        # Get vector results
        vector_results = await self.vector_store.search(
            query, top_k=vector_top_k, filter_metadata=metadata_filter
        )

        vector_items = {}
        for rank, result in enumerate(vector_results, start=1):
            datapoint_id = result["datapoint_id"]
            distance = result["distance"]
            score = 1.0 / (1.0 + abs(distance))

            vector_items[datapoint_id] = {
                "rank": rank,
                "score": score,
                "metadata": result["metadata"],
                "content": result.get("document"),
            }

        # Get graph results (if we have a seed node)
        graph_items = {}
        if vector_results:
            # Use top vector result as seed for graph traversal
            seed_node = vector_results[0]["datapoint_id"]

            try:
                related_nodes = self.knowledge_graph.get_related(
                    seed_node, max_depth=graph_max_depth
                )

                for rank, node in enumerate(related_nodes[:graph_top_k], start=1):
                    datapoint_id = node["node_id"]
                    distance = node["distance"]
                    score = 1.0 / distance if distance > 0 else 1.0

                    graph_items[datapoint_id] = {
                        "rank": rank,
                        "score": score,
                        "metadata": node["metadata"],
                        "content": None,
                    }
            except Exception as e:
                logger.warning(f"Graph traversal failed, using vector only: {e}")

        # Apply RRF (Reciprocal Rank Fusion)
        rrf_scores = self._apply_rrf(vector_items, graph_items)

        # Deduplicate and create final items
        items = []
        seen_ids: Set[str] = set()

        for datapoint_id, rrf_score in sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_k]:
            if datapoint_id in seen_ids:
                continue
            seen_ids.add(datapoint_id)

            # Get metadata from either source
            metadata = vector_items.get(datapoint_id, {}).get(
                "metadata", graph_items.get(datapoint_id, {}).get("metadata", {})
            )
            content = vector_items.get(datapoint_id, {}).get("content")

            # Determine source
            in_vector = datapoint_id in vector_items
            in_graph = datapoint_id in graph_items
            source = "hybrid" if (in_vector and in_graph) else (
                "vector" if in_vector else "graph"
            )

            items.append(
                RetrievedItem(
                    datapoint_id=datapoint_id,
                    score=rrf_score,
                    source=source,
                    metadata=metadata,
                    content=content,
                )
            )

        logger.debug(
            f"Hybrid retrieval: {len(vector_items)} vector + {len(graph_items)} graph "
            f"→ {len(items)} final items"
        )

        return items

    def _apply_rrf(
        self,
        vector_items: Dict[str, Dict[str, Any]],
        graph_items: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Apply Reciprocal Rank Fusion (RRF) to combine rankings.

        RRF formula: score(d) = Σ 1 / (k + rank(d))
        where k is a constant (typically 60) and rank(d) is the rank of document d.

        Args:
            vector_items: Dict of {datapoint_id: {rank, score, ...}}
            graph_items: Dict of {datapoint_id: {rank, score, ...}}

        Returns:
            Dict of {datapoint_id: rrf_score}
        """
        rrf_scores = {}

        # Get all unique datapoint IDs
        all_ids = set(vector_items.keys()) | set(graph_items.keys())

        for datapoint_id in all_ids:
            rrf_score = 0.0

            # Add vector contribution
            if datapoint_id in vector_items:
                vector_rank = vector_items[datapoint_id]["rank"]
                rrf_score += 1.0 / (self.rrf_k + vector_rank)

            # Add graph contribution
            if datapoint_id in graph_items:
                graph_rank = graph_items[datapoint_id]["rank"]
                rrf_score += 1.0 / (self.rrf_k + graph_rank)

            rrf_scores[datapoint_id] = rrf_score

        return rrf_scores
