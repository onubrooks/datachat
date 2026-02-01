"""
Vector Store

Chroma-based vector store for DataPoint embeddings with async interface.
Supports semantic search, persistence, and metadata storage.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from backend.config import get_settings
from backend.models.datapoint import DataPoint

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Raised when vector store operations fail."""

    pass


class VectorStore:
    """
    Vector store for DataPoint embeddings using Chroma.

    Provides async interface for adding, searching, and deleting DataPoints
    with semantic search capabilities. Uses OpenAI embeddings by default.

    Usage:
        store = VectorStore()
        await store.initialize()

        # Add datapoints
        await store.add_datapoints([datapoint1, datapoint2])

        # Search
        results = await store.search("sales data", top_k=5)

        # Delete
        await store.delete(["datapoint_id_1", "datapoint_id_2"])
    """

    def __init__(
        self,
        collection_name: str | None = None,
        persist_directory: str | Path | None = None,
        embedding_model: str | None = None,
        openai_api_key: str | None = None,
    ):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the Chroma collection (default from config)
            persist_directory: Directory for persistence (default from config)
            embedding_model: OpenAI embedding model (default from config)
            openai_api_key: OpenAI API key (default from config)
        """
        # Only load config if needed (allows tests to avoid config validation)
        if (
            collection_name is None
            or persist_directory is None
            or embedding_model is None
            or openai_api_key is None
        ):
            config = get_settings()
            self.collection_name = collection_name or config.chroma.collection_name
            self.persist_directory = Path(persist_directory or config.chroma.persist_dir)
            self.embedding_model = embedding_model or config.chroma.embedding_model
            self.openai_api_key = openai_api_key or config.llm.openai_api_key
        else:
            self.collection_name = collection_name
            self.persist_directory = Path(persist_directory)
            self.embedding_model = embedding_model
            self.openai_api_key = openai_api_key

        self.client: chromadb.ClientAPI | None = None
        self.collection: chromadb.Collection | None = None
        self.embedding_function: OpenAIEmbeddingFunction | None = None

        logger.info(
            f"VectorStore initialized: collection={self.collection_name}, "
            f"persist_dir={self.persist_directory}, embedding_model={self.embedding_model}"
        )

    async def initialize(self):
        """
        Initialize the Chroma client and collection.

        Must be called before using the vector store.
        Creates the persist directory if it doesn't exist.

        Raises:
            VectorStoreError: If initialization fails
        """
        try:
            # Ensure persist directory exists
            self.persist_directory.mkdir(parents=True, exist_ok=True)

            # Initialize Chroma client (sync, will wrap in to_thread)
            await asyncio.to_thread(self._init_client)

            logger.info("VectorStore initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize VectorStore: {e}")
            raise VectorStoreError(f"Initialization failed: {e}") from e

    def _init_client(self):
        """Initialize Chroma client (sync, called via to_thread)."""
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Create OpenAI embedding function
        self.embedding_function = OpenAIEmbeddingFunction(
            api_key=self.openai_api_key,
            model_name=self.embedding_model,
        )

        # Get or create collection with OpenAI embeddings
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_function,
        )

        logger.debug(
            f"Chroma collection '{self.collection_name}' ready "
            f"with {self.collection.count()} documents"
        )

    async def add_datapoints(
        self,
        datapoints: list[DataPoint],
        batch_size: int = 100,
    ) -> int:
        """
        Add DataPoints to the vector store.

        Generates embeddings for each DataPoint's searchable content and
        stores them with metadata.

        Args:
            datapoints: List of DataPoint objects to add
            batch_size: Number of datapoints to add per batch

        Returns:
            Number of datapoints successfully added

        Raises:
            VectorStoreError: If adding fails
        """
        if not self.collection:
            raise VectorStoreError("VectorStore not initialized. Call initialize() first.")

        if not datapoints:
            logger.warning("No datapoints to add")
            return 0

        try:
            total_added = 0

            # Process in batches
            for i in range(0, len(datapoints), batch_size):
                batch = datapoints[i : i + batch_size]

                # Prepare batch data
                ids = [dp.datapoint_id for dp in batch]
                documents = [self._create_document(dp) for dp in batch]
                metadatas = [self._create_metadata(dp) for dp in batch]

                # Add to Chroma (async wrapper)
                await asyncio.to_thread(
                    self.collection.add,
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                )

                total_added += len(batch)
                logger.debug(f"Added batch of {len(batch)} datapoints ({total_added} total)")

            logger.info(f"Successfully added {total_added} datapoints to vector store")
            return total_added

        except Exception as e:
            logger.error(f"Failed to add datapoints: {e}")
            raise VectorStoreError(f"Failed to add datapoints: {e}") from e

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar DataPoints using semantic search.

        Args:
            query: Search query text
            top_k: Number of results to return
            filter_metadata: Optional metadata filters (e.g., {"type": "Schema"})

        Returns:
            List of search results with datapoint_id, distance, metadata, and document

        Raises:
            VectorStoreError: If search fails
        """
        if not self.collection:
            raise VectorStoreError("VectorStore not initialized. Call initialize() first.")

        try:
            # Query Chroma (async wrapper)
            results = await asyncio.to_thread(
                self.collection.query,
                query_texts=[query],
                n_results=top_k,
                where=filter_metadata,
            )

            # Format results
            formatted_results = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    formatted_results.append(
                        {
                            "datapoint_id": results["ids"][0][i],
                            "distance": results["distances"][0][i]
                            if results["distances"]
                            else None,
                            "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                            "document": results["documents"][0][i] if results["documents"] else "",
                        }
                    )

            logger.debug(f"Search for '{query}' returned {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise VectorStoreError(f"Search failed: {e}") from e

    async def delete(self, datapoint_ids: list[str]) -> int:
        """
        Delete DataPoints from the vector store.

        Args:
            datapoint_ids: List of datapoint IDs to delete

        Returns:
            Number of datapoints deleted

        Raises:
            VectorStoreError: If deletion fails
        """
        if not self.collection:
            raise VectorStoreError("VectorStore not initialized. Call initialize() first.")

        if not datapoint_ids:
            logger.warning("No datapoint IDs to delete")
            return 0

        try:
            # Delete from Chroma (async wrapper)
            await asyncio.to_thread(
                self.collection.delete,
                ids=datapoint_ids,
            )

            logger.info(f"Deleted {len(datapoint_ids)} datapoints from vector store")
            return len(datapoint_ids)

        except Exception as e:
            logger.error(f"Failed to delete datapoints: {e}")
            raise VectorStoreError(f"Failed to delete datapoints: {e}") from e

    async def get_count(self) -> int:
        """
        Get the total number of DataPoints in the store.

        Returns:
            Number of datapoints

        Raises:
            VectorStoreError: If operation fails
        """
        if not self.collection:
            raise VectorStoreError("VectorStore not initialized. Call initialize() first.")

        try:
            count = await asyncio.to_thread(self.collection.count)
            return count

        except Exception as e:
            logger.error(f"Failed to get count: {e}")
            raise VectorStoreError(f"Failed to get count: {e}") from e

    async def list_datapoints(self, limit: int = 1000, offset: int = 0) -> list[dict[str, Any]]:
        """
        List DataPoints without embedding calls.

        Args:
            limit: Maximum number of datapoints to return
            offset: Offset for pagination

        Returns:
            List of datapoints with metadata
        """
        if not self.collection:
            raise VectorStoreError("VectorStore not initialized. Call initialize() first.")

        try:
            results = await asyncio.to_thread(
                self.collection.get,
                limit=limit,
                offset=offset,
                include=["metadatas"],
            )
            items = []
            ids = results.get("ids") or []
            metadatas = results.get("metadatas") or []
            for idx, datapoint_id in enumerate(ids):
                metadata = metadatas[idx] if idx < len(metadatas) else {}
                items.append(
                    {
                        "datapoint_id": datapoint_id,
                        "metadata": metadata,
                    }
                )
            return items
        except Exception as e:
            logger.error(f"Failed to list datapoints: {e}")
            raise VectorStoreError(f"Failed to list datapoints: {e}") from e

    async def clear(self):
        """
        Clear all DataPoints from the vector store.

        Warning: This removes all data from the collection.

        Raises:
            VectorStoreError: If operation fails
        """
        if not self.collection:
            raise VectorStoreError("VectorStore not initialized. Call initialize() first.")

        try:
            # Get all IDs and delete them
            results = await asyncio.to_thread(
                self.collection.get,
                limit=1000000,  # Large limit to get all
            )

            if results["ids"]:
                await asyncio.to_thread(
                    self.collection.delete,
                    ids=results["ids"],
                )
                logger.info(f"Cleared {len(results['ids'])} datapoints from vector store")
            else:
                logger.info("Vector store already empty")

        except Exception as e:
            logger.error(f"Failed to clear vector store: {e}")
            raise VectorStoreError(f"Failed to clear: {e}") from e

    def _create_document(self, datapoint: DataPoint) -> str:
        """
        Create searchable document text from a DataPoint.

        Combines name, description/purpose, and type-specific fields
        into a single searchable text.

        Args:
            datapoint: DataPoint object

        Returns:
            Document text for embedding
        """
        parts = [
            f"Name: {datapoint.name}",
            f"Type: {datapoint.type}",
        ]

        # Add type-specific fields
        if hasattr(datapoint, "business_purpose"):
            parts.append(f"Purpose: {datapoint.business_purpose}")

        if hasattr(datapoint, "calculation"):
            parts.append(f"Calculation: {datapoint.calculation}")

        if hasattr(datapoint, "synonyms") and datapoint.synonyms:
            parts.append(f"Synonyms: {', '.join(datapoint.synonyms)}")

        if hasattr(datapoint, "business_rules") and datapoint.business_rules:
            parts.append(f"Rules: {'; '.join(datapoint.business_rules)}")

        if hasattr(datapoint, "table_name"):
            parts.append(f"Table: {datapoint.table_name}")

        if hasattr(datapoint, "schedule"):
            parts.append(f"Schedule: {datapoint.schedule}")

        # Add tags
        if datapoint.tags:
            parts.append(f"Tags: {', '.join(datapoint.tags)}")

        return "\n".join(parts)

    def _create_metadata(self, datapoint: DataPoint) -> dict[str, Any]:
        """
        Create metadata dictionary for a DataPoint.

        Args:
            datapoint: DataPoint object

        Returns:
            Metadata dictionary
        """
        metadata = {
            "datapoint_id": datapoint.datapoint_id,
            "type": datapoint.type,
            "name": datapoint.name,
            "owner": datapoint.owner,
        }

        # Add type-specific metadata
        if hasattr(datapoint, "table_name"):
            metadata["table_name"] = datapoint.table_name
            metadata["schema"] = datapoint.schema_name

        if hasattr(datapoint, "related_tables") and datapoint.related_tables:
            metadata["related_tables"] = ",".join(datapoint.related_tables)

        if datapoint.tags:
            metadata["tags"] = ",".join(datapoint.tags)

        return metadata
