"""
ContextAgent

Pure retrieval agent (NO LLM calls) that gathers relevant DataPoints for a query.
Uses the Retriever to perform semantic + structural search across the knowledge base.

The ContextAgent:
1. Takes user query (and optional extracted entities)
2. Retrieves relevant DataPoints using vector search + knowledge graph
3. Formats results into InvestigationMemory for downstream agents
4. Tracks sources for citation purposes
5. NO LLM calls - pure retrieval only
"""

import logging
from typing import Any

from backend.agents.base import BaseAgent
from backend.knowledge.retriever import RetrievalMode, Retriever
from backend.models.agent import (
    AgentMetadata,
    ContextAgentInput,
    ContextAgentOutput,
    InvestigationMemory,
    RetrievalError,
    RetrievedDataPoint,
)

logger = logging.getLogger(__name__)


class ContextAgent(BaseAgent):
    """
    Context retrieval agent.

    Retrieves relevant DataPoints from the knowledge base using hybrid
    semantic + structural search. Does NOT use LLMs - pure retrieval only.

    Usage:
        retriever = Retriever(vector_store, knowledge_graph)
        agent = ContextAgent(retriever)

        input = ContextAgentInput(
            query="What were total sales last quarter?",
            entities=[...],  # optional
            retrieval_mode="hybrid",
            max_datapoints=10
        )

        output = await agent(input)
        memory = output.investigation_memory
    """

    def __init__(self, retriever: Retriever):
        """
        Initialize ContextAgent.

        Args:
            retriever: Retriever instance for knowledge base access
        """
        super().__init__(name="ContextAgent")
        self.retriever = retriever

        logger.info("ContextAgent initialized")

    async def execute(self, input: ContextAgentInput) -> ContextAgentOutput:
        """
        Execute context retrieval.

        Args:
            input: ContextAgentInput with query and optional entities

        Returns:
            ContextAgentOutput with InvestigationMemory

        Raises:
            RetrievalError: If retrieval fails
        """
        # Validate input type first
        self._validate_input(input)

        logger.info(f"Retrieving context for query: '{input.query[:100]}...'")

        metadata = AgentMetadata(agent_name=self.name)

        try:
            # Convert retrieval mode string to enum
            mode = RetrievalMode(input.retrieval_mode)

            # Perform retrieval
            result = await self.retriever.retrieve(
                query=input.query,
                mode=mode,
                top_k=input.max_datapoints,
            )

            # Convert retrieval results to RetrievedDataPoints
            datapoints = []
            sources_used = []

            for item in result.items:
                # Extract type from metadata
                # Vector store uses "type" field directly (Schema/Business/Process)
                # Knowledge graph uses "node_type" field (table/column/metric/process/glossary)
                datapoint_type = self._extract_datapoint_type(item.metadata)

                # Extract name from metadata (default to datapoint_id if not found)
                name = item.metadata.get("name", item.datapoint_id)

                retrieved_dp = RetrievedDataPoint(
                    datapoint_id=item.datapoint_id,
                    datapoint_type=datapoint_type,
                    name=name,
                    score=item.score,
                    source=item.source,
                    metadata=item.metadata,
                )

                datapoints.append(retrieved_dp)
                sources_used.append(item.datapoint_id)

            # Create investigation memory
            investigation_memory = InvestigationMemory(
                query=input.query,
                datapoints=datapoints,
                total_retrieved=len(datapoints),
                retrieval_mode=input.retrieval_mode,
                sources_used=list(set(sources_used)),  # Deduplicate
            )

            metadata.mark_complete()

            logger.info(
                f"Successfully retrieved {len(datapoints)} DataPoints "
                f"using {input.retrieval_mode} mode"
            )

            return ContextAgentOutput(
                success=True,
                data={},
                metadata=metadata,
                next_agent="SQLAgent",
                investigation_memory=investigation_memory,
            )

        except Exception as e:
            metadata.error = str(e)
            metadata.mark_complete()

            logger.error(f"Context retrieval failed: {e}")

            raise RetrievalError(
                agent=self.name,
                message=f"Failed to retrieve context: {e}",
                context={"query": input.query, "mode": input.retrieval_mode},
            ) from e

    def _validate_input(self, input: Any) -> None:
        """
        Validate input is ContextAgentInput.

        Args:
            input: Input to validate

        Raises:
            ValueError: If input is not ContextAgentInput
        """
        if not isinstance(input, ContextAgentInput):
            raise ValueError(f"ContextAgent requires ContextAgentInput, got {type(input)}")

    def _extract_datapoint_type(self, metadata: dict[str, Any]) -> str:
        """
        Extract DataPoint type from metadata.

        Vector store uses "type" field directly (Schema/Business/Process).
        Knowledge graph uses "node_type" field (table/column/metric/process/glossary).

        Maps node_type to DataPoint type:
        - table, column → Schema
        - metric, glossary → Business
        - process → Process

        Args:
            metadata: Item metadata from retrieval

        Returns:
            DataPoint type: "Schema", "Business", or "Process"
        """
        # First check if type is already set (vector store)
        if "type" in metadata:
            return metadata["type"]

        # Otherwise map from node_type (knowledge graph)
        node_type = metadata.get("node_type", "")

        # Map NodeType to DataPoint type
        if node_type in ("table", "column"):
            return "Schema"
        elif node_type in ("metric", "glossary"):
            return "Business"
        elif node_type == "process":
            return "Process"
        else:
            # Default to Schema if unknown
            logger.warning(f"Unknown node_type '{node_type}', defaulting to Schema")
            return "Schema"

    def _build_context_for_next(
        self, input: ContextAgentInput, output: ContextAgentOutput
    ) -> dict[str, Any]:
        """
        Build context dictionary for next agent.

        Passes InvestigationMemory in context for downstream agents to use.

        Args:
            input: Original input
            output: Agent output

        Returns:
            Context dict with investigation_memory
        """
        return {
            **input.context,
            "investigation_memory": output.investigation_memory.model_dump(),
        }
