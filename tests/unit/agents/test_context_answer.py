import pytest

from backend.agents.context_answer import ContextAnswerAgent
from backend.llm.models import LLMResponse, LLMUsage
from backend.models.agent import (
    AgentMetadata,
    ContextAnswerAgentInput,
    InvestigationMemory,
    RetrievedDataPoint,
)


@pytest.mark.asyncio
async def test_context_answer_agent_returns_structured_payload(mock_async_function):
    agent = ContextAnswerAgent()
    response = LLMResponse(
        content='{"answer":"Use the users table.","confidence":0.82,"evidence":[{"datapoint_id":"table_users_001","name":"Users","type":"Schema","reason":"Table list"}],"needs_sql":false,"clarifying_questions":[]}',
        model="mock",
        usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        finish_reason="stop",
        provider="mock",
    )
    agent.llm.generate = mock_async_function(return_value=response)

    memory = InvestigationMemory(
        query="what tables exist?",
        datapoints=[
            RetrievedDataPoint(
                datapoint_id="table_users_001",
                datapoint_type="Schema",
                name="Users",
                score=0.9,
                source="vector",
                metadata={"table_name": "users"},
            )
        ],
        total_retrieved=1,
        retrieval_mode="hybrid",
        sources_used=["table_users_001"],
    )

    input_data = ContextAnswerAgentInput(
        query="what tables exist?",
        conversation_history=[],
        investigation_memory=memory,
        intent="exploration",
        context_confidence=0.9,
    )

    output = await agent.execute(input_data)

    assert output.context_answer.answer == "Use the users table."
    assert output.context_answer.confidence == 0.82
    assert output.context_answer.evidence[0].datapoint_id == "table_users_001"
    assert output.context_answer.needs_sql is False
