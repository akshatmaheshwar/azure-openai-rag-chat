import json

import pytest
from azure.search.documents.aio import SearchClient
from openai import AsyncOpenAI

import quartapp

from . import mock_cred

NO_RESULTS_MESSAGE = "I couldn't find that in the documents."


def user_input(question: str) -> dict:
    return {
        "input": [
            {
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": "You are a helpful assistant."}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": question}],
            },
        ]
    }


async def get_events(response) -> list[dict]:
    return [json.loads(line) for line in (await response.get_data(as_text=True)).splitlines()]


@pytest.mark.asyncio
async def test_index(client):
    response = await client.get("/")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_chat_stream_text(client, snapshot):
    response = await client.post(
        "/chat/stream",
        json={
            "input": [
                {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": "You are a helpful assistant."}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "What is the capital of France?"}],
                },
            ]
        },
    )
    assert response.status_code == 200
    result = await response.get_data()
    snapshot.assert_match(result, "result.jsonlines")


@pytest.mark.asyncio
async def test_chat_stream_text_history(client, snapshot):
    response = await client.post(
        "/chat/stream",
        json={
            "input": [
                {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": "You are a helpful assistant."}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "What is the capital of France?"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Paris"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "What is the capital of Germany?"}],
                },
            ]
        },
    )
    assert response.status_code == 200
    result = await response.get_data()
    snapshot.assert_match(result, "result.jsonlines")


@pytest.mark.asyncio
async def test_chat_stream_searches_latest_user_message(client, mock_search):
    request_json = user_input("What is the capital of France?")
    request_json["input"] += [
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Paris"}]},
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "What is the capital of Germany?"}],
        },
    ]
    response = await client.post("/chat/stream", json=request_json)
    assert response.status_code == 200
    await response.get_data()
    assert mock_search == ["What is the capital of Germany?"]


@pytest.mark.asyncio
async def test_chat_stream_answer_with_sources(client, mock_openai_responses_stream):
    response = await client.post("/chat/stream", json=user_input("How many days of annual leave do I get?"))
    assert response.status_code == 200
    events = await get_events(response)

    assert events[0] == {
        "type": "sources",
        "sources": [{"title": "Leave Policy", "source": "docs/policies/leave-policy.md"}],
    }
    answer = "".join(event["delta"] for event in events[1:] if event["type"] == "response.output_text.delta")
    assert answer == "You get 25 days of annual leave [Leave Policy]."

    assert len(mock_openai_responses_stream) == 1
    instructions = mock_openai_responses_stream[0]["instructions"]
    assert "using ONLY the sources below" in instructions
    assert "[Leave Policy] (source: docs/policies/leave-policy.md)" in instructions
    assert "25 days of paid annual leave" in instructions


@pytest.mark.asyncio
async def test_chat_stream_instructions_require_fixed_reply_when_answer_missing(client, mock_openai_responses_stream):
    response = await client.post("/chat/stream", json=user_input("How many days of annual leave do I get?"))
    assert response.status_code == 200
    await response.get_data()

    instructions = mock_openai_responses_stream[0]["instructions"]
    assert f'If the sources do not contain the answer, reply exactly "{NO_RESULTS_MESSAGE}" and nothing else.' in (
        instructions
    )


@pytest.mark.asyncio
async def test_chat_stream_no_results_does_not_call_model(client, mock_openai_responses_stream):
    response = await client.post("/chat/stream", json=user_input("What is the meaning of life?"))
    assert response.status_code == 200
    events = await get_events(response)

    assert events == [
        {"type": "sources", "sources": []},
        {"type": "response.output_text.delta", "delta": NO_RESULTS_MESSAGE},
    ]
    assert mock_openai_responses_stream == []


@pytest.mark.asyncio
async def test_openai_managedidentity(monkeypatch):
    monkeypatch.setenv("AZURE_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test-openai-service.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "test-chatgpt")
    monkeypatch.setenv("AZURE_SEARCH_ENDPOINT", "https://test-search-service.search.windows.net")
    monkeypatch.setenv("AZURE_SEARCH_INDEX", "test-index")

    monkeypatch.setattr("azure.identity.aio.ManagedIdentityCredential", mock_cred.MockAzureCredential)
    monkeypatch.setattr("azure.identity.aio.AzureDeveloperCliCredential", mock_cred.MockAzureCredential)

    quart_app = quartapp.create_app(testing=True)

    async with quart_app.test_app():
        openai_client = quart_app.blueprints["chat"].openai_client
        assert isinstance(openai_client, AsyncOpenAI)
        assert str(openai_client.base_url).endswith("/openai/v1/")
        assert isinstance(quart_app.blueprints["chat"].search_client, SearchClient)
