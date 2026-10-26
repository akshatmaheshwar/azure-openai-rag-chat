import json
import os

from azure.identity.aio import (
    AzureDeveloperCliCredential,
    ChainedTokenCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
from azure.search.documents.aio import SearchClient
from openai import AsyncOpenAI
from quart import (
    Blueprint,
    Response,
    current_app,
    render_template,
    request,
    stream_with_context,
)

bp = Blueprint("chat", __name__, template_folder="templates", static_folder="static")

# Number of search results to include as sources in the prompt
SEARCH_TOP = 3
NO_RESULTS_MESSAGE = "I couldn't find that in the documents."
GROUNDING_INSTRUCTIONS = (
    "Answer the user's question using ONLY the sources below. "
    "Name the sources you used by their title in square brackets, for example [Leave Policy]. "
    f'If the sources do not contain the answer, reply exactly "{NO_RESULTS_MESSAGE}" and nothing else.'
)


@bp.before_app_serving
async def configure_openai():
    # Use ManagedIdentityCredential with the client_id for user-assigned managed identities
    user_assigned_managed_identity_credential = ManagedIdentityCredential(client_id=os.getenv("AZURE_CLIENT_ID"))

    # Use AzureDeveloperCliCredential with the current tenant.
    azure_dev_cli_credential = AzureDeveloperCliCredential(tenant_id=os.getenv("AZURE_TENANT_ID"), process_timeout=60)

    # Create a ChainedTokenCredential with ManagedIdentityCredential and AzureDeveloperCliCredential
    #  - ManagedIdentityCredential is used for deployment on Azure Container Apps

    #  - AzureDeveloperCliCredential is used for local development
    # The order of the credentials is important, as the first valid token is used
    # For more information check out:

    # https://learn.microsoft.com/azure/developer/python/sdk/authentication/credential-chains?tabs=ctc#chainedtokencredential-overview
    azure_credential = ChainedTokenCredential(user_assigned_managed_identity_credential, azure_dev_cli_credential)
    current_app.logger.info("Using Azure OpenAI Responses API with credential")

    # Get the token provider for Azure OpenAI based on the selected Azure credential
    token_provider = get_bearer_token_provider(azure_credential, "https://cognitiveservices.azure.com/.default")
    openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not openai_endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT is required for Azure OpenAI")
    if not os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"):
        raise ValueError("AZURE_OPENAI_CHAT_DEPLOYMENT is required for Azure OpenAI")
    search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    search_index = os.getenv("AZURE_SEARCH_INDEX")
    if not search_endpoint or not search_index:
        raise ValueError("AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_INDEX are required for Azure AI Search")

    # Create the asynchronous OpenAI client against Azure's v1 endpoint.
    bp.azure_credential = azure_credential
    bp.openai_client = AsyncOpenAI(
        base_url=f"{openai_endpoint.rstrip('/')}/openai/v1/",
        api_key=token_provider,
    )
    # Set the model name to the Azure OpenAI model deployment name
    bp.openai_model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

    # Create the asynchronous Azure AI Search client with the same credential chain (no API keys)
    bp.search_client = SearchClient(endpoint=search_endpoint, index_name=search_index, credential=azure_credential)


@bp.after_app_serving
async def shutdown_openai():
    await bp.openai_client.close()
    await bp.search_client.close()
    await bp.azure_credential.close()


def get_latest_user_text(request_input: list[dict]) -> str:
    for message in reversed(request_input):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            return " ".join(part.get("text", "") for part in content if part.get("type") == "input_text")
    return ""


async def search_documents(query: str) -> list[dict]:
    if not query.strip():
        return []
    results = await bp.search_client.search(
        search_text=query, top=SEARCH_TOP, select=["id", "title", "content", "source"]
    )
    return [document async for document in results]


def build_instructions(documents: list[dict]) -> str:
    sources = "\n\n".join(
        f"[{document['title']}] (source: {document['source']})\n{document['content']}" for document in documents
    )
    return f"{GROUNDING_INSTRUCTIONS}\n\nSources:\n\n{sources}"


@bp.get("/")
async def index():
    return await render_template("index.html")


@bp.post("/chat/stream")
async def chat_handler():
    request_input = (await request.get_json())["input"]

    @stream_with_context
    async def response_stream():
        try:
            documents = await search_documents(get_latest_user_text(request_input))
            sources = [{"title": document["title"], "source": document["source"]} for document in documents]
            yield json.dumps({"type": "sources", "sources": sources}, ensure_ascii=False) + "\n"
            if not documents:
                # Don't call the model when there is nothing to ground the answer in
                yield json.dumps({"type": "response.output_text.delta", "delta": NO_RESULTS_MESSAGE}) + "\n"
                return

            async with bp.openai_client.responses.stream(
                # Azure OpenAI takes the deployment name as the model name.
                model=bp.openai_model,
                instructions=build_instructions(documents),
                input=request_input,
                store=False,
            ) as openai_stream:
                async for event in openai_stream:
                    yield json.dumps(event.model_dump(), ensure_ascii=False) + "\n"
        except Exception as e:
            current_app.logger.exception("Responses stream failed")
            yield json.dumps({"error": str(e)}, ensure_ascii=False) + "\n"

    return Response(response_stream(), mimetype="application/x-ndjson")
