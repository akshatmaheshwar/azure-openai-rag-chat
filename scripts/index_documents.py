"""Create the Azure AI Search index and upload the documents in docs/policies.

Reads AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_INDEX from the environment (or a .env file),
and authenticates with DefaultAzureCredential (for example, after `azd auth login`).
No API keys are used.
"""

import logging
import os
import re
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchableField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
)
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs" / "policies"

logger = logging.getLogger("index_documents")


def build_index(index_name: str) -> SearchIndex:
    return SearchIndex(
        name=index_name,
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="title", type=SearchFieldDataType.String),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
        ],
    )


def load_documents(docs_dir: Path) -> list[dict]:
    documents = []
    for path in sorted(docs_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        # Use the first Markdown heading as the title, falling back to the file name
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        documents.append(
            {
                # Document keys may only contain letters, digits, underscores, dashes and equal signs
                "id": re.sub(r"[^A-Za-z0-9_\-=]", "_", path.stem),
                "title": title_match.group(1).strip() if title_match else path.stem,
                "content": content,
                "source": path.relative_to(REPO_ROOT).as_posix(),
            }
        )
    return documents


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv(override=True)

    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    index_name = os.getenv("AZURE_SEARCH_INDEX")
    if not endpoint or not index_name:
        raise ValueError("AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_INDEX are required")

    documents = load_documents(DOCS_DIR)
    if not documents:
        raise ValueError(f"No .md documents found in {DOCS_DIR}")

    credential = DefaultAzureCredential()

    with SearchIndexClient(endpoint=endpoint, credential=credential) as index_client:
        index_client.create_or_update_index(build_index(index_name))
        logger.info("Created or updated index '%s'", index_name)

    with SearchClient(endpoint=endpoint, index_name=index_name, credential=credential) as search_client:
        results = search_client.upload_documents(documents=documents)

    failed = [result.key for result in results if not result.succeeded]
    if failed:
        raise RuntimeError(f"Failed to upload documents: {', '.join(failed)}")
    logger.info("Uploaded %d documents to '%s'", len(documents), index_name)


if __name__ == "__main__":
    main()
