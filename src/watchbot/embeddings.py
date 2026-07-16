"""OpenAI embedding calls. This is the only module that touches the OpenAI API."""

from __future__ import annotations

from collections.abc import Iterator
from functools import cache

from openai import OpenAI

from watchbot.config import EMBEDDING_MODEL

# The embeddings endpoint caps total tokens per request; splitting batches by a
# conservative character budget (~4 chars/token) keeps every request well under it.
_MAX_CHARS_PER_REQUEST = 400_000


@cache
def _shared_client() -> OpenAI:
    return OpenAI()


def _char_budget_batches(texts: list[str]) -> Iterator[list[str]]:
    batch: list[str] = []
    chars = 0
    for text in texts:
        if batch and chars + len(text) > _MAX_CHARS_PER_REQUEST:
            yield batch
            batch, chars = [], 0
        batch.append(text)
        chars += len(text)
    if batch:
        yield batch


def embed_texts(texts: list[str], client: OpenAI | None = None) -> list[list[float]]:
    """Embed a batch of texts, preserving order."""
    client = client or _shared_client()
    vectors: list[list[float]] = []
    for batch in _char_budget_batches(texts):
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return vectors


def embed_query(text: str, client: OpenAI | None = None) -> list[float]:
    return embed_texts([text], client=client)[0]
