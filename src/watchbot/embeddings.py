"""OpenAI embedding calls. This is the only module that touches the OpenAI API."""

from __future__ import annotations

from openai import OpenAI

from watchbot.config import EMBEDDING_MODEL

_BATCH_SIZE = 256


def embed_texts(texts: list[str], client: OpenAI | None = None) -> list[list[float]]:
    """Embed a batch of texts, preserving order."""
    client = client or OpenAI()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[start : start + _BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return vectors


def embed_query(text: str, client: OpenAI | None = None) -> list[float]:
    return embed_texts([text], client=client)[0]
