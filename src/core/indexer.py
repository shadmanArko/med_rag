from __future__ import annotations

from typing import Any

from src.core.base import ProjectConfig
from src.core.settings import AppSettings


def build_qdrant_client(settings: AppSettings):
    from qdrant_client import QdrantClient

    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def _build_embed_model(settings: AppSettings):
    from llama_index.embeddings.fastembed import FastEmbedEmbedding

    return FastEmbedEmbedding(
        model_name=settings.embedding_model,
        embed_batch_size=settings.embedding_batch_size,
    )


def _build_vector_store(config: ProjectConfig, settings: AppSettings, client=None):
    from llama_index.vector_stores.qdrant import QdrantVectorStore

    return QdrantVectorStore(
        client=client or build_qdrant_client(settings),
        collection_name=config.collection_name,
        enable_hybrid=settings.query_mode == "hybrid",
    )


def collection_exists(config: ProjectConfig, settings: AppSettings) -> bool:
    client = build_qdrant_client(settings)
    try:
        return bool(client.collection_exists(config.collection_name))
    finally:
        client.close()


def build_index(documents: list[Any], config: ProjectConfig, settings: AppSettings):
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter

    # Rebuild from scratch so repeated reindexing doesn't duplicate chunks.
    client = build_qdrant_client(settings)
    if client.collection_exists(config.collection_name):
        client.delete_collection(config.collection_name)
    vector_store = _build_vector_store(config, settings, client=client)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    splitter = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=_build_embed_model(settings),
        transformations=[splitter],
        show_progress=True,
    )


def load_index(config: ProjectConfig, settings: AppSettings):
    from llama_index.core import VectorStoreIndex

    return VectorStoreIndex.from_vector_store(
        vector_store=_build_vector_store(config, settings),
        embed_model=_build_embed_model(settings),
    )
