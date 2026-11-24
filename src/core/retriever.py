from __future__ import annotations

from src.core.base import ProjectConfig
from src.core.settings import AppSettings


def _build_llm(config: ProjectConfig, settings: AppSettings):
    from llama_index.core.base.llms.types import LLMMetadata
    from llama_index.llms.anthropic import Anthropic

    class ClaudeLLM(Anthropic):
        # llama-index keeps a hard-coded model whitelist; bypass it so newer Claude
        # model IDs work without waiting for a library release.
        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(
                context_window=200_000,
                num_output=self.max_tokens,
                is_chat_model=True,
                is_function_calling_model=True,
                model_name=self.model,
            )

    return ClaudeLLM(
        model=settings.anthropic_model,
        temperature=0.1,
        max_tokens=1024,
        system_prompt=config.system_prompt,
    )


def build_query_engine(index, config: ProjectConfig, settings: AppSettings):
    llm = _build_llm(config, settings)
    query_kwargs = {
        "llm": llm,
        "similarity_top_k": settings.similarity_top_k,
        "response_mode": "compact",
    }
    if settings.query_mode == "hybrid":
        query_kwargs.update(
            {
                "vector_store_query_mode": "hybrid",
                "sparse_top_k": settings.sparse_top_k,
                "alpha": settings.hybrid_alpha,
            }
        )
    return index.as_query_engine(**query_kwargs)
