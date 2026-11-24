from __future__ import annotations


def _named(metric: object, name: str) -> object:
    setattr(metric, "name", name)
    return metric


def build_rag_metrics():
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
    )

    from eval.shared.judge import ClaudeJudge

    judge = ClaudeJudge()
    return [
        _named(AnswerRelevancyMetric(threshold=0.7, model=judge), "Answer Relevance"),
        _named(ContextualRelevancyMetric(threshold=0.7, model=judge), "Context Relevance"),
        _named(FaithfulnessMetric(threshold=0.7, model=judge), "Groundedness"),
    ]