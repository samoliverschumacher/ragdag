from typing import Any

from app.pipe import ErrorStack, Process, RAGStage


class SVMReranker(Process):
    """Uses a support vector machine to predict relevance scores."""

    stage = RAGStage.RERANK

    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def _SVM_infer(self, embeddings: list[list[float]]) -> list[float]:
        raise NotImplementedError

    def _process(self, text: Any, data: dict, errors: ErrorStack, sentinel) -> tuple[Any, dict, ErrorStack, RAGStage]:
        return text, data, errors, sentinel.increment()
