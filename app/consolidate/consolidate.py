from collections.abc import Callable
from typing import Any

import app.logconfig
from app.pipe import ErrorStack, Process, RAGStage
from app.retrieve.retrieve import Context, ContextWithMetadata

logger = app.logconfig.setup_logger("root")


class AllDocumentsExceedContextWindowError(Exception):
    """None of the retrieved documents fit in the context window for the LLM."""

    pass


class SimpleConsolidator(Process):

    stage = RAGStage.CONSOLIDATE

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.config = config
        self._text_to_tokens: Callable = config.get("text_to_tokens_func", lambda text: [c for c in text])

    def simple_consolidate(
        self, contexts: list[ContextWithMetadata | Context], config: dict
    ) -> list[Context] | list[ContextWithMetadata]:
        token_limit = config["token_limit"]
        strategy = config["strategy"]

        def _format(c: ContextWithMetadata | Context) -> str:
            if isinstance(c, ContextWithMetadata):
                template = "[title: '{title}', url: '{url}']\n{text}\n"
                return template.format(text=c.text, title=c.title, url=c.url)
            template = "{text}\n"
            return template.format(text=str(c.text))

        # limit contexts based on a strategy, and a token limit
        if strategy == "score_weighted":
            contexts = sorted(contexts, key=lambda x: x.score, reverse=True)
        elif strategy == "simple":
            contexts = sorted(contexts, key=lambda x: len(_format(x)))
        else:
            contexts = contexts

        new_contexts, token_cost_total = [], 0
        for idx, context in enumerate(contexts):
            text = str(_format(context))
            token_length = len(self._text_to_tokens(text))

            # Check if adding the current context exceeds the token limit
            if token_cost_total + token_length > token_limit:
                # Check if any context exceeds the token limit
                if idx < len(contexts) - 1:
                    doc_lengths = [len(self._text_to_tokens(str(_format(c)))) for c in contexts]
                    doc_ids = [c.doc_id for c in contexts]
                    docs_as_text = ",".join(
                        [f"({i=} {length=})" for i, length in zip(doc_ids, doc_lengths, strict=False)]
                    )
                    msg = f"{token_limit=} Documents: {docs_as_text}"
                    raise AllDocumentsExceedContextWindowError(msg)

                break

            context.text = text
            new_contexts.append(context)
            token_cost_total += token_length

        return new_contexts

    def _process(self, text: Any, data: dict, errors: ErrorStack, *_) -> tuple[Any, dict, ErrorStack, RAGStage]:
        context_items: list[Context] | list[ContextWithMetadata] = text

        try:
            consolidated_contexts = self.simple_consolidate(context_items, self.config)
            next_text: list[Context] | list[ContextWithMetadata] = consolidated_contexts
            next_data = {**data, "contexts": consolidated_contexts}
        except Exception as e:
            errors.append((self.stage, e))
            logger.error(e)
            next_text = text
            next_data = {**data, "contexts": context_items}

        return next_text, next_data, errors, self.next_stage
