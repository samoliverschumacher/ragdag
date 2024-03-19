from typing import Any

import app.logconfig
from app.pipe import ErrorStack, Process, RAGStage

logger = app.logconfig.setup_logger('root')


class Rewriter(Process):

    stage = RAGStage.REWRITE

    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def _process(self, text: Any, data: dict, errors: ErrorStack,
                 sentinel: RAGStage) -> tuple[Any, dict, ErrorStack, RAGStage]:
        return text, data, errors, sentinel.increment()
