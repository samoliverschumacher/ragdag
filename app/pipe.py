import enum
from abc import ABC, abstractmethod
from typing import Any


class RAGStage(enum.Enum):
    """SECURITY -> RCACHE -> REWRITE -> RETRIEVE -> RERANK -> CONSOLIDATE -> GENERATE -> WCACHE"""
    START = -1
    SECURITY = 0
    RCACHE = 1  # Read cache for semantically similar questions
    REWRITE = 2
    RETRIEVE = 3
    RERANK = 4
    CONSOLIDATE = 5
    GENERATE = 6
    WCACHE = 7  # Write to Cache, where data from any prior stage could be cached asynchronously.

    END = None

    def increment(self):
        if self.value is None:
            return RAGStage.END
        new = self.value + 1
        if new not in [e.value for e in RAGStage]:
            return RAGStage.END
        return RAGStage(new)


ErrorStack = list[tuple[RAGStage, Exception]]


class Process(ABC):
    """Process in a non-branching DAG pipeline.

    Each process takes 4 args, and returns 4 args of the same name.
         - `text`: The text to process. Could be any value, generally the main output of the process.
         - `data`: Data is a dictionary of values that acts as a container for data required by future processes.
         - `errors`: An accumulating list of errors that occurs throughout the processes.
                   Each item contains a RAGStage identifier for which process the error occured in.
         - `sentinel`: A flag indicating a stage. It's primary use is to indicate to the following
                       process if that process should be skipped.

    Behaviours supported:
        - `__call__`: Processes data, as per defined in the `_process` method.
        - `stage`: returns the current stage
        - `next_stage`: returns the next stage in the pipeline
        - `_process`: Processes data in a given stage. Respoonsible for:
                        - Handling raised errors,
                        - indicating what process happens next (or skipped),
                        - Calling the main process function.
    """

    stage: RAGStage
    _next_stage: RAGStage

    @property
    def next_stage(self) -> RAGStage:
        return self._next_stage

    @abstractmethod
    def _process(self, text: Any, data: dict, errors: ErrorStack,
                 sentinel: RAGStage) -> tuple[Any, dict, ErrorStack, RAGStage]:
        """
         - Process the task
         - Return its errors, if not raised.
         - Add to data, forward on data from previous tasks, or remove it.
         - return text, data, errors, sentinel
         - Must set the sentinel to a future stage
        """
        ...

    def __init__(self, config: dict) -> None:
        self._next_stage = self.stage.increment()
        self.config = config

    def __call__(self, text: Any, data: dict, errors: ErrorStack,
                 sentinel: RAGStage) -> tuple[Any, dict, ErrorStack, RAGStage]:
        """Calls the _process method if the sentinel is the current stage
        (Allows skipping stages, for example in a successful cache hit.)

        No need to override this method in subclasses.
        """
        if not sentinel == self.stage:
            # pass through
            next_sentinel = sentinel
            return text, data, errors, next_sentinel

        next_text, next_data, next_errors, next_sentinel = self._process(
            text, data, errors, sentinel)
        return next_text, next_data, next_errors, next_sentinel


def create_links(dag: list[Process]) -> None:
    """Connects the objects in the DAG, setting the `_next_stage` 
    property to the `_stage` of the object that follows it in the list.
    """
    for i in range(len(dag) - 1):
        dag[i]._next_stage = dag[i + 1].stage  # ignore: SLF001
    dag[-1]._next_stage = RAGStage.END  # ignore: SLF001
