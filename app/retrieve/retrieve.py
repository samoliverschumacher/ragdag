import time
from typing import Any

from pydantic import BaseModel

import app.logconfig
from app.pipe import ErrorStack, Process, RAGStage
from app.schemas import GenerateRequest
from app.store.qdrant import encode_text, get_client, get_encoder, search_collection
from app.utils import run_until_timeout

logger = app.logconfig.setup_logger("root")


class Context(BaseModel):
    """doc_id: str
    text: str"""

    doc_id: str
    text: str
    score: float | None = None


class ContextWithMetadata(Context):
    url: str
    title: str


class QueryEmbeddingTimeoutError(Exception):
    """Raised when query embedding takes longer than the specified timeout."""

    pass


class NoDcoumentsRetrievedError(Exception):
    pass


class QDRANTRetriever(Process):
    """User Query -> VectorDB -> (Documents, vectors)

    Behavioiurs:
        - Retrieves documents from vector store, and lables them with any metadata.
        - If no documents are retrieved, returns a helpful message fallback response as context.

    Handles Exceptions:
        - QueryEmbeddingTimeoutError: Returns this reason as the context document.
        - NoDcoumentsRetrievedError: Returns this reason as the context document.
    """

    stage = RAGStage.RETRIEVE

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.config = config
        self.client = get_client(self.config)
        self.encoder = get_encoder(self.config)

    def simple_retrieve(self, request: GenerateRequest, event_id: str) -> list[Context]:
        """
        A function that retrieves information based on a user query.

        Args:
            request (GenerateRequest): The request object containing the user query.
            event_id (str): The unique identifier for the event.

        Returns:
            List[Context]: A list of contexts containing the retrieved information.
        """

        logger.info("Retrieving...")
        t0 = time.time()
        embedding = run_until_timeout(
            encode_text, self.config["embedding_timeout"], QueryEmbeddingTimeoutError, self.encoder, request.user_query
        )
        t1 = time.time()

        logger.eval(event_id, {"metric": "embed_seconds", "value": t1 - t0})
        logger.eval(event_id, {"metric": "user_query_embedding", "value": ",".join(map(str, embedding))})

        if self.config["filter_on_user_metadata"]:
            if hasattr(request, "metadata") and request.metadata:
                self.config["search_filters"] = self._create_must_filter(request.metadata)

        t0 = time.time()
        documents, scores, doc_ids, metadatas = self.retrieve_from_vector(embedding, n=self.config["top_k"])
        t1 = time.time()
        logger.eval(event_id, {"metric": "retrieve_seconds", "value": t1 - t0})
        logger.eval(
            event_id,
            {
                "metric": "relevant_document_ids_scores",
                "value": ",".join(":".join(str(x) for x in tup) for tup in zip(doc_ids, scores, strict=False)),
            },
        )
        if not documents:
            raise NoDcoumentsRetrievedError("No relevant documents found")

        contexts = []
        for doc, id, meta in zip(documents, doc_ids, metadatas, strict=False):
            if len(meta) > 0:
                contexts.append(
                    ContextWithMetadata(doc_id=str(id), text=doc, url=meta.get("url", ""), title=meta["title"])
                )
            else:
                contexts.append(Context(doc_id=str(id), text=doc))
        return contexts

    def retrieve_from_vector(
        self, query_vector: list[float], n: int
    ) -> tuple[list[str], list[float], list[int], list[dict]]:
        """
        Retrieves documents, scores, document ids, and metadata from a vector search.

        Uses search filters and parameters from the instantiated classes config.

        Args:
            query_vector (List[float]): Vector representing the query for search.
            n (int): Number of results to retrieve.

        Returns:
            Tuple: A tuple containing lists of documents, scores, document ids, and metadata.
        """

        results = search_collection(
            self.client,
            query_vector,
            limit=n,
            collection_name=self.config["collection_name"],
            search_kwargs=self.config.get("search_params", {}),
            search_filters=self.config.get("search_filters", {}),
        )

        payloads = [d.payload for d in results]
        scores = [d.score for d in results]
        doc_ids = [d["doc_id"] for d in payloads]
        documents = [d["text"] for d in payloads]

        metadata = [{k: v for k, v in payload.items() if k not in {"scores", "doc_id", "text"}} for payload in payloads]

        return documents, scores, doc_ids, metadata

    def _process(self, text: Any, data: dict, errors: ErrorStack, *_) -> tuple[Any, dict, ErrorStack, RAGStage]:
        request: GenerateRequest = text
        data["user_query"] = request  # For use by later processes
        event_id: str = data["event_id"]

        next_sentinel = self.next_stage
        try:
            contexts = self.simple_retrieve(request, event_id)
            next_text: list[Context | ContextWithMetadata] = contexts

        except (NoDcoumentsRetrievedError, QueryEmbeddingTimeoutError) as e:
            if type(e) == QueryEmbeddingTimeoutError:
                logger.error("Query embedding took too long to respond")
                msg = "Sorry, something went wrong. "
            else:
                msg = "No relevant documents found. Perhaps reframe your query. "
            logger.error(str(e) + "Zero documents retrieved.")
            errors.append((self.stage, e))
            next_text = self._fallback_context(msg)
            next_sentinel = RAGStage.GENERATE

        except Exception as e:
            # Give the user the retrieved references verbatim.
            errors.append((self.stage, e))
            logger.error(str(e) + "Applied a no-VectorStore fallback document.")
            next_text = self._fallback_context("Sorry something went wrong. ")

        return next_text, data, errors, next_sentinel

    @staticmethod
    def _create_must_filter(user_metadata: dict) -> dict:
        return {"must": [{"key": k, "match": {"value": v}} for k, v in user_metadata.items()]}

    def _fallback_context(self, msg: str = "") -> list[Context]:
        return [Context(doc_id="", text=f"{msg}The website" "is an excellent resource for many related questions.")]
