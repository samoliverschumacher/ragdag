from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import FieldCondition, Filter, MatchValue, ScoredPoint
from sentence_transformers import SentenceTransformer


class QDRANTArgumentsError(Exception):
    """When arguments passed to search as config are in the input arguments"""

    pass


def get_client(params: dict) -> QdrantClient:
    """params expects '`host`' and '`port`' keys"""
    client = QdrantClient(params["host"], port=params["port"])
    return client


def encode_text(encoder: SentenceTransformer, text: str) -> list[float]:
    """
    A function that encodes the input text using a SentenceTransformer and returns a
    list of floats representing the vectorized text and an integer.

    Args:
    - encoder: The SentenceTransformer object used to encode the text.
    - text: The input text to be encoded.

    Returns:
    - A tuple containing a list of floats representing the encoded text.
    """
    vector = encoder.encode(text)
    return vector.tolist()  # type: ignore


def get_encoder(config: dict) -> SentenceTransformer:
    return SentenceTransformer(config["encoder"])


def search_collection(
    client: QdrantClient,
    query_vector: list[float],
    limit: int,
    collection_name: str,
    search_kwargs: dict[str, Any] | None = None,
    search_filters: dict | None = None,
) -> list[ScoredPoint]:
    """
    Query a collection using a query vector and optional filters.

    Args:
        query_vector (List[float]): The query vector.
        limit (int): The maximum number of results to return.
        collection_name (str): The name of the collection to search.
        search_kwargs (Optional[Dict[str, Any]]): Additional search parameters.
        search_filters (Optional[Dict[str, List[Dict[str, Dict[str, Any]]]]]): Filters to apply to the search.

    Returns:
        List[Dict[str, Any]]: The search results.

    Raises:
        QDRANTArgumentsError: If conflicting keys are found in `search_kwargs`.
    """

    if search_kwargs is None:
        search_kwargs = {}

    conflicting_keys = set(search_kwargs.keys()) & {"query_vector", "limit", "collection_name"}
    if conflicting_keys:
        raise QDRANTArgumentsError(f"Conflicting keys found in search_kwargs: {', '.join(conflicting_keys)}")

    if not search_filters:
        search_result = client.search(
            collection_name=collection_name, query_vector=query_vector, limit=limit, **search_kwargs
        )
    else:
        must_filters = search_filters.get("must", [])
        should_filters = search_filters.get("should", [])

        must_conditions = [
            FieldCondition(key=f["key"], match=MatchValue(value=f["match"]["value"])) for f in must_filters
        ]
        should_conditions = [
            FieldCondition(key=f["key"], match=MatchValue(value=f["match"]["value"])) for f in should_filters
        ]

        query_filter = Filter(must=must_conditions, should=should_conditions)  # type: ignore

        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            with_payload=True,
            limit=limit,
            **search_kwargs,
        )
    return search_result


def embed_create_collection(client: QdrantClient, data: list[dict], collection_name: str, encoder) -> None:
    """
    Generate a new collection in Qdrant with the provided data and collection name.

    Parameters:
        client (QdrantClient): The Qdrant client to interact with.
        data (list[dict]): List of dictionaries containing data for the collection.
        collection_name (str): Name of the collection to be created.
        encoder: The encoder object used to encode text into vectors.

    Returns:
        None
    """

    size = encoder.get_sentence_embedding_dimension()

    print(f"\nembedding size = {size=}\n")
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=size,
            distance=models.Distance.DOT,
        ),
    )

    points = []
    for idx, item in enumerate(data):
        vector = encoder.encode(item["text"]).tolist()
        payload = {k: v for k, v in item.items()}
        points.append(models.PointStruct(id=idx, vector=vector, payload=payload))

    client.upload_points(
        collection_name=collection_name,
        points=points,
    )
