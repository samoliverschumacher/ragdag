from functools import partial

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct

from app.config import QDRANT_URL, REDACTED_INFORMATION_TOKEN_MAP

client = QdrantClient("localhost", port=6333)


def qdrant_retriever_config():
    return dict(url=QDRANT_URL,
            host = 'localhost',
            port=6333,
            embedding_timeout=0.1,
            top_k=3,
            encoder='all-MiniLM-L6-v2',
            filter_on_user_metadata=True)

def gpt2_generation_config():
    model_cfg = {}
    model_cfg['model'] = 'gpt2'
    model_cfg['max_length'] = 100
    model_cfg['num_return_sequences'] = 1
    model_cfg['no_repeat_ngram_size'] = 3
    model_cfg['device'] = 'cpu'
    return dict(redacted_information_token_map = REDACTED_INFORMATION_TOKEN_MAP,
                        model_config = model_cfg,
                        retries=3,
                        no_llm_fallback_strategy = 'documents',
                        timeout=5,
                        prompt_template = 'basic',
                        seed = 42)

def create_collection_from_text(client: QdrantClient, data: list[dict], collection_name: str) -> None:
    """
    Create a new collection from the provided data using the Qdrant client.

    Args:
    - client: QdrantClient - The Qdrant client to use for creating the collection.
    - data: list[dict] - The list of dictionaries containing the data points to be uploaded.
    - collection_name: str - The name of the collection to be created.
    """
    size = 4
    print(f"\nembedding size = {size=}\n")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=size,  # Vector size is defined by used model
            distance=models.Distance.DOT,
        ),
    )

    points = []
    for idx, item in enumerate(data):
        vector = [0.1,.43,.5,.6]
        payload = {k:v for k,v in item.items() if k != "text"}
        points.append(PointStruct(id=idx, vector=vector, payload=payload))

    client.upload_points(
        collection_name=collection_name,
        points=points,
    )

def create_collection(data: list[dict], collection_name: str) -> None:
    """
    data: List of dict with keys 'id' (int), 'vector' (List[float]), 'payload' (dict)
    """
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=4, distance=models.Distance.DOT),
    )

    points = []
    for item in data:
        points.append(PointStruct(id=item['id'], vector=item['vector'], payload={k:v for k,v in item['payload'].items() if k != 'text'}))

    operation_info = client.upsert(
        collection_name=collection_name,
        wait=True,
        points=points,
    )
    print(operation_info)


def char_to_int_encoder(text: str, size: int) -> list[int]:
    assert all([c.isdigit() for c in text[:size]]), "simple encoder needs numbers at first size chars"
    return [int(c) for c in text[:size]]



def embed_create_collection(client: QdrantClient, data: list[dict], collection_name: str, embed_size: int = None, encoder = None) -> None:
    """
    data: List of dict with at least key = 'text' (str)
    """
    if encoder is None:
        encoder = partial(char_to_int_encoder, size=embed_size)
        size = embed_size
    else:
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
        payload = {k:v for k,v in item.items()}
        points.append(PointStruct(id=idx, vector=vector, payload=payload))

    client.upload_points(
        collection_name=collection_name,
        points=points,
    )
