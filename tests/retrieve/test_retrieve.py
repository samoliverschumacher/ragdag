import json
import time

import pytest
from qdrant_client import QdrantClient
from stubs import create_collection, qdrant_retriever_config

from app import ROOT_DIR
from app.pipe import RAGStage
from app.retrieve.retrieve import QDRANTRetriever, QueryEmbeddingTimeoutError
from app.schemas import GenerateRequest
from app.store.qdrant import QDRANTArgumentsError, search_collection


@pytest.fixture
def mock_embed(monkeypatch):
    """Mock the embed method."""
    def mock_embed(encoder, text):
        return {'encoding': [1.0, 2.0, 3.0], 'tokens': 3}
    monkeypatch.setattr('app.retrieve.retrieve.encode_text', mock_embed)


@pytest.fixture
def mock_retrieve_vectors(monkeypatch):
    """Mock the query function."""
    def mock_retrieve_vectors(*args, **kwargs):
        return (['friends', 'family', '42'],
                    [0.9, 0.8, 0.7],
                    ['id1', 'id2', 'id3'],
                    [{},{},{}])

    monkeypatch.setattr('app.retrieve.retrieve.QDRANTRetriever.retrieve_from_vector', mock_retrieve_vectors)


@pytest.fixture
def mock_retrieve_vectors_filtered(monkeypatch):
    """Mock the query function."""
    def mock_retrieve_vectors(*args, **kwargs):
        return {
            'documents': ['friends', 'family', '42'],
            'scores': [0.9, 0.8, 0.7],
            'doc_ids': ['id1', 'id2', 'id3'],
            'metadata': None
        }
    monkeypatch.setattr('app.retrieve.retrieve.QDRANTRetriever.retrieve_from_vector', mock_retrieve_vectors)


@pytest.fixture
def mock_embed_slow(monkeypatch):
    """Mock the embed method."""
    def mock_embed(encoder, text):
        time.sleep(2.0)

    monkeypatch.setattr('app.retrieve.retrieve.encode_text', mock_embed)


@pytest.fixture(scope="class")
def qdrant_client():
    """Initialize QDRANT client and populate a collection."""
    print("Initializing QDRANT client and populating collection...")
    # Pytest mark qdrant would be used here to indicate that the service was started
    client = QdrantClient("localhost", port=6333)

    test_data_f = ROOT_DIR.parent / "tests" / "test_data.json"
    with open(test_data_f) as file:
        data = json.load(file)
    collection_name = "example_collection"

    create_collection(data, collection_name)

    yield client
    client.delete_collection(collection_name)
    print("Teardown complete.")


class TestSearch:

    @pytest.mark.qdrant
    def test_search_collection_qdrant_basic(self, qdrant_client):
        """Searches the QDRANT service for similar documents."""
        client = qdrant_client
        query_vector = [0.2, 0.1, 0.9, 0.7]
        limit = 2
        collection_name = "example_collection"
        search_kwargs = {}
        search_filters = None

        search_result_2 = search_collection(client, query_vector, limit, collection_name, search_kwargs, search_filters)
        limit = 4
        search_result_4 = search_collection(client, query_vector, limit, collection_name, search_kwargs, search_filters)

        assert len(search_result_2) == 2 and len(search_result_4) == 4
        assert sum(r.score for r in search_result_2)/2 >= sum(r.score for r in search_result_4)/4

    @pytest.mark.qdrant
    def test_search_collection_qdrant_with_filters(self, qdrant_client):
        """Searches the QDRANT service for similar documents, conditional on filters.
        
        Cases:
            - Filter on topics
            - AND filters.
            - OR filters
        """
        client = qdrant_client
        query_vector = [0.2, 0.1, 0.9, 0.7]
        limit = 10
        collection_name = "example_collection"
        search_kwargs = {}
        search_filters = {
            "must": [{"key": "topic", "match": {"value": "Payroll"}}],
            "should": [{"key": "chunk", "match": {"value": 1}},
                       {"key": "chunk", "match": {"value": 2}}]
        }

        search_result = search_collection(client, query_vector, limit, collection_name, search_kwargs, search_filters)

        assert all([r.payload['topic'] == 'Payroll' for r in search_result])
        assert all([r.payload['chunk'] in [1,2] for r in search_result])

    @pytest.mark.qdrant
    def test_search_collection_qdrant_conflicting_keys(self, qdrant_client):
        """Raises error when configuration is invalid."""
        client = qdrant_client
        query_vector = [0.2, 0.1, 0.9, 0.7]
        limit = 10
        collection_name = "example_collection"
        search_kwargs = {"query_vector": "invalid_value"}
        search_filters = None

        with pytest.raises(QDRANTArgumentsError):
            search_collection(client, query_vector, limit, collection_name, search_kwargs, search_filters)


class TestQDRANTRetriever:

    @classmethod
    def setup_class(cls):
        cls.config = qdrant_retriever_config()
        cls.retriever = QDRANTRetriever(cls.config)

    @pytest.mark.transformers
    def test_process(self, mock_embed_slow, mock_retrieve_vectors):
        """Process propgates to a END stage. If error, helpful message propogates"""

        query = "What is the meaning of life?"

        data = dict(event_id='1234')
        sentinel = RAGStage.RETRIEVE
        errors = []
        text = GenerateRequest(user_query=query)

        self.retriever.config['embedding_timeout'] = 0.1  # Limit too low for embedding process
        response = self.retriever(text, data, errors, sentinel)

        # Error captured in last position in error list
        accum_errors = response[2]
        recent_error = accum_errors.pop()
        assert recent_error[0] == RAGStage.RETRIEVE
        assert type(recent_error[1]) == QueryEmbeddingTimeoutError
        assert ('Sorry, something went wrong. The website'
                'is an excellent resource for many related questions.') in response[0][0].text

    @pytest.mark.transformers
    def test_retrieve(self, mock_embed, mock_retrieve_vectors):
        """Generates a list of documents given a user question.
        
        Inputs: 
            - Query: User query. Has no security risk information in it.
            - Event metadata.
            - Configuration for the retrieval.
            
        Outputs:
            - Documents, and doc_ids that match the query.
            - Optionally tracked metrics: query embedding vector, embedding token count, document_ids, document similarity scores, latency.
            
        Set-up:
            - Embedding model. (has methods to `.embed()` text to a vector.)
            - Storage system for documents. (has methods to `.query()` the storage for similar vectors)
        """
        query = "What is the meaning of life?"
        expected_docs = ['friends', 'family', '42']
        expected_ids = ['id1', 'id2', 'id3']

        user_request = GenerateRequest(user_query=query)
        documents = self.retriever.simple_retrieve(user_request, event_id = '')

        actual_documents = [doc.text for doc in documents]
        actual_document_ids = [doc.doc_id for doc in documents]
        assert expected_docs == actual_documents
        assert expected_ids == actual_document_ids

    @pytest.mark.transformers
    def test_retrieve_timeout(self, mock_embed_slow, mock_retrieve_vectors):
        """When embedding model doesnt respond in time (>0.1 s), raise timeout error.
        
        Set-up:
            - mocked slow embedding process (0.4s)
        """

        query = "What is the meaning of life?"
        user_request = GenerateRequest(user_query=query)
        self.config['embedding_timeout'] = 0.1

        with pytest.raises(QueryEmbeddingTimeoutError):
            _ = self.retriever.simple_retrieve(user_request, event_id = 'abc-2024:03:14')
