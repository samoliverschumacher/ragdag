from unittest.mock import patch

import pytest
import stubs
from qdrant_client import QdrantClient

from app.config import REDACTED_INFORMATION_TOKEN_MAP
from app.runner import rag_runner
from app.schemas import GenerateRequest
from app.store.qdrant import get_encoder


@pytest.fixture
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
                        timeout=1.0,
                        prompt_template = 'basic',
                        seed = 42)


@pytest.fixture
def au_privacy_config():
    return dict(url="privacy-service",
            port=5000,
            redaction_map = REDACTED_INFORMATION_TOKEN_MAP)


@pytest.fixture
def qdrant_retriever_config():
    return dict(host='localhost', port=6333, embedding_timeout=2, top_k=3, collection_name = 'test_collection', encoder='all-MiniLM-L6-v2', filter_on_user_metadata=True)


@pytest.fixture
def basic_consolidator_config():
    return dict(token_limit=1000, strategy='simple')


@pytest.fixture
def mock_get_config(monkeypatch, gpt2_generation_config, au_privacy_config, qdrant_retriever_config, basic_consolidator_config):
    def mock_config(config_name):

        config_map = {
            'au-privacy': au_privacy_config,
            'qdrant-retrieval': qdrant_retriever_config,
            'basic-consolidator': basic_consolidator_config,
            'gpt2-generation': gpt2_generation_config,
        }

        return config_map.get(config_name, ValueError(f'Unknown config name: {config_name}'))

    monkeypatch.setattr('app.config._get_config', mock_config)


@pytest.fixture
def mock_query_llm(monkeypatch):
    """Mock the query_llm function to return a predefined response."""
    def mock_response(*args, **kwargs):
        return f"This is a mocked LLM response to {args[2]}", 12

    monkeypatch.setattr('app.generate.generate.gpt2_infer', mock_response)

@pytest.fixture
def mock_synthesize_response(monkeypatch):
    """Mock the query_llm function to return a predefined response."""
    def mock_response(user_query, contexts, event_id):
        contexts_string = ''.join([c.text for c in contexts])
        return f"This is a mocked LLM response to {user_query.user_query}, containing documents:{contexts_string}", 12

    monkeypatch.setattr('app.generate.generate.GPT2Generator.synthesize_response', mock_response)


@pytest.fixture
def mock_retrieve_vectors_empty(monkeypatch):
    """Mock the query function."""
    def mock_retrieve_vectors(*args, **kwargs):
        return [], [], [], []

    monkeypatch.setattr('app.retrieve.retrieve.QDRANTRetriever.retrieve_from_vector', mock_retrieve_vectors)


class TestRAGProcessor:

    @classmethod
    def setup_class(cls):
        cls.qdrant_client = QdrantClient("localhost", port=6333)
        cls.encoder = get_encoder({'encoder':'all-MiniLM-L6-v2'})

    @classmethod
    def teardown_class(cls):
        cls.qdrant_client.delete_collection('test_collection')

    @pytest.mark.transformers
    @pytest.mark.qdrant
    def test_user_request_answer_monitoring_logs(self, mock_query_llm, mock_get_config): # mock_write_log_to_db
        """Basic test. 
        
        Expected behaviour:
         - Given a user response, generate an answer.
         - Metrics throughout the stages are captured for system evaluation.
        
        Set-up:
         - Create a collection with test data.
         - Configure the mock query_llm function to return a predefined response (mocked)
         - Define configuration files for each stage (mocked)
        """
        request = GenerateRequest(user_query="What is AI?")
        test_data = [{"doc_id": 1, "text": "123text1"}, {"doc_id": 2, "text": "456text2"}]
        stubs.embed_create_collection(self.qdrant_client, test_data, 'test_collection', encoder=self.encoder)

        with (patch('app.logconfig.write_data') as mock_write_log_to_db):
            response, _ = rag_runner(request)

        assert response.startswith("This is a mocked LLM response")
        assert "What is AI?" in response

        expected_metrics = ['clean_sensitive_info_seconds',
                            'cleaned_request',
                            'embed_seconds',
                            'user_query_embedding',
                            'retrieve_seconds',
                            'relevant_document_ids_scores',
                            'prompt',
                            'query_llm_seconds',
                            'response_text',
                            'response_token_count']
        actual_metrics = [e[1]['log']['metric'] for e in list(mock_write_log_to_db.call_args_list)]
        assert set(actual_metrics) == set(expected_metrics)

    @pytest.mark.transformers
    @pytest.mark.qdrant
    def test_user_request_user_metadata_relevant_answer(self, mock_query_llm, mock_get_config):
        """Retriever filters its search based on user metadata.
        
        Expected Behaviour:
         - Documents returned are relevant to the user.
        
        Set-up:
         - Create a collection with test data, with 'regions' defined on each point.
         - User request has metadata for their region.
        """
        request = GenerateRequest(user_query="What is AI?", metadata={'region': 'NZ'})
        test_data = [{"doc_id": 1, "text": "123text1", 'region': 'NZ', 'title': 'New Zealand document'},
                     {"doc_id": 2, "text": "456text2", 'region': 'AU', 'title': 'Australian document'}]
        stubs.embed_create_collection(self.qdrant_client, test_data, 'test_collection', encoder=self.encoder)

        response, _ = rag_runner(request)

        assert ('123text1' in response) and ('456text2' not in response)

    @pytest.mark.transformers
    @pytest.mark.qdrant
    def test_user_request_store_docs_empty(self, mock_query_llm, mock_get_config, mock_synthesize_response):
        """Fallback behaviour if no relevant documents are found.
        
        Expected behaviour:
         - RAG will skip forward to GENERATE stage after no documents are found.
         - GENERATE stage is given 1 context item: A message about no relevant documents.
        
        Set-up:
         - Create a collection without data.
        """
        request = GenerateRequest(user_query="What is AI?")
        test_data = [{"doc_id": 1, "text": "123text1"}, {"doc_id": 2, "text": "456text2"}]
        test_data = []
        stubs.embed_create_collection(self.qdrant_client, test_data, 'test_collection', encoder=self.encoder)

        with (patch('app.retrieve.retrieve.QDRANTRetriever.retrieve_from_vector') as mock_retrieve_vectors_empty,
             patch('app.consolidate.consolidate.SimpleConsolidator.simple_consolidate') as mock_consolidate):
            mock_retrieve_vectors_empty.return_value = ([], [], [], [])

            response, _ = rag_runner(request)

        assert "No relevant documents found. Perhaps reframe your query" in response
        mock_consolidate.assert_not_called()


if __name__ == '__main__':
    pytest.main(["-s", __file__])
