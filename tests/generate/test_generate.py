import time

import pytest
import stubs

from app.generate.generate import GPT2Generator, LLMTimeoutError
from app.pipe import RAGStage
from app.retrieve.retrieve import Context, ContextWithMetadata
from app.schemas import GenerateRequest


@pytest.fixture
def mock_llm_response(monkeypatch):
    """Mock the query function."""
    def mock_response(*args, **kwargs):
        time.sleep(0.1)
        return "The answer is 42.", 12

    monkeypatch.setattr('app.generate.generate.gpt2_infer', mock_response)

@pytest.fixture
def mock_llm_response_slow(monkeypatch):
    """Mock the query function."""
    def mock_response(*args, **kwargs):
        time.sleep(2.0)
        return "The answer is 42.", 12

    monkeypatch.setattr('app.generate.generate.gpt2_infer', mock_response)


@pytest.fixture
def mock_llm_no_response(monkeypatch):
    """Mock the query function."""
    def mock_response(*args, **kwargs):
        raise Exception

    monkeypatch.setattr('app.generate.generate.gpt2_infer', mock_response)

class TestGPT2Generator:

    @classmethod
    def setup_class(cls):
        cls.config = stubs.gpt2_generation_config()
        cls.generator = GPT2Generator(cls.config)

    @pytest.mark.transformers
    def test_process(self):
        """Process propgates to an WCACHE stage."""

        query = "What is the meaning of life?"
        documents = ['friends', 'family', '42']
        doc_ids = ['id1', 'id2', 'id3']

        data = dict(user_query = query,
                    event_id='1234')
        sentinel = RAGStage.GENERATE
        errors = []
        text = [Context(doc_id=doc_id, text=doc) for doc_id, doc in zip(doc_ids, documents, strict=False)]
        response = self.generator(text, data, errors, sentinel)

        assert '42' in response[0]
        assert response[3] == RAGStage.WCACHE

    @pytest.mark.transformers
    def test_generate(self, mock_llm_response_slow, monkeypatch):
        """Given context information, and a user question, syntesise an answer.
        
        Expected behaviour:
            - Calls LLM, raises LLMTimeoutError if LLM is not available after timeout seconds.
            
        Set-up:
            - Slow Generation model returns a repsonse after 1.0 seconds. (mocked)
        """
        query = "What is the meaning of life?"
        documents = ['friends', 'family', '42']
        doc_ids = ['id1', 'id2', 'id3']

        contexts = [Context(doc_id=doc_id, text=doc) for doc_id, doc in zip(doc_ids, documents, strict=False)]
        response = self.generator.synthesize_response(GenerateRequest(user_query=query), contexts, event_id='')

        assert '42' in response

        cfg = stubs.gpt2_generation_config()
        cfg['timeout'] = 0.01
        monkeypatch.setattr(self.generator, 'config', cfg)

        with pytest.raises(LLMTimeoutError):
            response = self.generator.synthesize_response(GenerateRequest(user_query=query), contexts, event_id='')

    @pytest.mark.transformers
    def test_generate_handle_error(self, mock_llm_no_response, monkeypatch):
        """Use found references as answer when LLM errors.
        
        Expected behaviour:
            - References is returned if strategy is 'references'.
            - Documents is returned if strategy is 'documents'.
            
        Set-up:
            - LLM raises (mocked)
        """

        query = "What is the meaning of life?"
        titles = ['Friends.', 'La Famila.', 'Numbers.']
        urls = ['www.friends.com', 'www.family.com', 'www.numbers.com']
        documents = ["[title: 'Friends.', url: 'www.friends.com']\nfriends\n",
                     "[title: 'La Famila.', url: 'www.family.com']\nfamily\n",
                     "[title: 'Numbers.', url: 'www.numbers.com']\n42\n"]
        doc_ids = ['id1', 'id2', 'id3']
        monkeypatch.setattr(self.generator, 'config', {'no_llm_fallback_strategy': 'documents'})

        data = dict(user_query = GenerateRequest(user_query=query),
                    event_id='1234')
        sentinel = RAGStage.GENERATE
        errors = []
        text_in = [ContextWithMetadata(doc_id=doc_id, text=doc, url=url, title=_title)
                for doc_id, doc, url, _title in zip(doc_ids, documents, urls, titles, strict=False)]

        text, _ ,_ ,_ = self.generator(text_in, data, errors, sentinel)

        assert "[title: 'Friends.', url: 'www.friends.com']\nfriends\n" in text

        monkeypatch.setattr(self.generator, 'config', {'no_llm_fallback_strategy': 'references'})
        text, _ ,_ ,_ = self.generator(text_in, data, errors, sentinel)

        assert 'www.friends.com' in text and "friends\n" not in text

    @pytest.mark.transformers
    def test_query_llm(self):
        """GPT model correctly plugged in."""

        # Model params established on init
        config = stubs.gpt2_generation_config()
        self.generator = GPT2Generator(config)

        query = "What is the meaning of life?"
        response = self.generator.query_llm(query)

        assert len(response['text']) > 0
        assert response['token_count'] > 0

        # Model params established on init
        config = stubs.gpt2_generation_config()
        config['model_config']['max_length'] = 10
        self.generator = GPT2Generator(config)

        response = self.generator.query_llm(query)

        assert response['token_count'] == 10
        assert 0 < len(response['text']) < 100

    def test__fallback_response(self):

        titles = ['Friends.', 'La Famila.', 'Numbers.']
        urls = ['www.friends.com', 'www.family.com', 'www.numbers.com']
        documents = ["[title: 'Friends.', url: 'www.friends.com']\nfriends\n",
                     "[title: 'La Famila.', url: 'www.family.com']\nfamily\n",
                     "[title: 'Numbers.', url: 'www.numbers.com']\n42\n"]
        doc_ids = ['id1', 'id2', 'id3']
        contexts = [ContextWithMetadata(doc_id=doc_id, text=doc, url=url, title=_title)
                    for doc_id, doc, url, _title in zip(doc_ids, documents, urls, titles, strict=False)]

        response = GPT2Generator._fallback_response(contexts)
        response_documents = GPT2Generator._fallback_response(contexts, just_text=True)

        assert response['text'] == 'Xbot is down. Here are some references that may help you:0. Friends.: www.friends.com\n1. La Famila.: www.family.com\n2. Numbers.: www.numbers.com'
        assert response_documents['text'] == "Xbot is down. Here are some document excerpts that may help you:[title: 'Friends.', url: 'www.friends.com']\nfriends\n[title: 'La Famila.', url: 'www.family.com']\nfamily\n[title: 'Numbers.', url: 'www.numbers.com']\n42\n"
