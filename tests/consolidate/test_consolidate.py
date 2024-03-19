import pytest

from app.consolidate.consolidate import AllDocumentsExceedContextWindowError, SimpleConsolidator
from app.pipe import RAGStage
from app.retrieve.retrieve import Context, ContextWithMetadata


class TestSimpleConsolidator:

    @classmethod
    def setup_class(cls):
        cls.config = {'strategy': 'simple',
                      'token_limit': 100}
        cls.consolidator = SimpleConsolidator(cls.config)

    def test_process(self):
        """Process propgates to a GENERATE stage."""

        contexts = [Context(doc_id='id1', text='hello'),
                        Context(doc_id='id2', text='world'),
                        Context(doc_id='id3', text='goodbye')]
        text, data, errors, sentinel = contexts, {}, [], RAGStage.CONSOLIDATE
        expected_response = [Context(doc_id='id1', text='hello\n'),
                             Context(doc_id='id2', text='world\n'),
                             Context(doc_id='id3', text='goodbye\n')]

        response = self.consolidator(text, data, errors, sentinel)

        assert response[0] == expected_response
        assert response[3] == RAGStage.GENERATE

    def test_consolidate(self):
        """Converts a list of document chunks into a single string ready for generation."""

        contexts = [Context(doc_id='id1', text='hello'),
                        Context(doc_id='id2', text='world'),
                        Context(doc_id='id3', text='goodbye')]
        config = self.consolidator.config
        expected_response = [Context(doc_id='id1', text='hello\n'),
                             Context(doc_id='id2', text='world\n'),
                             Context(doc_id='id3', text='goodbye\n')]

        response = self.consolidator.simple_consolidate(contexts, config)

        assert response == expected_response

    def test_consolidate_context_window_threshold(self):
        """No document fits in context window."""

        contexts = [Context(doc_id='id1', text='hello'),
                        Context(doc_id='id2', text='world'),
                        Context(doc_id='id3', text='goodbye')]
        config = self.consolidator.config
        config['token_limit'] = 1

        with pytest.raises(AllDocumentsExceedContextWindowError):
            _ = self.consolidator.simple_consolidate(contexts, config)

    def test_consolidate_limited_window(self):
        """Converts a list of document chunks into a single string ready for generation."""

        contexts = [Context(doc_id='id1', text='hello'),
                        Context(doc_id='id2', text='world'),
                        Context(doc_id='id3', text='goodbye')]
        config = self.consolidator.config
        config['token_limit'] = 13
        expected_response = [Context(doc_id='id1', text='hello\n'),
                             Context(doc_id='id2', text='world\n')]

        response = self.consolidator.simple_consolidate(contexts, config)

        assert response == expected_response

    def test_consolidate_metadata(self):
        """Converts a list of document chunks into a single string ready for generation."""

        contexts = [ContextWithMetadata(doc_id='id1', text='hello', title="title1", url="url1"),
                                  ContextWithMetadata(doc_id='id2', text='world', title="title2", url="url2"),
                                  ContextWithMetadata(doc_id='id3', text='goodbye', title="title3", url="url3")]
        config = self.consolidator.config
        config['token_limit'] = 1000
        expected_response = [ContextWithMetadata(doc_id='id1', text="[title: 'title1', url: 'url1']\nhello\n", title="title1", url="url1"),
                             ContextWithMetadata(doc_id='id2', text="[title: 'title2', url: 'url2']\nworld\n", title="title2", url="url2"),
                             ContextWithMetadata(doc_id='id3', text="[title: 'title3', url: 'url3']\ngoodbye\n", title="title3", url="url3")]

        response = self.consolidator.simple_consolidate(contexts, config)

        assert response == expected_response

    def test_score_weighted_strategy(self):
        # Define contexts with some scores
        contexts = [ContextWithMetadata(doc_id='id1', text='hello', title="title1", url="url1", score=0.5),
                    ContextWithMetadata(doc_id='id2', text='world', title="title2", url="url2", score=0.8)]

        config = {'token_limit': 1000, 'strategy': 'score_weighted'}

        result = self.consolidator.simple_consolidate(contexts, config)

        # Assert that the contexts are sorted by score in descending order
        assert result == [contexts[1], contexts[0]]
