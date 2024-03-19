import pytest

from app.pipe import RAGStage
from app.retrieve.retrieve import Context
from app.schemas import GenerateRequest
from app.security.security import AccountNumberRedactor, SecurityServiceError


class TestAccountNumberRedactor:

    @classmethod
    def setup_class(cls):
        REDACTED_INFORMATION_TOKEN_MAP = { 'Persons names': '<REDACTED: Persons names>',
                                           'Financial information': '<REDACTED: Financial information>' }
        cls.config = dict(url="privacy-service", port=5000, redaction_map = REDACTED_INFORMATION_TOKEN_MAP)
        cls.consolidator = AccountNumberRedactor(cls.config)

    def test_process(self):
        """Process propgates to a END stage."""

        query = "My bank account number is 1234 5678 9012 3456. What is the meaning of life?"

        data = dict(contexts = [Context(doc_id='id', text='text')], event_id='1234')
        sentinel = RAGStage.SECURITY
        errors = []
        text = GenerateRequest(user_query=query)
        response = self.consolidator(text, data, errors, sentinel)

        assert '<REDACTED: Financial information>' in response[0].user_query and '1234 5678 9012 3456' not in response[0].user_query
        assert response[3] == RAGStage.RCACHE

    # @pytest.mark.skip
    def test_pid_scrub(self):
        """Service would use replacement tokens to censor information it deems sensitive.
        
        Do not return original query if service fails.
        """
        query = "My bank account number is 1234 5678 9012 3456. What is the meaning of life?"
        config = { 'Financial information': '<REDACTED: Financial information>' }

        response = AccountNumberRedactor.pid_scrub(GenerateRequest(user_query=query), config, event_id='')

        assert '1234 5678 9012 3456' not in response.user_query

        # Simulate no response from service / error.
        config = None

        with pytest.raises(SecurityServiceError):
            response = AccountNumberRedactor.pid_scrub(GenerateRequest(user_query=query), config, event_id='')
