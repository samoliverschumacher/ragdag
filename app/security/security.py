import re
import time
from typing import Any

import app.logconfig
from app.pipe import ErrorStack, Process, RAGStage
from app.schemas import CleansedRequest, GenerateRequest

logger = app.logconfig.setup_logger('root')


class SecurityServiceError(Exception):
    """Some kind of error from the security service."""


class AccountNumberRedactor(Process):
    """Raw User request -> Scrubbing service -> Cleansed user request

    Behaviours:
        - Removes sensitive information
    """

    stage = RAGStage.SECURITY

    def __init__(self, config: dict) -> None:
        super().__init__(config)

    @staticmethod
    def pid_scrub(request: GenerateRequest, config: dict,
                  event_id: str) -> CleansedRequest | None:
        """
        A method to scrub private information from the input request.

        Ideally an external security service for removing sensitive information,
        and replaces it with a token the LLM has trained on.

        Args:
            request (GenerateRequest): The input request object to be cleaned.
            config (dict): A dictionary containing configuration settings.
            event_id (str): The unique identifier for the event.

        Returns:
            CleansedRequest | None: The cleaned request object with sensitive information removed,
            or None if an error occurs.
        """

        t0 = time.time()
        try:
            # Would call external service here to remove private information. Bank details could be one.
            cleaned_text = _remove_bank_account_number(
                request.user_query,
                replacement=config['Financial information'])
        except Exception as e:
            logger.error(f'Security service had a problem. reason: {e!s}')
            raise SecurityServiceError

        fields = {k: v for k, v in request.dict().items() if k != 'user_query'}
        cleaned_request = CleansedRequest(**fields, user_query=cleaned_text)
        t1 = time.time()
        logger.eval(event_id, {
            "metric": "clean_sensitive_info_seconds",
            "value": t1 - t0
        })
        logger.eval(event_id, {
            "metric": "cleaned_request",
            "value": cleaned_request.user_query
        })

        return cleaned_request

    def _process(self, text: Any, data: dict, errors: ErrorStack,
                 sentinel: RAGStage) -> tuple[Any, dict, ErrorStack, RAGStage]:
        request: GenerateRequest = text
        event_id: str = data['event_id']

        try:
            rewritten_query = self.pid_scrub(request,
                                             self.config['redaction_map'],
                                             event_id)
            next_text: GenerateRequest = rewritten_query
        except Exception as e:
            # Give the user the retrieved references verbatim.
            errors.append((self.stage, e))
            logger.error(
                str(e) + 'Could not remove sensitive information. Aborting.')
            raise e

        return next_text, data, errors, self.next_stage


def _remove_bank_account_number(text: str, replacement: str) -> str:
    """Would be an API to an external service in production.

    Example: text = "My bank account number is 1234 5678 9012 3456."
    """
    # Matches a bank account number in the specific format
    pattern = r'\b\d{4} \d{4} \d{4} \d{4}\b'

    cleaned_text = re.sub(pattern, replacement, text)
    return cleaned_text
