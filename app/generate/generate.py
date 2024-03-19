import time
from typing import Any

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import app.logconfig
from app.pipe import ErrorStack, Process, RAGStage
from app.retrieve.retrieve import Context, ContextWithMetadata
from app.schemas import GenerateRequest
from app.utils import run_until_timeout

logger = app.logconfig.setup_logger('root')

NO_LLM_REFERENCES_PROMPT = """Xbot is down. Here are some references that may help you:{urls_titles}"""
NO_LLM_DOCUMENTS_PROMPT = """Xbot is down. Here are some document excerpts that may help you:{documents}"""

PROMPT_TEMPLATE= ("Given a user query, and context documents, generate a response"
                  ".\nUSER_QUERY:\n'{user_query}'\nCONTEXT_ITEMS:\n{context}\n")


class LLMTimeoutError(Exception):
    """Raised when querying the LLM takes too long to respond."""
    pass


class ContextLengthError(Exception):
    """Raised when the context length is too long."""
    pass


class GPT2Generator(Process):
    """Prompt -> LLM -> Response.

    Behaviours:
        - Initialises model endpoint.
        - If error in LLM service, prepares a response 
          containing the context references as fallback strategy
    """
    stage = RAGStage.GENERATE

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.config = config
        self.template = "{user_query}{context}"
        if config['prompt_template'] == 'basic':
            self.template = PROMPT_TEMPLATE
        self.eos_token = self.config['model_config'].get(
            'eos_token', '<|endoftext|>')

        self.model, self.device = _gpt2model(config['model_config'])
        self.tokenizer = _gpt2tokeniser(config['model_config'])

    def query_llm(self, query: str) -> dict:
        """
        A function that queries a language model to generate text based on a given input query string.

        Parameters:
            query (str): The input query string to generate text from.

        Returns:
            dict: A dictionary containing the generated text and the token count.
        """
        answer, token_count = self._generate_text(query)
        return {'text': answer, 'token_count': token_count}

    def synthesize_response(self,
                            user_query: GenerateRequest,
                            contexts: list[Context] | list[ContextWithMetadata], event_id: str) -> str:
        """
        Synthesizes a response to a user query using a language model.

        Query's the LLM with a timeout limit. 
        Captures metics: `query_llm_seconds` `response_token_count` `response_text` `prompt`

        Raises:
            LLMTimeoutError: If the LLM takes too long to respond.

        Args:
            user_query (GenerateRequest): The user query to generate a response for.
            contexts: The contexts to use for generating the response.
            event_id: The ID of the event.

        Returns:
            str: The generated response.
        """

        prompt = self._format_prompt(user_query.user_query, contexts)
        if self.config['model_config']['max_length'] - len(prompt) <= 10:
            logger.error(
                "Query is too long to fit in the model's context window. {prompt_len}, {max_len}"
                .format(prompt_len=len(prompt),
                        max_len=self.config['model_config']['max_length']))

        logger.debug(f"Prompt: {prompt}\n")

        logger.eval(event_id, {"metric": "prompt", "value": prompt})

        t0 = time.time()
        response = run_until_timeout(self.query_llm, self.config['timeout'],
                                     LLMTimeoutError, prompt)
        t1 = time.time()
        logger.eval(event_id, {
            "metric": "query_llm_seconds",
            "value": t1 - t0
        })

        logger.eval(event_id, {
            "metric": "response_text",
            "value": response['text']
        })
        logger.eval(event_id, {
            "metric": "response_token_count",
            "value": response['token_count']
        })

        return response['text']

    def _process(self, text: Any, data: dict, errors: ErrorStack,
                 sentinel: RAGStage) -> tuple[Any, dict, ErrorStack, RAGStage]:
        query: GenerateRequest = data['user_query']
        contexts: list[Context] | list[ContextWithMetadata] = text
        event_id: str = data['event_id']

        try:
            consolidated_contexts = self.synthesize_response(
                query, contexts, event_id)
            next_text: str = consolidated_contexts
        except Exception as e:
            if type(e) == LLMTimeoutError:
                logger.error("Querying the LLM took too long to respond")
            # Give the user the retrieved references verbatim.
            response = self._fallback_response(
                contexts,
                just_text=self.config['no_llm_fallback_strategy'] ==
                'documents')
            errors.append((self.stage, e))
            logger.error(str(e) + 'Applied a no-LLM fallback response.')
            next_text = response['text']

        return next_text, data, errors, self.next_stage

    @staticmethod
    def _fallback_response(contexts: list[ContextWithMetadata | Context],
                           just_text: bool = False) -> dict:
        if just_text:
            context_text = ''.join([c.text for c in contexts])
            response = {
                'text': NO_LLM_DOCUMENTS_PROMPT.format(documents=context_text),
                'token_count': None
            }
        else:  # contexts: ContextWithMetadata
            context_urls_titles = '\n'.join([
                f'{i}. {context.title}: {context.url}'
                for i, context in enumerate(contexts)
            ])
            response = {
                'text':
                NO_LLM_REFERENCES_PROMPT.format(
                    urls_titles=context_urls_titles),
                'token_count':
                None
            }
        return response

    def _generate_text(self, text: str) -> tuple[str, int]:
        text, tc = gpt2_infer(self.model, self.tokenizer, text, self.device,
                              self.config['model_config'])
        return text, tc

    def _format_prompt(
            self, user_query: str,
            contexts: list[Context] | list[ContextWithMetadata]) -> str:
        context_text = ''.join([c.text for c in contexts])
        query = self.template.format(user_query=user_query,
                                     context=context_text)
        return query + self.eos_token


def gpt2_infer(model: GPT2LMHeadModel,
               tokeniser: GPT2Tokenizer,
               input_text: str,
               device: str = 'cpu',
               config: dict = None) -> tuple[str, int]:
    """
    Uses a GPT-2 model to generate text based on the input_text provided.

    Args:
        model: The GPT-2 model to use for text generation.
        tokeniser: The tokeniser to encode and decode text for the model.
        input_text (str): The input text to base the generation on.
        device (str): The device to use for generation, default is 'cpu'.
        config (dict): A dictionary containing configuration parameters for generation.

    Returns:
        Tuple[str,int]: A tuple containing the generated response text and the token count.

    Example:
    >>> config = {}
    >>> config['model'] = 'gpt2'
    >>> config['max_length'] = 100
    >>> config['num_return_sequences'] = 1
    >>> config['device'] = 'cpu'
    >>> m, d = _gpt2model(config)
    >>> tk = _gpt2tokeniser(config)
    >>> input_text = "Tell me a joke"
    >>> response = gpt2_infer(m, tk, input_text, d, config)
    """

    assert isinstance(
        input_text,
        str), f"input_text must be a string, got {type(input_text)}"

    input_ids = tokeniser.encode(input_text, return_tensors='pt').to(device)

    # Generate text
    output = model.generate(
        input_ids,
        max_length=config['max_length'],
        num_return_sequences=config['num_return_sequences'],
        no_repeat_ngram_size=config['no_repeat_ngram_size'])

    # Decode the output
    generated_text = tokeniser.decode(output[0], skip_special_tokens=True)
    token_count = len(output[0])
    response = generated_text[len(input_text) - 1:]
    return response, token_count


def _gpt2tokeniser(config: dict) -> GPT2Tokenizer:
    # Load the pre-trained GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config['model'])
    return tokenizer


def _gpt2model(config: dict) -> GPT2LMHeadModel:

    if config['device'] == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    model = GPT2LMHeadModel.from_pretrained(config['model'])

    if config['device'] == 'cuda':
        model.to(device)
    return model, device
