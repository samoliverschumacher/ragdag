import datetime
import time
import uuid

import app.logconfig
from app.config import get_config
from app.consolidate.consolidate import SimpleConsolidator
from app.generate.generate import GPT2Generator
from app.pipe import Process, create_links
from app.retrieve.retrieve import QDRANTRetriever
from app.schemas import GenerateRequest
from app.security.security import AccountNumberRedactor

logger = app.logconfig.setup_logger('root')
logger.debug('runner.py is running...')


def run_dag(request: GenerateRequest, event_id: str,
             processes: list[Process]) -> str:
    """
    Runs the directed acyclic graph (DAG) defined by the given request, event ID, and list of processes,
    and returns the next text after processing all stages.

    This function captures the latency of each stage and reports
    it to the info logger. It also logs any errors that were
    captured during the processing of each stage but not raised.

    Args:
        request (GenerateRequest): The request to be processed.
        event_id (str): The ID of the event triggering the DAG run, used to align metrics.
        processes (List[Process]): The list of processes to be executed in the DAG.

    Returns:
        str: The next text after processing the entire DAG.
    """
    next_text, data, errors, sentinel = request, {
        'event_id': event_id
    }, [], processes[0].stage
    _err_counter = len(errors)
    for process in processes:
        t0 = time.time()
        logger.info(f"Running {process.stage=}")
        next_text, data, errors, sentinel = process(next_text, data, errors,
                                                    sentinel)
        t1 = time.time()
        if len(errors) > _err_counter:
            logger.error(
                f"Error in {process.stage=}. Attempting to Continue...")
            for error in errors:
                logger.error(error)
            _err_counter = len(errors)
        logger.info(f"Completed {process.stage=} in {t1 - t0} seconds")

    return next_text


def rag_runner(request: GenerateRequest) -> tuple[str, str]:
    """
    This function runs the R.A.G. (Retrieval, Augmented, Generation) process.
    It retrieves configuration settings, initializes several components, creates
    a DAG, and then runs the DAG to generate a response.
    """

    privacy_config = get_config('au-privacy')
    logger.debug(f"Privacy config: {privacy_config}")
    retrieval_config = get_config('qdrant-retrieval')
    logger.debug(f"Retrieval config: {retrieval_config}")
    consolidator_config = get_config('basic-consolidator')
    logger.debug(f"Consolidator config: {consolidator_config}")
    generation_config = get_config('gpt2-generation')
    logger.debug(f"Generation config: {generation_config}")

    security_cleaner = AccountNumberRedactor(privacy_config)
    document_retriever = QDRANTRetriever(retrieval_config)
    context_consolidator = SimpleConsolidator(consolidator_config)
    prompt_reader = GPT2Generator(generation_config)

    event_id = str(uuid.uuid4()) + str(datetime.datetime.now())
    logger.info(f"Starting Xbot with {event_id=}")

    dag = [
        security_cleaner, document_retriever, context_consolidator,
        prompt_reader
    ]
    create_links(dag)

    logger.info('Initialising R.A.G. DAG: ' +
                ' -> '.join([f"{d.stage.name}" for d in dag]))
    response = run_dag(request, event_id, dag)
    return response, event_id
