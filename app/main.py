import datetime
import uuid

from fastapi import FastAPI
from fastapi.responses import JSONResponse

import app.logconfig
from app import runner
from app.config import get_config
from app.consolidate.consolidate import SimpleConsolidator
from app.generate.generate import GPT2Generator
from app.pipe import create_links
from app.retrieve.retrieve import QDRANTRetriever
from app.schemas import GenerateRequest, GenerateResponse, GetArticleResponse, PatchArticleRequest
from app.security.security import AccountNumberRedactor

logger = app.logconfig.setup_logger("root")

app = FastAPI()

privacy_config = get_config("au-privacy")
logger.debug(f"Privacy config: {privacy_config}")
retrieval_config = get_config("qdrant-retrieval")
logger.debug(f"Retrieval config: {retrieval_config}")
consolidator_config = get_config("basic-consolidator")
logger.debug(f"Consolidator config: {consolidator_config}")
generation_config = get_config("gpt2-generation")
logger.debug(f"Generation config: {generation_config}")

security_cleaner = AccountNumberRedactor(privacy_config)
document_retriever = QDRANTRetriever(retrieval_config)
context_consolidator = SimpleConsolidator(consolidator_config)
prompt_reader = GPT2Generator(generation_config)

dag = [security_cleaner, document_retriever, context_consolidator, prompt_reader]
logger.info("Initialising R.A.G. DAG: " + " -> ".join([f"{d.stage.name}" for d in dag]))
create_links(dag)  # type: ignore


def rag_runner(request: GenerateRequest, event_id) -> str:
    """Interface to the R.A.G. DAG."""
    response = runner.run_dag(request, event_id, dag)  # type: ignore
    return response


@app.post("/xbot/generate")
def generate(request: GenerateRequest) -> GenerateResponse:
    """Process a user request through the RAG pipeline and return the response."""

    event_id = str(uuid.uuid4()) + str(datetime.datetime.now())
    logger.info(f"Starting app with {event_id=}")

    generated_text = rag_runner(request, event_id)

    return GenerateResponse(generated_text=generated_text)


@app.patch("/xbot/collection/articles")
def patch_articles(articles: PatchArticleRequest) -> JSONResponse:
    """
    Perform an upsert on the collection for a list of articles
    using the `doc_id` as the unique key. I.e. Insert the article
    if no document with that `doc_id` exists, update it if it does
    and if the `text` field is empty, delete the article from the collection.
    """
    # Implement upserting of the collection here
    try:
        ...
    except Exception:
        JSONResponse(content={"message": "Failure"}, status_code=422)

    return JSONResponse(content={"message": "Success!"}, status_code=200)


@app.get(
    "/xbot/collection/articles/{doc_id}", response_model=GetArticleResponse
)  # Fail fast. Instead of validating on return  (https://www.youtube.com/watch?v=7jtzjovKQ8A)
def get_article(doc_id: str) -> GetArticleResponse:
    return GetArticleResponse(
        doc_id=doc_id,
        text="To setup a bank feed ...",
        created_by="cx user",
        created_date=datetime.datetime.fromisoformat("2024-02-04T13:16:09.812614Z"),
    )
