from qdrant_client import QdrantClient

from app.database import get_data
from app.runner import rag_runner
from app.schemas import GenerateRequest

client = QdrantClient("localhost", port=6333)

if __name__ == "__main__":
    response, event_id = rag_runner(GenerateRequest(user_query="Download and print"))

    print("\n\nFinsihed RAG\n\n" + "-  -  " * 10)
    print(response)
    print(event_id)

    ls = get_data(event_id)

    print("Metrics:")
    print([l["log"]["metric"] for l in ls])

    prompt = next(filter(lambda e: e["log"]["metric"] == "prompt", list(ls)))
    print("\nPROMPT\n" + prompt["log"]["value"] + "\n" + "-  -  " * 10)
    response = next(filter(lambda e: e["log"]["metric"] == "response_text", list(ls)))
    doc_ids = next(filter(lambda e: e["log"]["metric"] == "relevant_document_ids_scores", list(ls)))

    print("\nRESPONSE\n" + response["log"]["value"])
    print("\nDOCUMENT_IDS\n" + doc_ids["log"]["value"])

    """
    Check the metrics sent to DB fro monitoring:

    ```
    select log from eval where event_id LIKE '%2024-03-17%' order by event_id desc limit 1;
    ```
    """
