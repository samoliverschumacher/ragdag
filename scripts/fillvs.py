import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from app import ROOT_DIR

from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct

import json
import yaml
import os
import json
import yaml

def config_load():
    REDACTED_INFORMATION_TOKEN_MAP = { 'Persons names': '<REDACTED: Persons names>',
                                    'Financial information': '<REDACTED: Financial information>' }

    au_security_config = dict(url="privacy-service", port=5000, redaction_map = REDACTED_INFORMATION_TOKEN_MAP)
    
    qdrant_retriever_config = dict(host = 'localhost', 
                                port=6333, 
                                embedding_timeout=10.0, 
                                top_k=1, 
                                search_params = dict(), 
                                collection_name='articles_short_100',
                                encoder='all-MiniLM-L6-v2', 
                                filter_on_user_metadata=True)

    gpt2_default_config = {}
    gpt2_default_config['model'] = 'gpt2'
    gpt2_default_config['max_length'] = 300
    gpt2_default_config['num_return_sequences'] = 1
    gpt2_default_config['no_repeat_ngram_size'] = 3
    gpt2_default_config['device'] = 'cpu'
    gpt2_generation_config = dict(redacted_information_token_map = REDACTED_INFORMATION_TOKEN_MAP,
                        model_config = gpt2_default_config,
                        retries=3,
                        no_llm_fallback_strategy = 'references',
                        timeout=10.0,
                        prompt_template = 'basic',
                        seed = 42)

    basic_consoldiator_config = dict(token_limit=1000, strategy='simple') 

    semantic_cache_config = dict(size = None,
                                match_tolerance = 1e3,
                                metric = 'cosine',
                                encoder_config = dict())

    svm_reranker_config = dict(top_k = None)

    # The configurations to save
    configs = {
        "REDACTED_INFORMATION_TOKEN_MAP": REDACTED_INFORMATION_TOKEN_MAP,
        "qdrant_retrieval_config": qdrant_retriever_config,
        "gpt2_generation_config": gpt2_generation_config,
        "basic_consoldiator_config": basic_consoldiator_config,
        "semantic_cache_config": semantic_cache_config,
        "svm_reranker_config": svm_reranker_config,
        'au_privacy_config': au_security_config
    }

    # Save to YAML
    with open('config.yaml', 'w') as yaml_file:
        yaml.dump(configs, yaml_file, default_flow_style=False)

    # Save to JSON
    with open('config.json', 'w') as json_file:
        json.dump(configs, json_file, indent=4)
        


    def load_config(file_path):
        # Check the file extension and load the data accordingly
        _, file_extension = os.path.splitext(file_path)
        try:
            with open(file_path, 'r') as f:
                if file_extension == '.json':
                    return json.load(f)
                elif file_extension in ('.yaml', '.yml'):
                    return yaml.safe_load(f)
                else:
                    raise ValueError('Unsupported file type')
        except FileNotFoundError:
            print(f"The file {file_path} was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

    # Example usage:
    # Assuming 'config.yaml' and 'config.json' are in the current directory
    yaml_config = load_config('config.yaml')
    json_config = load_config('config.json')

    print('YAML config:', yaml_config)
    print('JSON config:', json_config)
        
# config_load()


def add_missing_fields(payload):
    if 'doc_id' not in payload:
        raise ValueError("Payload must contain 'doc_id' field")
    
    if 'title' not in payload:
        payload['title'] = f"Title: {payload['doc_id']}"
    

def create_collection(client, data, collection_name):
    """
    data: List of dict with keys 'id' (int), 'vector' (List[float]), 'payload' (dict)
    """
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    size = encoder.get_sentence_embedding_dimension()

    print(f"\nembedding size = {size=}\n")
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=size,  # Vector size is defined by used model
            distance=models.Distance.DOT,
        ),
    )

    points = []
    for idx, item in enumerate(data):
        vector = encoder.encode(item["text"]).tolist()
        # vector = [0.1,.43,.5,.6]
        payload = {k:v for k,v in item.items()} # if k != "text"}
        points.append(PointStruct(id=idx, vector=vector, payload=payload))

    client.upload_points(
        collection_name=collection_name,
        points=points,
    )

import click
import json
from qdrant_client import QdrantClient


@click.group()
def cli():
    pass

@cli.command()
@click.argument('filename')
@click.option('--collection_name', default=None, help='Name of the collection')
def fill_collection(filename, collection_name):
    """Given a json file and a collection name, create a QDRANT client and fill it with the data.
    If the collection name is not given, a name will be created from the filename.
    """
    print("Initializing QDRANT client and populating collection...")
    # Pytest mark qdrant would be used here to indicate that the service was started
    client = QdrantClient("localhost", port=6333)

    
    filename = Path(filename)
    collection_name = None
    if collection_name is None:
        collection_name = filename.stem # Creating collection name from the filename

    with open(filename, "r") as file:
        data = json.load(file)

    create_collection(client, data, collection_name)

    # client.delete_collection(collection_name)
    print("Created collection '{collection_name}' with {n} records.".format(n=len(data), collection_name=collection_name))


@cli.command()
@click.argument('collection_name')
def delete_collection(collection_name):
    """Delete a collection by name"""
    client = QdrantClient("localhost", port=6333)
    client.delete_collection(collection_name)
    print("Deleted collection: {}".format(collection_name))
    

@cli.command()
@click.argument('query_text')
@click.argument('collection_name')
@click.option('--limit', default=3, help='Limit')
def query(query_text, collection_name, limit):
    """Query a collection by text
    `python scripts/fillvs.py query "bank feeds" "articles_short" --limit 2`
    """
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    client = QdrantClient("localhost", port=6333)
    query_vector = encoder.encode(query_text).tolist()
    result = client.search(collection_name=collection_name, 
                           query_vector=query_vector, 
                           limit=limit)
    for r in result:
        print(f"{r.id=}, {r.score=}, {r.payload=}") # (score, payload) (lower score, higher.)
    
@cli.command()
def test():
    print('test')
    
if __name__ == '__main__':
    cli()