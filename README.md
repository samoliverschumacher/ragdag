# RAG as a Pipeline / Directed Acyclic Graph architecure

RAG (Retrieval Augmented Generation) systems can involve more than just _Retrieval_ of context documents, and _Generation_ of an informed response:
 - **Query rewriting**: Where the input question is re-written internally, to a query that more preceisely represents the question, according to the embedding model in the reitriever, and/or the final LLM that responds to the query. This can involve another LLM to re-write the query.
 - **Document Reranking**: The retriever performs a search returning an ordered list of documents according to similarity to the question, but it's objective has more emphasis on speed of retriveal. The Reranking step allows for a higher accuracy ordering of the retrieved documents, which are a much smaller set, so the hit to speed isn't impractical.
 - **Consolidate**: Preprocessing the retrieved documents can sometimes be a lengthy process, especially when describing the structural relationship between document chunks in natural language for the LLM becomes complicated. Organising this process into it's own step helps to keep the final Generation step as simple as possible.
 - **Caching**: In low latency, large scale production environments, caching can be a great way to speed up the retrieval of documents. This could be for context documents given an input query, rewritten query given an input query, or even generated response given an input query. This process is it's own step, since the cache hash function could become complicated, or it's own service in the case of semantic caching. More than one Cache process could be in a production RAG system, either due to also writing to cache, or caching of more than one process.

Mostly, the following is true of RAG systems;
1. They are _sequential processes_, and when they are recursive, it's usually within a single step - i.e. retries of calling a service.
2. Most steps are simply a _text transformation_. A text input is converted to a text output. Stretching this statement, a text that is an embedding, is still a text, but of a different representation. 
3. With cahcing, some intermediate steps could be skipped dynamically.

And of managing RAG projects;

4. There is a lot of fine tuning required across multiple steps. Unlike a traditional ML-centric system, there is not just one model with hyperparameters. Fine tuning can involve replacing or removing steps all together. ML Engineers and Data Scientists could all be making changes to the codebase in parallel.

## The `ragdag` - primitives to support a RAG architecture.

So the design of RAG systems is a Directed Acyclic Graph (D.A.G.), or Pipeline. This pattern allows for the sequentiality (1), and statelessness (2) of RAG systems. Standard DAGs don't allow for dynamic skipping (the graph is predefined), but incorporating message passing allows a component to iteself decide to skip.

`ragdag` incorporates some building blocks that support the assumptions `1.` - `4.`.

`Process`: The bulding block. A stateless step in a pipeline, or a edge in a DAG.
- It's inputs and outputs are the same type. (`text`, `data`, `error`, `sentinel`) -> (`text`, `data`, `error`, `sentinel`)
- `text`: First input and output is a text (or array of). 
- `data`: Second is a hashmap of supporting data that has accumulated through the pipeline. 
- `error`: Third is accumulating errors handled along the way. These are passed on to inform future processes how to act.
- `sentinel`: Fourth is a sentinel that indicates the next process to enter. While the Pipeline sequence is pre-defined, in production a RAG system has many failure points that can change the trajectory of the remaining process on the fly. While this could be solved with reading the errors passed down, though this starts to encourage tighter coupling between processes. Since RAG systems have many fine tuning points (`4.`), a sentinel message helps to keep coupling loose helping the practice of continuous development.

> _While `Process` does have attributes, and a non-empty `__init__` method, it could be written just as easily in a functional language. The spirit of `Process` is a pure function, but in the OOP paradigm (🫣)._

> _See [pipe.py](app/pipe.py) or [test_pipe.py](tests/test_pipe.py) for explicit design of RAG as Pipeline / DAG._

## Examples

Designing a `Process`
1. Define it's [`RAGStage`](app/pipe.py#L6).
2. Implement the methods unique to its role in the system.
3. Define logic for error handling, and dynamically setting next stage if necessary. ([`_process()`](app/pipe.py#L61))


### Writing a RAG system;
```python
# Map out the DAG
retrieve = Retrieve()
rerank = Rerank()
generate = Generate()
pipeline = [retrieve, rerank, generate]
# Initialise, and run the pipeline
text = 'A question?'
data, err, sentinel = {}, [], RAGStage.RETRIEVE
for process in pipeline:
    text, data, err, sentinel = process(text, data, err, sentinel)
```
