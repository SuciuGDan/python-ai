import json
import os
import pprint
import sys
from pathlib import Path

import chromadb
import pymupdf4llm
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re

# def return_result():
#     result = 10
#     date_calculated = "2026"
#
#     return result, date_calculated
#
# res1, date1 = return_result()

# list1 = [30, 40, 1, -5, 0]
# list1.sort(reverse=True)

# incarcam fisierul -> chunk-uri (spargem bucati) -> embeddings -> incarcam in baza de date vectoriala -> semantic search pe DB vectorial -> returnam rezultatele.
# pentru hybrid search: calculam pentru toate chunk-urile care ne intereseaza, un scor. la fel si pentru intrebare. verificam intrebarea cu fiecare <-> chunk
# extragem un scor, si o integram cu scorurile de la semantic search.

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "pdf_chunks"
FILE_PATH = "StarMaker.pdf"
INVOICES_FILE_PATH = "invoices.pdf"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
HYBRID_SEARCH_RESULT_COUNT = 10


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(description="Question about the contents of StarMaker.pdf.")


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        # get a slice of text, using start as the starting index.
        chunk = text[start : start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        # make a step, update the index(start)
        start = start + CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def index_file(collection, model, file_path):
    pages = pymupdf4llm.to_markdown(str(file_path), page_chunks=True)
    records = []
    ids = []
    documents = []
    metadatas = []

    for page in pages:
        page_number = page["metadata"]["page_number"]
        page_text = page["text"]

        chunks = chunk_text(page_text)
        chunk_number = 0

        for chunk in chunks:
            # tuples -> ("john", "carpenter", 33)
            #                 0            1   2
            records.append((chunk, page_number, chunk_number))
            ids.append(f"{file_path.name}:p{page_number}:c{chunk_number}")
            documents.append(chunk)
            metadatas.append(
                {
                    "source": file_path.name,
                    "page": page_number,
                    "chunk": chunk_number,
                }
            )
            chunk_number += 1

    collection.delete(where={"source": file_path.name})

    collection.upsert(
        ids=ids,
        # this is where embedding happens for each chunk
        # normalize_embeddings normalizes all vectors, their values are between [0, 1]
        embeddings=model.encode(documents, normalize_embeddings=True).tolist(),
        documents=documents,
        # aici inseram datele vectoriale in baza de date:
        metadatas=metadatas,
    )

    return documents, metadatas


def complete(client: OpenAI, messages: list[dict], tools: list[dict]):
    return client.chat.completions.create(
        messages=messages,
        model=os.getenv("OPEN_ROUTER_MODEL_NAME"),
        tools=tools,
        tool_choice="auto",
    ).choices[0].message


def hybrid_search(collection, model, bm25, documents, metadatas, args):
    request = SearchRequest.model_validate(json.loads(args))
    question = request.question
    question_vector = model.encode(question, normalize_embeddings=True).tolist()

    # o sa ne folosim de 2 seturi de rezultate, si le combinam pentru un scor hibrid.
    # intai facem cautare semantica.
    semantic_results = collection.query(
        query_embeddings=[question_vector],
        n_results=HYBRID_SEARCH_RESULT_COUNT,
        include=["documents", "metadatas", "distances"],
    )

    hybrid_scores = {}
    semantic_documents = semantic_results["documents"][0]
    semantic_docs_metadatas = semantic_results["metadatas"][0]

    # Rank 0 receives 1 point, rank 1 receives 1/2 point, and so on.
    rank = 0

    for i in range(len(semantic_docs_metadatas)):
        metadata = semantic_docs_metadatas[i]
        # Find this semantic result's position in the original documents list.
        chunk_index = metadatas.index(metadata)
        hybrid_scores[chunk_index] = 1 / (rank + 1)
        rank += 1

    bm25_scores = bm25.get_scores(tokenize(question))
    bm25_results = []
    chunk_index = 0
    for score in bm25_scores:
        bm25_results.append((score, chunk_index))
        chunk_index += 1
    # sortam invers toate rezultatele bm25. sortarea asta pune pe primele locuri chunk-urile cu cel mai ridicat scor.
    bm25_results.sort(reverse=True)

    rank = 0
    for _, chunk_index in bm25_results[:HYBRID_SEARCH_RESULT_COUNT]:
        if chunk_index not in hybrid_scores:
            hybrid_scores[chunk_index] = 0
        hybrid_scores[chunk_index] += 1 / (rank + 1)
        rank += 1

    combined_results = []
    for chunk_index, score in hybrid_scores.items():
        combined_results.append((score, chunk_index))
    combined_results.sort(reverse=True)

    results = []
    for _, chunk_index in combined_results[:HYBRID_SEARCH_RESULT_COUNT]:
        results.append(
            f"{documents[chunk_index]}\nSource: {metadatas[chunk_index]['source']}, page {metadatas[chunk_index]['page']}"
        )
    return results


def answer_with_hybrid_context(client: OpenAI, args: str, documents: list[str]) -> str:
    try:
        request = SearchRequest.model_validate(json.loads(args))
        context = "\n\n".join(documents)
        return client.chat.completions.create(
            model=os.getenv("OPEN_ROUTER_MODEL_NAME"),
            messages=[
                {
                    "role": "system",
                    "content": "Answer only from the supplied context. If it does not answer the question, say so.",
                },
                {"role": "user", "content": f"Question: {request.question}\n\nContext:\n{context}"},
            ],
        ).choices[0].message.content
    except (json.JSONDecodeError, ValidationError) as error:
        return f"error: {error}"


def answer_with_context(client: OpenAI, collection, model, args: str) -> str:
    try:
        request = SearchRequest.model_validate(json.loads(args))
        vector = model.encode(request.question, normalize_embeddings=True).tolist()
        result = collection.query(
            query_embeddings=[vector],
            n_results=8,
            include=["documents", "metadatas"],
        )
        context_pieces = []
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]

        # construim un string structurat pentru LLM
        for i in range(len(documents)):
            document = documents[i]
            metadata = metadatas[i]
            context_pieces.append(
                f"{document}\nSource: {metadata['source']}, page {metadata['page']}"
            )

        context = "\n\n".join(context_pieces)
        return client.chat.completions.create(
            model=os.getenv("OPEN_ROUTER_MODEL_NAME"),
            messages=[
                {
                    "role": "system",
                    "content": "Answer only from the supplied context. If it does not answer the question, say so. Cite every factual claim with the provided source.",
                },
                {"role": "user", "content": f"Question: {request.question}\n\nContext:\n{context}"},
            ],
        ).choices[0].message.content
    except (json.JSONDecodeError, ValidationError, KeyError) as error:
        return f"error: {error}"


def main():
    load_dotenv()
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    )
    model = SentenceTransformer(MODEL_NAME)
    collection = chromadb.PersistentClient().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    documents = []
    metadatas = []
    for file_path in [FILE_PATH, INVOICES_FILE_PATH]:
        file_documents, file_metadatas = index_file(collection, model, Path(file_path))
        documents.extend(file_documents)
        metadatas.extend(file_metadatas)

    # alt pas de indexare, unde calculam vectorii pentru BM25
    # intai, tokenizam chunk-urile
    tokenized_documents = []
    for doc in documents:
        tokens = tokenize(doc)
        tokenized_documents.append(tokens)

    bm25 = BM25Okapi(tokenized_documents)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_starmaker",
                "description": "Searches StarMaker.pdf and invoices.pdf and answers from them. Use only when the user asks about their contents.",
                "parameters": SearchRequest.model_json_schema(),
            },
        }
    ]
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use search_starmaker only for questions about StarMaker.pdf, invoices.pdf, or their contents. Use it before answering those questions. Answer other questions normally.",
        }
    ]

    while True:
        question = input("you> ")
        if question == "quit":
            break

        messages.append({"role": "user", "content": question})
        response = complete(client, messages, tools)
        messages.append(response.model_dump(exclude_none=True))

        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call.function.name == "search_starmaker":
                    results = hybrid_search(collection, model, bm25, documents, metadatas, tool_call.function.arguments)
                    output = answer_with_hybrid_context(client, tool_call.function.arguments, results)
                else:
                    output = f"error: unknown tool {tool_call.function.name}"
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": output})

            response = complete(client, messages, tools)
            messages.append(response.model_dump(exclude_none=True))

        print("agent>", response.content)


if __name__ == "__main__":
    main()