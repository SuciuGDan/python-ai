import json
import os
import re

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from lesson_08_semantic_search import COLLECTION_NAME, HYBRID_SEARCH_RESULT_COUNT

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "my_collection"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000

HYBRID_SEARCH_RESULT_COUNT = 10
MAX_QUESTION_LENGTH = 255

MAX_DISTANCE = 0.7
SHOW_DISTANCES = True    # False after tuning the MAX_DISTANCE is done

NO_RESULTS_MESSAGE = "No relevant information was found for this question"


def UserQuestion(BaseModel):
    model_config = ConfigDict (extra="forbid", str_strip_whitespace=True)      #str_strip_whitespaces to avoid empty questions

    question: str = Field(
        min_length= 1,
        max_length=MAX_QUESTION_LENGTH,
        description="The user's question about the documents",
    )

def ContextOutput(BaseModel):     #Validates the tool's output ------ UserQuestion covers the tool's input
    model_config = ConfigDict (extra="forbid")
    context:str = Field(min_length=1,description="Plain text context for the agent")   #Cannot hand the agent an empty string


def tokenize(text:str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def complete(client: OpenAI, messages: list[dict], tools: list[dict]):
    return client.chat.completions.create(
        messages=messages,
        model=os.getenv("OPEN_ROUTER_MODEL_NAME"),
        tools=tools,
        tool_choice="auto",
    ).choices[0].message


def load_chunks(collection):    # loading chunks from the ChromaDB database
    data = collection.get(include=["documents","metadatas"])

    positions = {}
    for i in range(len(data["ids"])):
        positions[data["ids"][i]] = i

    return data["documents"], data["metadatas"],positions


def rag_search(collection, model, bm25, positions, documents, metadatas, args: str) -> str:

    # ========Validate the input of the tool========
    try:
        request = UserQuestion.model_validate(json.loads(args))
    except (json.JSONDecodeError, ValidationError) as error:
        return f"error: {error}"


    question = request.question
    question_vector = model.encode(question, normalize_embeddings=True).tolist()

    #========Semantic search==========

    semantic_results = collection.query(
        query_embeddings = [question_vector],
        n_results=HYBRID_SEARCH_RESULT_COUNT,
        include = ["distances"],
    )
    semantic_ids = semantic_results["ids"][0]
    semantic_distances = semantic_results["distances"][0]

    if SHOW_DISTANCES:
        print ("[distances]", [round(distance, 3) for distance in semantic_distances])


    hybrid_scores = {}

    # Rank 0 receives 1 point, rank 1 is 1/2 points etc

    rank = 0
    for i in range(len(semantic_ids)):
        if semantic_distances[i] > MAX_DISTANCE:
            continue
        chunk_index = positions[semantic_ids[i]]
        hybrid_scores[chunk_index] = 1 / (rank+1)
        rank += 1

        # if nothing was close enough in meaning -> nothing usable was found
        if len(hybrid_scores) == 0:
            return NO_RESULTS_MESSAGE


        #======= BM25 search: ignore chunks that do not match any word ==============

        bm25_scores = bm25.get_scores(tokenize(question))
        bm25_results = []
        chunk_index = 0
        for score in bm25_scores:
            if score > 0:
                bm25_results.append((score, chunk_index))
            chunk_index += 1
        bm25_results.sort(reverse=True)

        rank = 0
        for _, chunk_index in bm25_results[:HYBRID_SEARCH_RESULT_COUNT]:
            if chunk_index not in hybrid_scores:
                hybrid_scores[chunk_index] = 0
            hybrid_scores[chunk_index] += 1 / (rank+1)
            rank += 1

        #========== Combine both rankings and build context string ==============

        combined_results = []
        for chunk_index, score in hybrid_scores.items():
            combined_results.append((score, chunk_index))
        combined_results.sort(reverse=True)

        context_pieces = []
        for _, chunk_index in combined_results[:HYBRID_SEARCH_RESULT_COUNT]:
            context_pieces.append(
                f"{documents[chunk_index]}\n"
                f"Source:{metadatas[chunk_index]['source']}, page{metadatas[chunk_index]['page']}"
            )
        context = "\n\n".join(context_pieces)


        #=========Validate the output of the tool===============

        try:
            return ContextOutput(context=context).context
        except ValidationError as error:
            return f"error: {error}"

