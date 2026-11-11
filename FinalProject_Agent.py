import json
import os
import re

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "my_collection"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000

HYBRID_SEARCH_RESULT_COUNT = 10
MAX_QUESTION_LENGTH = 255


MAX_DISTANCE = 0.7
SHOW_DISTANCES = False

NO_RESULTS_MESSAGE = "No relevant information was found in the indexed documents for this question."


class QuestionRequest(BaseModel):
    # Used for the question typed by the user AND for the arguments the LLM sends to the tool
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(
        min_length=1,
        max_length=MAX_QUESTION_LENGTH,
        description="The user's question about the content of the indexed documents.",
    )


class ContextOutput(BaseModel):
    # Used for the plain string the tool returns
    model_config = ConfigDict(extra="forbid")

    context: str = Field(min_length=1, description="Plain text context for the agent.")


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def complete(client: OpenAI, messages: list[dict], tools: list[dict]):
    return client.chat.completions.create(
        messages=messages,
        model=os.getenv("OPEN_ROUTER_MODEL_NAME"),
        tools=tools,
        tool_choice="auto",
    ).choices[0].message


def load_chunks(collection):
    # Reading from the database (Populated by Part1 of the project)
    data = collection.get(include=["documents", "metadatas"])

    positions = {}  # chunk id -> position in the lists below
    for i in range(len(data["ids"])):
        positions[data["ids"][i]] = i

    return data["documents"], data["metadatas"], positions


def rag_search(collection, model, bm25, positions, documents, metadatas, args: str) -> str:
    # ============== Validate the input of the tool ==============
    try:
        request = QuestionRequest.model_validate(json.loads(args))
    except (json.JSONDecodeError, ValidationError) as error:
        return f"error: {error}"

    question = request.question
    question_vector = model.encode(question, normalize_embeddings=True).tolist()

    # ============== Semantic search: keep only the results that are close enough ==============
    semantic_results = collection.query(
        query_embeddings=[question_vector],
        n_results=HYBRID_SEARCH_RESULT_COUNT,
        include=["distances"],
    )
    semantic_ids = semantic_results["ids"][0]
    semantic_distances = semantic_results["distances"][0]

    if SHOW_DISTANCES:
        print("  [distances]", [round(distance, 3) for distance in semantic_distances])

    hybrid_scores = {}
    # Rank 0 receives 1 point, rank 1 receives 1/2 point, and so on.
    rank = 0
    for i in range(len(semantic_ids)):
        if semantic_distances[i] > MAX_DISTANCE:
            continue
        chunk_index = positions[semantic_ids[i]]
        hybrid_scores[chunk_index] = 1 / (rank + 1)
        rank += 1

    # ============== Nothing is close enough in meaning -> nothing usable was found ==============
    if len(hybrid_scores) == 0:
        return NO_RESULTS_MESSAGE

    # ============== Keyword search (BM25) ==============
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
        hybrid_scores[chunk_index] += 1 / (rank + 1)
        rank += 1

    # ============== Combine both rankings and build the context string ==============
    combined_results = []
    for chunk_index, score in hybrid_scores.items():
        combined_results.append((score, chunk_index))
    combined_results.sort(reverse=True)

    context_pieces = []
    for _, chunk_index in combined_results[:HYBRID_SEARCH_RESULT_COUNT]:
        context_pieces.append(
            f"{documents[chunk_index]}\n"
            f"Source: {metadatas[chunk_index]['source']}, page {metadatas[chunk_index]['page']}"
        )
    context = "\n\n".join(context_pieces)

    # ============== Validate the output of the tool ==============
    try:
        return ContextOutput(context=context).context
    except ValidationError as error:
        return f"error: {error}"


def main():
    load_dotenv()
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    )
    model = SentenceTransformer(MODEL_NAME)

    try:
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception as error:
        print(f"Could not open collection '{COLLECTION_NAME}' on {CHROMA_HOST}:{CHROMA_PORT}: {error}")
        print("Check that the Chroma server running")
        return

    documents, metadatas, positions = load_chunks(collection)
    if len(documents) == 0:
        print("The collection is empty. Run the indexing program first.")
        return

    # BM25 needs the tokenized chunks, in the same order as the documents list
    tokenized_documents = []
    for doc in documents:
        tokenized_documents.append(tokenize(doc))
    bm25 = BM25Okapi(tokenized_documents)
    print(f"Loaded {len(documents)} chunks from the database.")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "rag_search",
                "description": "Searches the indexed documents in the database and returns the most relevant "
                               "parts, each followed by its source file and page. "
                               "Use it for any question about the content of the documents.",
                "parameters": QuestionRequest.model_json_schema(),
            },
        }
    ]
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use rag_search for questions about the content of the indexed "
                       "documents, and use it before answering those questions. Answer only from the returned "
                       "context and mention the source file and page you used. If the tool reports that no relevant "
                       "information was found, tell the user that nothing was found. Answer other questions normally.",
        }
    ]

    while True:
        user_text = input("you> ")
        if user_text.strip() == "quit":
            break

        # validate what the user typed, and ask again if it is not valid
        try:
            user_request = QuestionRequest(question=user_text)
        except ValidationError:
            if user_text.strip() == "":
                print("agent> The question is empty. Please type a question.")
            else:
                print(f"agent> The question is too long (maximum {MAX_QUESTION_LENGTH} characters). Please shorten it.")
            continue

        messages.append({"role": "user", "content": user_request.question})
        response = complete(client, messages, tools)
        messages.append(response.model_dump(exclude_none=True))

        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call.function.name == "rag_search":
                    output = rag_search(
                        collection, model, bm25, positions, documents, metadatas, tool_call.function.arguments
                    )
                else:
                    output = f"error: unknown tool {tool_call.function.name}"
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": output})

            response = complete(client, messages, tools)
            messages.append(response.model_dump(exclude_none=True))

        print("agent>", response.content)


if __name__ == "__main__":
    main()
