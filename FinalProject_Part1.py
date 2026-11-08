import re
from pathlib import Path

import chromadb
import pymupdf4llm
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "my_collection"
FILE_PATH = Path(__file__).resolve().parent / "data_sources"
DB_PATH = Path(__file__).resolve().parent / "chroma_db"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


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

    if not documents:
        return documents, metadatas

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


def index_folder(collection, model, folder_path: Path):
    if not folder_path.is_dir():
        print(f"Folder not found: {folder_path}")
        return

    total_files = 0
    total_chunks = 0

    for file_path in folder_path.iterdir():
        if not file_path.is_file():
            continue

        if file_path.suffix != ".pdf":
            print(f"Skipping {file_path.name} as we only index PDF files")
            continue

        documents, metadatas = index_file(collection, model, file_path)
        print(f"Indexed {file_path.name}: {len(documents)} chunks")
        total_files += 1
        total_chunks += len(documents)

    print(f"Done: {total_files} files, {total_chunks} chunks")


def main():
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.HttpClient(host="localhost", port=8000)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space","cosine"}
    )

    index_folder(collection, model, FILE_PATH)


if __name__ == "__main__":
    main()