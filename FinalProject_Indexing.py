from pathlib import Path

import chromadb
import pymupdf4llm
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "my_collection"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000

# the folder with the documents, next to this script
FOLDER_PATH = Path(__file__).resolve().parent / "data_sources"
SUPPORTED_EXTENSIONS = (".pdf", ".txt")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        # get a slice of text, using start as the starting index
        chunk = text[start : start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        # make a step: the chunk size minus the overlap
        start = start + CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def read_pages(file_path: Path):
    if file_path.suffix.lower() == ".pdf":
        return pymupdf4llm.to_markdown(str(file_path), page_chunks=True)

    # a .txt file has no pages, so the whole file counts as page 1
    text = file_path.read_text(encoding="utf-8-sig")
    return [{"metadata": {"page_number": 1}, "text": text}]


def index_file(collection, model, file_path: Path):
    pages = read_pages(file_path)

    ids = []
    documents = []
    metadatas = []

    for page in pages:
        page_number = page["metadata"]["page_number"]
        page_text = page["text"]

        chunks = chunk_text(page_text)
        chunk_number = 0

        for chunk in chunks:
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

    # remove the old chunks of this file, so running the program again leaves no old data
    collection.delete(where={"source": file_path.name})

    # if there's a file with no text ->
    if not documents:
        return documents, metadatas

    collection.upsert(
        ids=ids,
        # this is where the embedding happens for each chunk
        # normalize_embeddings scales every vector to length 1
        embeddings=model.encode(documents, normalize_embeddings=True).tolist(),
        documents=documents,
        metadatas=metadatas,
    )

    return documents, metadatas


def index_folder(collection, model, folder_path: Path):
    if not folder_path.is_dir():
        print(f"Folder not found: {folder_path}")
        return

    total_files = 0
    total_chunks = 0

    for file_path in sorted(folder_path.iterdir()):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"Skipping {file_path.name}: only .pdf and .txt files are indexed")
            continue

        try:
            documents, metadatas = index_file(collection, model, file_path)
        except UnicodeDecodeError:
            print(f"Skipping {file_path.name}: it cannot be read as UTF-8 text")
            continue

        if not documents:
            print(f"No text found in {file_path.name}, nothing indexed")
            continue

        print(f"Indexed {file_path.name}: {len(documents)} chunks")
        total_files += 1
        total_chunks += len(documents)

    print(f"Done: {total_files} files, {total_chunks} chunks")


def main():
    # connect first, so a missing server is reported before the model loads
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as error:
        print(f"Could not connect to the Chroma server on {CHROMA_HOST}:{CHROMA_PORT}: {error}")
        return

    model = SentenceTransformer(MODEL_NAME)

    index_folder(collection, model, FOLDER_PATH)


if __name__ == "__main__":
    main()
