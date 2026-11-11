from pathlib import Path

import chromadb
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "my_collection"

XLSX_FILE = Path(__file__).resolve().parent / "chroma_export.xlsx"

HEADER = ["id", "source", "page", "chunk", "text"]


def load_rows(collection):
    data = collection.get(include=["documents", "metadatas"])

    rows = []
    for i in range(len(data["ids"])):
        metadata = data["metadatas"][i]
        rows.append(
            [
                data["ids"][i],
                metadata["source"],
                metadata["page"],
                metadata["chunk"],
                data["documents"][i],
            ]
        )

    # sort by file, then page, then chunk, so the table is easy to read
    rows.sort(key=lambda row: (row[1], row[2], row[3]))
    return rows


def print_summary(rows):
    # one line per file: how many chunks and which pages
    chunks_per_source = {}
    pages_per_source = {}

    for row in rows:
        source = row[1]
        chunks_per_source[source] = chunks_per_source.get(source, 0) + 1
        pages_per_source.setdefault(source, set()).add(row[2])

    print("Files found in the collection:")
    for source in sorted(chunks_per_source):
        pages = sorted(pages_per_source[source])
        print(f"  {source}: {chunks_per_source[source]} chunks, pages {pages[0]} to {pages[-1]}")


def write_xlsx(rows):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "chunks"

    sheet.append(HEADER)
    for row in rows:
        sheet.append(row)

    # header row: bold
    for cell in sheet[1]:
        cell.font = Font(name="Arial", bold=True)

    # data rows: same font, text column wraps onto several lines
    for sheet_row in sheet.iter_rows(min_row=2):
        for cell in sheet_row:
            cell.font = Font(name="Arial")
            cell.alignment = Alignment(vertical="top", wrap_text=(cell.column == 5))

    sheet.column_dimensions["A"].width = 38
    sheet.column_dimensions["B"].width = 30
    sheet.column_dimensions["C"].width = 8
    sheet.column_dimensions["D"].width = 8
    sheet.column_dimensions["E"].width = 100

    sheet.freeze_panes = "A2"  # header stays visible while scrolling
    sheet.auto_filter.ref = sheet.dimensions  # filter buttons on the header

    workbook.save(XLSX_FILE)


def main():
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as error:
        print(f"Could not open collection '{COLLECTION_NAME}' on {CHROMA_HOST}:{CHROMA_PORT}: {error}")
        print("Is the Chroma server running, and did you run the indexing program?")
        return

    rows = load_rows(collection)
    if not rows:
        print("The collection is empty, nothing to export.")
        return

    print(f"Server: {CHROMA_HOST}:{CHROMA_PORT} | collection: {COLLECTION_NAME} | total chunks: {len(rows)}")
    print_summary(rows)

    write_xlsx(rows)
    print(f"Exported {len(rows)} chunks to {XLSX_FILE.name}")


if __name__ == "__main__":
    main()