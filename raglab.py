from pathlib import Path
'''
def load_and_chunk(folder="documents", chunk_size=700, overlap=100):
    records = []
    for path in Path(folder).glob("*.txt"):
        text = path.read_text(encoding="utf-8").strip()
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                records.append({
                    "id": f"{path.name}-{start}",
                    "text": chunk,
                    "source": path.name,
                    "start": start
                })
            if end == len(text):
                break
            start = end - overlap
    return records

chunks = load_and_chunk()
print("Number of chunks:", len(chunks))
for item in chunks:
    print("\nSOURCE:", item["source"], "| ID:", item["id"])
    print(item["text"][:180], "...")
'''

def load_and_chunk(folder="documents", size=700, overlap=100):
    chunks = []

    for file in Path(folder).glob("*.txt"):
        text = file.read_text(encoding="utf-8").strip()

        for start in range(0, len(text), size - overlap):
            end = min(start + size, len(text))
            chunk = text[start:end].strip()

            chunks.append({
                "id": f"{file.name}-{start}",
                "text": chunk,
                "source": file.name,
                "start": start
            })

    return chunks


# Run
chunks = load_and_chunk()
print("Total chunks:", len(chunks))

for c in chunks:
    print("\nSOURCE:", c["source"], "| ID:", c["id"])
    print(c["text"][:180], "...")

