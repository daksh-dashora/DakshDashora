def chunk_text(documents, chunk_size = 500, overlap = 100):
    chunks = []

    for doc in documents:
        text = doc["text"]
        start = 0
        while start<len(text):
            end = start+chunk_size
            chunk = text[start:end]
            chunks.append({
                "source": doc["source"],
                "page": doc["page"],
                "text": chunk
            })

            start+= chunk_size - overlap

    return chunks

if __name__ == "__main__":

    from pdf_loader import load_all_pdfs

    documents = load_all_pdfs("../data")

    chunks = chunk_text(documents)

    print(f"\nTotal chunks created: {len(chunks)}")

    print("\n--- First Chunk ---\n")

    print(f"Source: {chunks[0]['source']}")
    print(f"Page: {chunks[0]['page']}")

    print("\nChunk Text:\n")

    print(chunks[0]["text"])

