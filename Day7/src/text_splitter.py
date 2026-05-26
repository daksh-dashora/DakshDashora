from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents, chunk_size = 500, chunk_overlap = 50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap= chunk_overlap,
        separators = ["\n\n","\n", ".", " ", ""]
        )
    chunks = text_splitter.split_documents(documents)
    print(f"[INFO] splitted documents into {len(chunks)} chunks successfully")
    return chunks

if __name__ == "__main__":
    from pdf_loader import load_pdf

    docs = load_pdf("../data/sample.pdf")
    chunks = split_documents(docs)

    print(f"\nTotal chunks: {len(chunks)}")
    print(f"\n--- Chunk 1 ---")
    print(chunks[0].page_content)
    print(f"\nMetadata: {chunks[0].metadata}")