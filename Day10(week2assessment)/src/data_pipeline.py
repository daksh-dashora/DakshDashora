import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def main():
    data_dir = "../data"
    chroma_dir = "../chroma_db"
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created '{data_dir}' directory. Please drop your PDF files there and re-run.")
        return

    loader = DirectoryLoader(data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    if not documents:
        print("No PDF documents found in the data directory.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    for idx, chunk in enumerate(chunks):
        source_file = os.path.basename(chunk.metadata.get("source", "unknown"))
        chunk.metadata["source"] = source_file
        chunk.metadata["chunk_index"] = idx

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir
    )
    print(f"Successfully processed {len(chunks)} chunks locally and saved to {chroma_dir}")

if __name__ == "__main__":
    main()