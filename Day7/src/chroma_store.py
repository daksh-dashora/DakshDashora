from langchain_chroma import Chroma
from embedding_model import load_embedding_model

class ChromaVectorStore:

    def __init__(self, persist_directory="../chroma_db", collection_name="rag_collection"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = load_embedding_model()
        self.vector_store = None

    def create_vector_store(self, documents):

        self.vector_store = Chroma.from_documents(
            documents = documents,
            embedding = self.embedding_model,
            persist_directory= self.persist_directory,
            collection_name= self.collection_name
        )

        return self.vector_store

    def load_vector_store(self):

        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model,
            collection_name=self.collection_name
        )

        return self.vector_store
    
    def add_documents(self, documents):

        if self.vector_store is None:
            self.load_vector_store()

        self.vector_store.add_documents(documents)

    def similarity_search(self, query, k=3):

        if self.vector_store is None:
            self.load_vector_store()

        results = self.vector_store.similarity_search(
            query = query,
            k = k
        )

        return results
    
    def as_retriever(self, k=3):

        if self.vector_store is None:
            self.load_vector_store()

        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )

        return retriever
    
if __name__ == "__main__":

    from pdf_loader import load_pdf
    from text_splitter import split_documents

    docs = load_pdf("../data/sample.pdf")

    chunks = split_documents(docs)

    chroma_store = ChromaVectorStore()

    chroma_store.create_vector_store(chunks)

    print("\nVector DB created successfully!")

    print(
        f"\nTotal vectors stored: "
        f"{chroma_store.vector_store._collection.count()}"
    )

    retriever = chroma_store.as_retriever(k=3)

    results = retriever.invoke(
        "What is the document about?"
    )

    print("\n--- Retrieved Chunk ---\n")

    print(results[0].page_content)

    print("\nMetadata:\n")

    print(results[0].metadata)
