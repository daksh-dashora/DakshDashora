from sentence_transformers import SentenceTransformer

class EmbeddingModel:

    def __init__(self, model_name = "all-MiniLM-L6-v2" ):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, chunks):
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(texts)
        return embeddings
    
    def embed_query(self, query):
        embedding = self.model.encode(query)
        return embedding

if __name__ == "__main__":

    from pdf_loader import load_all_pdfs
    from chunking import chunk_text

    documents = load_all_pdfs("../data")

    chunks = chunk_text(documents)

    embedding_model = EmbeddingModel()

    embeddings = embedding_model.embed_documents(chunks)

    print(f"\nTotal embeddings: {len(embeddings)}")

    print("\nFirst embedding preview:\n")

    print(embeddings[0][:10])

    print(f"\nEmbedding dimension: {len(embeddings[0])}")

    query_embedding = embedding_model.embed_query(
        "What is retrieval augmented generation?"
    )

    print("\nQuery embedding preview:\n")

    print(query_embedding[:10])