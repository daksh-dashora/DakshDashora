from langchain_huggingface import HuggingFaceEmbeddings

def load_embedding_model(model_name = "nomic-ai/nomic-embed-text-v1.5"):
    print(f"[EMBEDDING MODEL] loading model: {model_name}")

    embeddings = HuggingFaceEmbeddings(
        model_name = model_name,
        model_kwargs={"trust_remote_code" : True}

    )

    print("[EMBEDDING_MODEL] Model loaded successfully")
    return embeddings


if __name__ == "__main__":

    embeddings = load_embedding_model()

    test_sentence = "LangChain makes building RAG pipelines easier."

    vector = embeddings.embed_query(test_sentence)

    print(f"\nSentence: {test_sentence}")

    print(f"Vector dimensions: {len(vector)}")

    print(f"First 5 values: {vector[:5]}")