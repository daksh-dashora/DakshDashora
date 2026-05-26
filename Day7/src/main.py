from rag_chain import RAGChain

if __name__ == "__main__":

    rag_chain = RAGChain()
    while True:

        query = input("\nAsk a question (type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        response = rag_chain.invoke(query)
        print("\n--- Response ---\n")
        print(response)