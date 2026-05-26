import os

from dotenv import load_dotenv

from langchain_groq import ChatGroq


load_dotenv()


def load_llm(
    model_name: str = "llama-3.1-8b-instant",
    temperature: float = 0
):

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name=model_name,
        temperature=temperature
    )

    return llm


if __name__ == "__main__":

    llm = load_llm()

    response = llm.invoke(
        "Explain RAG in one sentence."
    )

    print("\n--- Response ---\n")

    print(response.content)