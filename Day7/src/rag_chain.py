from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from chroma_store import ChromaVectorStore
from groq_model import load_llm


class RAGChain:

    def __init__(self):

        self.vector_store = ChromaVectorStore()
        self.vector_store.load_vector_store()
        self.retriever = self.vector_store.as_retriever(k=3)

        self.llm = load_llm()
        self.chat_history = InMemoryChatMessageHistory()
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                 """
You are a helpful document assistant.

Answer ONLY from the provided context.

If answer is not present in context, say:
"I could not find the answer in the document."
"""
            ), 
            MessagesPlaceholder(
                variable_name = "chat_history"
            ),
            (
                "human",
                 """
Context:
{context}

Question:
{question}
"""
            )
        ])

        self.chain = (
            {
                "context": self.retriever,
                "question": RunnablePassthrough(),
                "chat_history" : lambda _ : self.chat_history.messages  
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def invoke(self, query: str):
        response = self.chain.invoke(query)
        self.chat_history.add_user_message(query)
        self.chat_history.add_ai_message(response)

        return response


