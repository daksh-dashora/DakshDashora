import os
import sys
from mcp.server.fastmcp import FastMCP
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

mcp = FastMCP("RAG-Utility-Server")

def get_vector_store():
    chroma_dir = "../chroma_db"
    if not os.path.exists(chroma_dir):
        raise FileNotFoundError("Vector store database not found. Please run the data pipeline first.")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=chroma_dir, embedding_function=embeddings)

@mcp.tool()
def retrieve_documents(query: str) -> str:
    try:
        db = get_vector_store()
        retriever = db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)
        
        results = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            idx = doc.metadata.get("chunk_index", "unknown")
            content = doc.page_content.strip()
            results.append(f"[Source: {source} | Chunk: {idx}]\nContent: {content}\n---")
            
        return "\n".join(results) if results else "No matching documents found."
    except Exception as e:
        return f"Error executing retrieval: {str(e)}"

@mcp.tool()
def calculate_expression(expression: str) -> str:
    try:
        allowed_chars = set("0123456789+-*/(). ")
        if not set(expression).issubset(allowed_chars):
            return "Error: Invalid characters detected in mathematical expression."
        
        result = eval(expression, {"__builtins__": None}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating calculation: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")