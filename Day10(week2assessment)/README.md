
# RAG Agent with MCP + LangGraph

A conversational AI agent that retrieves documents from a vector store and evaluates math expressions, powered by MCP tools, LangGraph, and Gemini.

---

## Setup

1. Clone the repo and navigate into it
2. `pip install -r requirements.txt`
3. Add your API key to a `.env` file: `GOOGLE_API_KEY=your_key_here`
4. Place your PDF/text files in the `data/` folder
5. Run the data pipeline: `python data_pipeline.py`
6. Start the agent: `python main.py`

---

## Example Questions

- **"What is the refund policy mentioned in the documents?"**
- **"Explain the onboarding process for new employees."**
- **"What is 15 * 8 + 200?"**


---

## Architecture

```
User Input → LangGraph Agent → MCP Client → MCP Server
                  ↑                              ↓
            Final Answer ←── Tool Result (retrieve / calculate)
```

---

## Live Demo Modifications

| Change | File | What to edit |
|---|---|---|
| Swap top-k | `mcp_server.py` | `search_kwargs={"k": 3}` → any number |
| Add a new tool | `mcp_server.py` | Add a new `@mcp.tool()` function |
| Add a document | `data/` | Drop file in, re-run `python ingest.py` |
