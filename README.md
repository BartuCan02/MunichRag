# 🧠 MunichRAG: A Minimal Retrieval-Augmented Generation Pipeline

This project is a simple but functional **Retrieval-Augmented Generation (RAG)** system built with Python and [SentenceTransformers](https://www.sbert.net/). It allows you to ask questions about local `.md` files (e.g., about Munich, Bavaria, or the Roman Empire), and it returns semantically relevant answers with source citations.

## ✨ Features

- Loads Markdown documents from a local folder (`data/`)
- Splits the content into semantically meaningful text chunks
- Uses a multilingual MiniLM model for semantic search
- Retrieves the top-k relevant passages for a user query
- Displays a simulated answer with full context and source traceability







