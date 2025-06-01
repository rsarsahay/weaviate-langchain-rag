# Weaviate-LangChain-RAG

## Overview

This project implements a robust Retrieval-Augmented Generation (RAG) pipeline using Weaviate as a vector database, LangChain for orchestration, and Ollama for both embedding generation and LLM inference. The architecture is designed for scalable, production-grade document ingestion and semantic search, enabling precise question answering over large unstructured text corpora.

---

## Architecture

- **Document Ingestion**: Text files are preprocessed, chunked, and embedded using Ollama's embedding models. Chunks and their embeddings are stored in Weaviate for efficient vector search.
- **Vector Database**: Weaviate serves as the core vector store, supporting hybrid search (dense + keyword) and scalable retrieval.
- **RAG Pipeline**: At query time, user questions are embedded and used to retrieve relevant document chunks from Weaviate. Retrieved context is passed to an LLM (via LangChain and Ollama) for answer generation.
- **Orchestration**: LangChain manages prompt templating and the end-to-end flow between retrieval and generation.

---

## Components

- `ingest.py`: Handles document loading, chunking, embedding, and ingestion into Weaviate.
- `query.py`: Implements the RAG pipeline, including hybrid search and LLM-based answer generation.
- `docker-compose.yml`: Provides a reproducible environment for Weaviate deployment.
- `data/`: Directory containing source text files for ingestion (user-provided, create if not present).

---

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repo-url>
   cd weaviate-langchain-rag
   ```

2. **Start Weaviate**
   ```bash
   docker-compose up -d
   ```

3. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Data**
   - Place your `.txt` files in the `data/` directory (create this directory if it does not exist).

5. **Ingest Documents**
   ```bash
   python ingest.py
   ```

6. **Run the RAG Query Pipeline**
   ```bash
   python query.py
   ```

---

## Configuration

- **Weaviate**: Configured for local deployment with anonymous access and persistent storage.
- **Embeddings & LLM**: Uses Ollama's `nomic-embed-text:latest` for embeddings and `granite3.3:2b` for LLM inference. Ensure Ollama is running and models are available locally.

---

## Extensibility

- **Model Swapping**: Easily switch embedding or LLM models by updating the model names in the code.
- **Scalability**: The architecture supports scaling to large document sets and can be containerized for production deployment.
- **Custom Retrieval**: Hybrid search parameters (e.g., `alpha`, `limit`) are configurable for optimal retrieval performance.

---

## Best Practices

- Modular codebase for maintainability and extensibility.
- Clear separation of concerns between ingestion, retrieval, and generation.
- Uses industry-standard libraries and patterns for vector search and RAG.

---

## License

This project is provided for educational and research purposes. For production use, review and comply with the licenses of all dependencies.

---

## Contact

For questions or contributions, please open an issue or submit a pull request.
