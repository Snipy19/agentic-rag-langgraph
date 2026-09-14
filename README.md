# Agentic RAG and RAG Evaluation with LangGraph and LangSmith

This repository contains two notebooks that explore building and evaluating Retrieval-Augmented Generation (RAG) systems using LangGraph and LangSmith.

## Contents

### 1. `agentic_rag.ipynb`
An implementation of an Agentic RAG pipeline using LangGraph.

- Builds a stateful agent graph with `StateGraph` from LangGraph
- Uses a conditional routing node that decides whether retrieval is required for a given question
- Retrieves relevant context from a FAISS vector store built with OpenAI embeddings
- Generates a final answer using `ChatOpenAI` (GPT-4.1), grounded in the retrieved documents
- Demonstrates the full graph execution flow: decide → retrieve (if needed) → generate

### 2. `rag_evaluation.ipynb`
A workflow for evaluating RAG applications using LangSmith.

- Creates a test dataset of questions and expected answers
- Runs a RAG chatbot (built over Lilian Weng's blog posts) against the dataset
- Evaluates the responses using multiple metrics:
  - Correctness
  - Groundedness
  - Answer relevance
  - Retrieval relevance
- Uses LangSmith's `evaluate` API to run experiments and inspect results as a pandas DataFrame

## Tech Stack

- Python
- LangGraph
- LangChain / LangChain Community
- OpenAI (GPT-4.1, embeddings)
- FAISS (vector store)
- LangSmith (evaluation and tracing)

## Setup

1. Clone the repository
   ```bash
   git clone <repo-url>
   cd agentic-rag-langgraph
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with the following keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   LANGSMITH_API_KEY=your_langsmith_api_key
   ```

4. Run the notebooks
   ```bash
   jupyter notebook
   ```

## Notes

- The vector store in `agentic_rag.ipynb` uses a small set of sample documents for demonstration purposes and can be swapped out for any custom document set.
- The evaluation workflow in `rag_evaluation.ipynb` requires a LangSmith account and API key. Free-tier accounts have monthly trace limits, which may need to be considered when running large evaluation batches.

## License

This project is for personal learning and reference purposes.
