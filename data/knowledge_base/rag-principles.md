# RAG principles

Retrieval-Augmented Generation separates information retrieval from answer generation. The pipeline indexes content, splits it into overlapping chunks, computes embeddings, and stores vectors and metadata. At query time, the system searches for the most similar chunks, passes them to the language model, and returns the sources.

A reliable RAG system should validate source quality, avoid answering when context is insufficient, preserve verifiable citations, and measure latency, coverage, and retrieval quality.
