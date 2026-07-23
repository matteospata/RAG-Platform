# Data contracts

Every record entering the platform is normalized into a `SourceDocument` with:

- a stable `document_id`, derived from the source and content hash;
- a `content_hash` for idempotent indexing;
- traceable title, text, and metadata;
- an ingestion timestamp.

Chunks always retain a reference to the original document. In each answer, the retriever exposes identifiers such as `[S1]` and `[S2]` so that claims can be verified.
