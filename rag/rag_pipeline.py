class RAGPipeline:
    def __init__(self, datasource, embedder, vectordb, llm):
        self.datasource = datasource
        self.embedder = embedder
        self.vectordb = vectordb
        self.llm = llm

    def build_vector_store(self):
        print("[RAGPipeline] Loading documents...")
        docs = self.datasource.load()
        print(f"[RAGPipeline] Loaded {len(docs)} documents: {docs[:2]} ...")
        print("[RAGPipeline] Generating embeddings...")
        embeddings = self.embedder.embed(docs)
        print(f"[RAGPipeline] Generated {len(embeddings)} embeddings. Example: {embeddings[0] if embeddings else 'None'}")
        print("[RAGPipeline] Adding to vector DB...")
        self.vectordb.add(embeddings, docs)
        print("[RAGPipeline] Vector store build complete.")

    def query(self, user_query, top_k=None):
        query_emb = self.embedder.embed([user_query])[0]
        # If top_k is None, use all documents for context
        if top_k is None:
            if hasattr(self.vectordb, 'get_all_documents'):
                all_docs = self.vectordb.get_all_documents()
                context = "\n".join(all_docs)
            else:
                # fallback: try to retrieve a large number
                retrieved_docs = self.vectordb.query(query_emb, top_k=10000)
                context = "\n".join(retrieved_docs)
        else:
            retrieved_docs = self.vectordb.query(query_emb, top_k=top_k)
            context = "\n".join(retrieved_docs)
        prompt = f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"
        return self.llm.generate(prompt)
