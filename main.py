import yaml
import os
os.environ["CHROMA_TELEMETRY_ENABLED"] = "FALSE"

from datasources.base import DataSource
from embeddings.base import EmbeddingModel
from vectordb.base import VectorDB
from llms.base import LLM
from rag.rag_pipeline import RAGPipeline

# Placeholder imports for demonstration. Replace with actual implementations.


# Example imports for different data sources
from datasources.csv_source import CSVDataSource  # File-based
from datasources.rdbms_source import RDBMSDataSource  # RDBMS
from datasources.hive_source import HiveDataSource  # Hive
# Embedding, VectorDB, LLM imports remain as before
from embeddings.openai_embed import OpenAIEmbedding  # Example
from vectordb.chromadb import ChromaDB  # Example
from llms.openai_llm import OpenAILLM  # Example

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)


# DataSource selection
ds_type = config['datasource']['type']
ds_params = config['datasource']['params']
if ds_type == 'csv':
    datasource = CSVDataSource(**ds_params)
elif ds_type == 'rdbms':
    datasource = RDBMSDataSource(**ds_params)
elif ds_type == 'hive':
    datasource = HiveDataSource(**ds_params)
else:
    raise ValueError(f"Unsupported datasource type: {ds_type}")


# Embedding
embedding_params = config['embedding']['params'].copy()
if embedding_params.get('api_key', '').upper().startswith('PULL FROM ENVIRONMENT'):
    embedding_params['api_key'] = os.environ.get('OPENAI_API_KEY')
if config['embedding']['type'] == 'openai':
    embedder = OpenAIEmbedding(**embedding_params)
# Add more embedding types as needed

# VectorDB
if config['vectordb']['type'] == 'chromadb':
    vectordb = ChromaDB(**config['vectordb']['params'])
# Add more vector dbs as needed


# LLM
llm_params = config['llm']['params'].copy()
if llm_params.get('api_key', '').upper().startswith('PULL FROM ENVIRONMENT'):
    llm_params['api_key'] = os.environ.get("OPENAI_API_KEY")
if config['llm']['type'] == 'openai':
    llm = OpenAILLM(**llm_params)
# Add more LLMs as needed

# Build and run RAG
try:
    print(datasource)
    pipeline = RAGPipeline(datasource, embedder, vectordb, llm)
    print("post RAGPipeline")
    pipeline.build_vector_store()
    print("post pipeline build vector")
except Exception as e:
    print("Error during pipeline build:", e)
    import traceback; traceback.print_exc()
print("RAG pipeline is ready. You can now ask questions.")
try:
    while True:
        user_query = input("Ask a question (or 'exit'): ")
        if user_query.lower() == 'exit':
            break
        answer = pipeline.query(user_query)
        print("Answer:", answer)
except KeyboardInterrupt:
    print("\nExiting gracefully.")
