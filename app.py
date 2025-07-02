import streamlit as st
import os
import yaml

from datasources.csv_source import CSVDataSource
from datasources.rdbms_source import RDBMSDataSource
from datasources.hive_source import HiveDataSource
from embeddings.openai_embed import OpenAIEmbedding
from vectordb.chromadb import ChromaDB
from llms.openai_llm import OpenAILLM
from rag.rag_pipeline import RAGPipeline

# Supported options
EMBEDDING_MODELS = {
    'OpenAI': OpenAIEmbedding,
    # Add more embedding models here
}
LLM_MODELS = {
    'OpenAI': OpenAILLM,
    # Add more LLMs here
}
VECTOR_DBS = {
    'ChromaDB': ChromaDB,
    # Add more vector DBs here
}
DATASOURCES = {
    'CSV': CSVDataSource,
    'RDBMS': RDBMSDataSource,
    'Hive': HiveDataSource,
}

def main():
    st.title('Modular RAG Framework')
    st.sidebar.header('Configuration')

    # Data source selection
    datasource_type = st.sidebar.selectbox('Data Source', list(DATASOURCES.keys()))
    ds_params = {}
    if datasource_type == 'CSV':
        csv_path = st.sidebar.text_input('CSV Path', 'transaction_data.csv')
        ds_params = {'path': csv_path}
        datasource = CSVDataSource(**ds_params)
    elif datasource_type == 'RDBMS':
        rdbms_conn = st.sidebar.text_input('RDBMS Connection String')
        ds_params = {'conn_str': rdbms_conn}
        datasource = RDBMSDataSource(**ds_params)
    elif datasource_type == 'Hive':
        hive_conn = st.sidebar.text_input('Hive Connection String')
        ds_params = {'conn_str': hive_conn}
        datasource = HiveDataSource(**ds_params)
    else:
        st.error('Unsupported datasource type')
        return

    # Embedding model selection
    embedding_choice = st.sidebar.selectbox('Embedding Model', list(EMBEDDING_MODELS.keys()))
    embedding_params = {}
    if embedding_choice == 'OpenAI':
        openai_embed_model = st.sidebar.text_input('OpenAI Embedding Model', 'text-embedding-ada-002')
        openai_embed_key = st.sidebar.text_input('OpenAI API Key', type='password', value=os.environ.get('OPENAI_API_KEY', ''))
        embedding_params = {'model': openai_embed_model, 'api_key': openai_embed_key}
        embedder = OpenAIEmbedding(**embedding_params)
    else:
        st.error('Unsupported embedding model')
        return

    # Vector DB selection
    vectordb_choice = st.sidebar.selectbox('Vector DB', list(VECTOR_DBS.keys()))
    vectordb_params = {}
    if vectordb_choice == 'ChromaDB':
        chroma_dir = st.sidebar.text_input('ChromaDB Directory', './chromadb')
        vectordb_params = {'persist_directory': chroma_dir}
        vectordb = ChromaDB(**vectordb_params)
    else:
        st.error('Unsupported vector DB')
        return

    # LLM selection
    llm_choice = st.sidebar.selectbox('LLM', list(LLM_MODELS.keys()))
    llm_params = {}
    if llm_choice == 'OpenAI':
        openai_llm_model = st.sidebar.text_input('OpenAI LLM Model', 'gpt-4o-mini')
        openai_llm_key = st.sidebar.text_input('OpenAI LLM API Key', type='password', value=os.environ.get('OPENAI_API_KEY', ''))
        llm_params = {'model': openai_llm_model, 'api_key': openai_llm_key}
        llm = OpenAILLM(**llm_params)
    else:
        st.error('Unsupported LLM')
        return

    # Build pipeline
    if st.sidebar.button('Build Vector Store'):
        # Update config.yaml with selected options
        config_data = {
            'datasource': {
                'type': datasource_type.lower(),
                'params': ds_params
            },
            'vectordb': {
                'type': vectordb_choice.lower(),
                'params': vectordb_params
            },
            'embedding': {
                'type': embedding_choice.lower(),
                'params': embedding_params
            },
            'llm': {
                'type': llm_choice.lower(),
                'params': llm_params
            }
        }
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        pipeline = RAGPipeline(datasource, embedder, vectordb, llm)
        with st.spinner('Building vector store...'):
            pipeline.build_vector_store()
        st.success('Vector store built!')
        st.session_state['pipeline'] = pipeline

    # Query interface
    pipeline = st.session_state.get('pipeline', None)
    st.header('Ask a Question')
    user_query = st.text_input('Your question:')
    if st.button('Get Answer'):
        if not pipeline:
            st.warning('Please build the vector store first!')
        else:
            with st.spinner('Generating answer...'):
                answer = pipeline.query(user_query)
            st.markdown(f'**Answer:** {answer}')

if __name__ == '__main__':
    main()
