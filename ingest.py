import os
import chromadb
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
import toml

def load_config():
    # Baca file toml
    with open("openai_config.toml", "r") as file:
        config_data = toml.load(file)
    return config_data

def create_vector_db():
    # Baca konfigurasi dari file toml
    config_data = load_config()
    
    pdfs = PyPDFDirectoryLoader('./')
    data = pdfs.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=100
    )

    texts = text_splitter.split_documents(data)

    persist_directory = 'db'
    if not os.path.exists(persist_directory):
        os.mkdir(persist_directory)

    embeddings = OpenAIEmbeddings(
        openai_api_key=config_data["api"]["openai_api_key"]
    )
    print(embeddings)

    client_settings = chromadb.config.Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=persist_directory,
        anonymized_telemetry=False
    )

    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        collection_name='pdf_data',
        client_settings=client_settings,
        persist_directory=persist_directory
    )
    vectordb.persist()

    return vectordb
