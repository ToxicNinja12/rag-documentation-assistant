from os import listdir
from os.path import isfile, join
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

def get_files():
    files = [f for f in listdir("corpus") if isfile(join("corpus", f))]
    return files


def init_ingestion():
    corpus = []
    files = get_files()
    for file in files:
        loader = TextLoader(f"corpus/{file}", encoding="utf-8")
        corpus.extend(loader.load())

    # Embedding model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
    )

    # Chunking Process
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(corpus)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    print("[ALERT] Ingestion successfully completed")
    return vector_store
