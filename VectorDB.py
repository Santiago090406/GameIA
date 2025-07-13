from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function
import os

CHROMA_PATH = "chroma"
DATA_DIR = "data"

def create_vector_db():
    embedding_function = get_embedding_function()
    documents = []

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_DIR, filename))
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    db = Chroma.from_documents(
        texts,
        embedding=embedding_function,
        persist_directory=CHROMA_PATH
    )

    db.persist()
    print("✅ Base de datos vectorial creada.")

if __name__ == "__main__":
    create_vector_db()
