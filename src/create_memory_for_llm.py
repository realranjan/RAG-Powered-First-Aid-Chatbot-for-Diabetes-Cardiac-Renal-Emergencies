from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import sys

## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


# Step 1: Load raw PDF(s) and Excel file(s)
DATA_PATH = "data/"

def load_documents(data):
    documents = []
    # Load PDFs
    for file in os.listdir(data):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data, file))
            documents.extend(loader.load())
        elif file.endswith(".xlsx"):
            loader = UnstructuredExcelLoader(os.path.join(data, file))
            documents.extend(loader.load())
    return documents

documents = load_documents(DATA_PATH)
print("Loaded documents:", len(documents))
if not documents:
    print("No PDF or Excel documents found in the data directory. Exiting.")
    sys.exit(1)


# Step 2: Create Chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(extracted_data=documents)
print("Created text chunks:", len(text_chunks))
if not text_chunks:
    print("No text chunks created. Exiting.")
    sys.exit(1)

# Step 3: Create Vector Embeddings 

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()

# Step 4: Store embeddings in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)