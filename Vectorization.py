import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from llama_index.core import VectorStoreIndex

DATA_PATH = 'my_data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Function to load text files from a directory
def load_text_files(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                content = file.read()
                documents.append(content)
    return documents

# Create Vector DB
def CreateVectorDB():
    # Load text files from the specified directory
    documents = load_text_files(DATA_PATH)

    # Split text into chunks for vectorization
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.create_documents(documents)

    # Use Hugging Face embeddings for vectorization
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={"device": "cpu"})
    
    # Create vector database using FAISS
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    print("Vector Database creation in progress.")
    # Create the vector database
    CreateVectorDB()
