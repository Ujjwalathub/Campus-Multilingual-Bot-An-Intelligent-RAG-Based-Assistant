"""
Data Ingestion Module
Processes campus PDFs and text files, creates vector database
"""
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def ingest_documents(data_dir="./data", persist_dir="./chroma_db"):
    """
    Load PDFs from data directory and create vector database
    """
    print("Loading documents...")
    all_docs = []
    
    # Load all PDFs and text files from data directory
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(data_dir, filename)
            print(f"Processing PDF: {filename}")
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            all_docs.extend(docs)
        elif filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            print(f"Processing TXT: {filename}")
            loader = TextLoader(filepath, encoding='utf-8')
            docs = loader.load()
            all_docs.extend(docs)
    
    if not all_docs:
        print("No PDF or TXT files found in data directory!")
        return None
    
    print(f"Loaded {len(all_docs)} pages")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    split_docs = text_splitter.split_documents(all_docs)
    print(f"Split into {len(split_docs)} chunks")
    
    # Create embeddings using multilingual model
    print("Creating embeddings...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # Create and persist vector database
    print("Building vector database...")
    vector_db = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    
    print(f"Vector database created at {persist_dir}")
    return vector_db

if __name__ == "__main__":
    ingest_documents()
