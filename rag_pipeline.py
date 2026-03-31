"""
RAG Pipeline Module
Handles query retrieval and LLM response generation
"""
import os
import re
from dotenv import load_dotenv

# Load environment variables before importing clients that depend on them.
load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_sarvam import ChatSarvam
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import chromadb
from chromadb.config import Settings


class FixedChatSarvam(ChatSarvam):
    """Ensure model is always forwarded to Sarvam completions API."""

    def _default_params(self):
        params = super()._default_params()
        params.setdefault("model", self.model_name)
        return params

class RAGPipeline:
    def __init__(self, persist_dir="./chroma_db"):
        """Initialize RAG pipeline with vector DB and LLM"""
        print("Initializing RAG pipeline...")

        api_key = os.getenv("SARVAM_API_KEY")
        if not api_key:
            raise ValueError("SARVAM_API_KEY not found! Check your .env file.")
        
        # Load embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # FIX: Disable ChromaDB telemetry to prevent terminal warnings
        # This removes "Failed to send telemetry event" errors for a clean user experience
        chroma_settings = Settings(anonymized_telemetry=False)
        chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=chroma_settings
        )
        
        # Load vector database with telemetry disabled
        self.vector_db = Chroma(
            client=chroma_client,
            persist_directory=persist_dir,
            embedding_function=self.embedding_model
        )
        
        # Initialize Sarvam AI LLM (cloud-based, optimized for Indic languages)
        # "sarvam-m" is optimized for high-speed conversational tasks
        self.llm = FixedChatSarvam(
            model="sarvam-m",
            sarvam_api_key=api_key,
            temperature=0.1,  # Low temperature for factual campus answers
            streaming=False  # Ensure streaming is off for initial stability
        )
        
        # Custom prompt template for multilingual consistency
        template = """You are an official Campus Assistant AI. Use the provided context to answer the student's question accurately and helpfully.

CRITICAL RULES:
1. Respond in the EXACT SAME LANGUAGE as the user's question (English, Hindi, or any other language)
2. If the answer is not in the context, clearly state "I don't know" or "मुझे नहीं पता" and suggest contacting campus administration
3. Be concise and factual - do not make up information

Context: {context}

Question: {question}

Helpful Answer:"""
        
        qa_prompt = PromptTemplate(
            template=template, 
            input_variables=["context", "question"]
        )
        
        # Create retrieval chain with custom prompt
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": qa_prompt}
        )
        
        print("RAG pipeline ready!")
    
    def _clean_response(self, text):
        """
        FIX: Removes <think> tags and their content from the final output
        Ensures clean user interface without reasoning block display
        """
        # Regular expression to remove everything between <think> tags (case-insensitive)
        cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        return cleaned_text.strip()
    
    def query(self, question):
        """
        Process user query and return CLEANED answer with sources
        """
        result = self.qa_chain({"query": question})
        
        # Apply the fixation here: clean reasoning blocks from output
        final_answer = self._clean_response(result["result"])
        
        return {
            "answer": final_answer,
            "sources": result["source_documents"]
        }

if __name__ == "__main__":
    # Test the pipeline
    pipeline = RAGPipeline()
    response = pipeline.query("What are the scholarship options?")
    print(response["answer"])
