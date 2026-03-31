"""
Streamlit Frontend for Campus Multilingual Bot
"""
import streamlit as st
import os
import re
import time
from rag_pipeline import RAGPipeline
from audio_recorder_streamlit import audio_recorder
import whisper
import numpy as np
import io
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Campus Multilingual Bot",
    page_icon="🎓",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pipeline" not in st.session_state:
    if os.path.exists("./chroma_db"):
        with st.spinner("Loading RAG pipeline..."):
            st.session_state.pipeline = RAGPipeline()
    else:
        st.session_state.pipeline = None

if "whisper_model" not in st.session_state:
    st.session_state.whisper_model = None

if "query_times" not in st.session_state:
    st.session_state.query_times = []

if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

# Header
st.title("🎓 Campus Multilingual Bot")
st.caption("Ask questions in English or Hindi about fees, scholarships, schedules, and campus services")

# Language indicator
col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    st.markdown("**🇬🇧 English**")
with col2:
    st.markdown("**🇮🇳 हिंदी**")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This intelligent campus assistant uses RAG (Retrieval-Augmented Generation) to answer queries in multiple languages.")
    
    st.markdown("---")
    st.subheader("✨ Features")
    st.markdown("""
    - 🌐 Multilingual support (EN/HI)
    - 🎤 Voice input powered by Whisper
    - 📚 Context-aware responses
    - 🔍 Source citation
    - 🤝 Human fallback
    """)
    
    st.markdown("---")
    
    if st.session_state.pipeline is None:
        st.error("⚠️ Vector database not found")
        st.info("Run: `python ingest.py` in your terminal to initialize the system")
    else:
        st.success("✅ System ready")
        st.caption("Model: Sarvam-2B (Cloud) + Multilingual Embeddings")
    
    # Performance Metrics
    st.markdown("---")
    st.subheader("📊 Session Stats")
    st.metric("Total Queries", st.session_state.total_queries)
    
    if st.session_state.query_times:
        avg_time = sum(st.session_state.query_times) / len(st.session_state.query_times)
        st.metric("Avg Response Time", f"{avg_time:.2f}s")
    
    st.markdown("---")
    
    # Voice Input Toggle
    use_voice = st.checkbox("🎤 Enable Voice Input", value=False)
    
    if use_voice and st.session_state.whisper_model is None:
        with st.spinner("Loading Whisper model (one-time)..."):
            try:
                st.session_state.whisper_model = whisper.load_model("base")
                st.success("✅ Voice input ready!")
            except Exception as e:
                st.error(f"Failed to load Whisper: {e}")
                use_voice = False
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.query_times = []
        st.session_state.total_queries = 0
        st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📄 View Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.text(f"Source {i}: {source.metadata.get('source', 'Unknown')}")
                    st.text(f"Page: {source.metadata.get('page', 'N/A')}")
                    st.text(f"Content: {source.page_content[:200]}...")
                    st.divider()

# Voice Input Section
transcribed_text = None
if 'use_voice' in locals() and use_voice:
    st.markdown("### 🎤 Voice Input")
    st.caption("Click to record your question in English or Hindi")
    
    audio_bytes = audio_recorder(
        text="Click to record",
        recording_color="#e74c3c",
        neutral_color="#3498db",
        icon_name="microphone",
        icon_size="2x"
    )
    
    if audio_bytes and st.session_state.whisper_model:
        with st.spinner("Transcribing your voice..."):
            try:
                # Convert audio bytes to numpy array
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Transcribe
                result = st.session_state.whisper_model.transcribe(audio_data, language=None, fp16=False)
                transcribed_text = result["text"]
                
                st.success(f"✅ Transcribed: {transcribed_text}")
                
            except Exception as e:
                st.error(f"Transcription failed: {e}")

# Chat input (text or voice)
prompt = None
if transcribed_text:
    prompt = transcribed_text
else:
    prompt = st.chat_input("Ask about scholarships, fees, or schedules... (या हिंदी में पूछें)")

if prompt:
    if st.session_state.pipeline is None:
        st.error("Please run `python ingest.py` first to create the vector database.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start_time = time.time()
                response = st.session_state.pipeline.query(prompt)
                end_time = time.time()
                
                response_time = end_time - start_time
                st.session_state.query_times.append(response_time)
                st.session_state.total_queries += 1
                
                # Secondary safeguard: ensure no residual think tags are displayed
                display_text = re.sub(r'<think>.*?</think>', '', response["answer"], flags=re.DOTALL | re.IGNORECASE).strip()
                st.markdown(display_text)
                st.caption(f"⏱️ Response time: {response_time:.2f}s")
                
                # Human Fallback Detection
                uncertain_phrases = ["don't know", "नहीं पता", "i don't know", "मुझे नहीं पता"]
                if any(phrase in response["answer"].lower() for phrase in uncertain_phrases):
                    st.warning("⚠️ This query might require human assistance.")
                    st.info("📧 Need more help? Contact **Campus Support**: [support@iitjammu.ac.in](mailto:support@iitjammu.ac.in) or visit the Registrar Office.")
                
                # Show sources
                with st.expander("📄 View Sources"):
                    for i, source in enumerate(response["sources"], 1):
                        st.text(f"Source {i}: {source.metadata.get('source', 'Unknown')}")
                        st.text(f"Page: {source.metadata.get('page', 'N/A')}")
                        st.text(f"Content: {source.page_content[:200]}...")
                        st.divider()
        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"],
            "response_time": response_time
        })
