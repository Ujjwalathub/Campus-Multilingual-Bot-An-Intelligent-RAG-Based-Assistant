# 🎓 Campus Multilingual Bot (IIT Jammu)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg.svg)](https://share.streamlit.io/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Sarvam AI](https://img.shields.io/badge/LLM-Sarvam--m-orange)](https://www.sarvam.ai/)

An intelligent campus assistant built for the **Neural Nexus Hackathon**. This bot uses **Retrieval-Augmented Generation (RAG)** to answer student queries in English and Hindi, pulling real-time data from campus circulars and PDFs.



## ✨ Key Features
* **🌐 Multilingual Support**: Seamlessly switch between English and Hindi queries.
* **🎙️ Voice Input**: Powered by **OpenAI Whisper** for natural voice interactions.
* **📂 PDF Intelligence**: Automatically indexes campus circulars using **ChromaDB**.
* **🇮🇳 Indic LLM**: Integrated with **Sarvam-m** for superior Hindi and "Hinglish" understanding.
* **📍 Source Attribution**: Every answer includes "View Sources" to verify data from original PDFs.
* **🛠️ Human Fallback**: Provides administrative contact details if the answer isn't in the database.

## 🚀 System Architecture
The system utilizes a split-compute model to maximize local hardware (RTX 3050) while leveraging cloud intelligence:
1.  **Ingestion**: PDFs are chunked and embedded using `paraphrase-multilingual-MiniLM-L12-v2`.
2.  **Storage**: Vector embeddings are stored locally in **ChromaDB**.
3.  **Retrieval**: Similarity search finds the top 3 relevant document chunks.
4.  **Generation**: **Sarvam AI** generates the final response in the user's input language.

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/campus-multilingual-bot.git](https://github.com/your-username/campus-multilingual-bot.git)
cd campus-multilingual-bot
