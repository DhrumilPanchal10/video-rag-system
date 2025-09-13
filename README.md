# 🎥 Real-Time Adaptive RAG System for Long-Form Video QA

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Intelligent Question Answering for Long-Form Video Content**  
> *Advanced RAG system that processes, indexes, and answers questions about hour-long YouTube videos in real-time*

![Video RAG System Demo](https://via.placeholder.com/800x400.png?text=Video+RAG+System+Demo+Placeholder)
*Replace with your actual demo GIF/screenshot*

## 🚀 Features

- **🎬 Video Processing**: Automatic YouTube video download & Whisper transcription
- **🧠 Semantic Chunking**: Intelligent content segmentation using spaCy NLP
- **🔍 Multi-Vector Retrieval**: FAISS-based semantic search with metadata
- **🎯 Smart Reranking**: Cross-encoder reranking for precision relevance
- **💬 LLM Integration**: Groq-powered answers with timestamp citations
- **⚡ Real-Time Performance**: <5 second query responses on long videos
- **🌐 Web Interface**: Streamlit frontend with intuitive UI
- **📊 Production Ready**: FastAPI backend, Docker support, comprehensive API

## 🏗️ System Architecture

![System Architecture](./RAG DIagram.png)

