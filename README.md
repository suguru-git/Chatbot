# 📚 PDF/YouTube Transcript Chatbot 🤖

A Python chatbot that can answer questions from **PDF documents** and **YouTube video transcripts** using AI (powered by Ollama).

---
📁 Files
- File Name  	Description
- mybot.py  	ollama mistral version
- newbot.py	  ollama  phi version
- Knowledge_base.pkl   chunks+embeddings
- conversation_history.pkl   conversation history


## 🚀 What It Does

- 📄 Upload PDF or text files and ask questions about them  
- 📺 Fetch YouTube video transcripts and chat about the content  
- 🧠 Uses AI to find accurate, context-aware answers from your documents  
- 💾 Saves conversation history for easy review  

---

## 🔧 Quick Start

### 📥 Download the Project  Folder
### 🛠 Requirements  
- Python 3.8+  
- [Ollama](https://ollama.ai) (used for AI responses)

### 📦 Install Dependencies

```bash
pip install sentence-transformers scikit-learn numpy PyPDF2 youtube-transcript-api requests

⚙️ Setup Ollama
Install Ollama from ollama.ai

Run this command
ollama pull phi

Start Ollama:
ollama serve

▶️ Run the Chatbot
python chatbot.py

Features
✅ PDF and TXT file support

✅ YouTube transcript extraction

✅ AI-powered question answering

✅ Conversation history

✅ Offline-friendly (after setup)

Team Members
Sivaguru
Dilipkumar
