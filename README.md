# 🧠 LITE LEARN – Offline AI Learning Assistant

> An intelligent, fully offline AI-powered learning assistant that helps students understand documents through conversational Q&A, summarization, quizzes, and flashcards.

---

## 📌 About the Project

LITE LEARN is a **Retrieval-Augmented Generation (RAG)** based offline AI assistant built for students who want to learn smarter. Upload any document and interact with it — ask questions, generate summaries, create quizzes, or make flashcards — all without an internet connection.

Built as a final-year B.Tech project at **Muthoot Institute of Technology and Science, Kochi**.

---

## 🎯 Problem It Solves

Students often struggle to extract key information from large PDFs and textbooks. Traditional search is keyword-based and slow. LITE LEARN lets you **have a conversation with your documents** using a locally running LLM — no data leaves your device.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 Document Q&A | Ask any question about your uploaded document |
| 📝 Summarization | Get concise summaries of long documents |
| 🧠 Quiz Generation | Auto-generate MCQs from document content |
| 🃏 Flashcard Creation | Create study flashcards automatically |
| 🔒 100% Offline | No internet required — complete data privacy |
| ⚡ Fast Retrieval | Semantic search using FAISS vector database |

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python |
| LLM | Mistral-7B (Quantized) |
| Embeddings | Sentence-BERT |
| Vector Database | FAISS |
| Backend | Flask |
| Interface | REST API |
| IDE | VS Code / Jupyter Notebook |

---

## 🏗️ System Architecture

```
User Input (Question)
        ↓
Document Loader → Text Chunking
        ↓
Sentence-BERT Embeddings
        ↓
FAISS Vector Store (Indexing & Retrieval)
        ↓
Top-K Relevant Chunks → Context
        ↓
Mistral-7B LLM (Local)
        ↓
Answer / Summary / Quiz / Flashcard
```

---

## 🚀 How to Run

### Prerequisites
- Python 3.9+
- 8GB RAM minimum (16GB recommended for Mistral-7B)

### Installation

```bash
# Clone the repository
git clone https://github.com/Kuriakosesunny/lite-learn-ai-assistant.git
cd lite-learn-ai-assistant

# Install dependencies
pip install -r requirements.txt

# Download the Mistral-7B quantized model
# Place the .gguf file in the /models folder

# Run the application
python app.py
```

### Usage
1. Upload a PDF document via the interface
2. Wait for document indexing (usually under 30 seconds)
3. Start asking questions in natural language
4. Use the menu to switch between Q&A, Summary, Quiz, or Flashcard mode

---

## 📦 Requirements

```
flask
sentence-transformers
faiss-cpu
llama-cpp-python
PyPDF2
numpy
```

---

## 📸 Screenshots

<img width="1914" height="903" alt="Screenshot 2026-04-03 222835" src="https://github.com/user-attachments/assets/7370767b-ea60-4d1d-bbb7-d08bc5c2d947" />
<img width="1919" height="907" alt="Screenshot 2026-04-03 223215" src="https://github.com/user-attachments/assets/bd61dc26-7761-436c-882f-99be2c1b5de5" />
<img width="1918" height="911" alt="Screenshot 2026-04-03 223239" src="https://github.com/user-attachments/assets/47cb1663-215f-4645-99fa-197c33946685" />
<img width="1918" height="908" alt="Screenshot 2026-04-03 223300" src="https://github.com/user-attachments/assets/9ba3fff9-0885-4f3c-887f-09047cd2cbfc" />
<img width="1919" height="904" alt="Screenshot 2026-04-03 223324" src="https://github.com/user-attachments/assets/4bc0c5d4-baca-4c07-be01-47b13cf895c8" />
<img width="1919" height="905" alt="Screenshot 2026-04-03 223346" src="https://github.com/user-attachments/assets/1b0bf3c0-19ea-439f-a75a-a017dc596413" />
<img width="1919" height="904" alt="Screenshot 2026-04-03 223429" src="https://github.com/user-attachments/assets/e3887492-0dc8-4f59-8b3c-c977c4009615" />


---



---

## 📄 License

This project is for educational purposes as part of B.Tech final year project.

---

> *"Making learning accessible, private, and intelligent."* 🚀














