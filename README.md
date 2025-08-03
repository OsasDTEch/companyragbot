# 🤖 Company RAG Bot

A **Role-Based Access Control (RBAC)** powered **Retrieval-Augmented Generation (RAG)** assistant for intelligent internal document Q&A — built using **LangChain**, **Gemini 1.5 Flash**, and **Streamlit**.

🔗 **Live App**: [https://companyragbot.streamlit.app](https://companyragbot.streamlit.app)

---

## ✨ Features

- 🔍 **RAG Workflow**: Get accurate answers directly from internal documents
- 🔐 **Role-Based Access**: Each user (admin, HR, finance, marketing, etc.) sees only what's relevant
- 🧠 **LLM-Powered**: Uses Google Gemini 1.5 Flash for fast, high-quality responses
- 📂 **Vector Search**: FAISS vector store for fast and efficient retrieval
- 🧱 **Document Types Supported**: PDF, unstructured text, etc.
- 🌐 **Built with**: Streamlit for UI, LangChain for orchestration

---

## 🚀 How It Works

1. **Document Ingestion**  
   Upload internal documents, which are chunked, embedded, and stored in a FAISS vectorstore.

2. **User Role Assignment**  
   Each user is assigned a role (admin, marketing, HR, finance). The system filters documents based on this role.

3. **Ask Questions**  
   Users ask natural language questions and get concise answers generated from documents **relevant to their role only**.

4. **Live Demo**  
   Hosted on [Streamlit Cloud](https://companyragbot.streamlit.app) for easy access and interaction.

---

## 🛠 Tech Stack

| Tool            | Purpose                          |
|-----------------|----------------------------------|
| `LangChain`     | Agent and memory orchestration  |
| `Gemini 1.5`    | LLM response generation         |
| `FAISS`         | Vector store for fast retrieval |
| `Chroma`        | Optional vector store backend   |
| `Streamlit`     | Frontend UI                     |
| `python-dotenv` | Env management                  |
| `pypdf` / `unstructured` | File parsing             |

---

## 📁 Folder Structure

