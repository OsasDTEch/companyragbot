# main.py - Complete RAG-based RBAC System

import os
import streamlit as st
from typing import List, Dict, Optional
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import hashlib
# llm_setup.py
import os

from dotenv import load_dotenv

load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
# Configuration
BASE_DOC_PATH = './docs'
BASE_DB = './db'
EMBEDDING_MODEL = 'mxbai-embed-large'
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# You must have your GOOGLE_API_KEY set as an environment variable
embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


# Initialize embeddings and LLM



# User Authentication and Role Management
class UserAuth:
    """Handle user authentication and role management"""

    USER_CREDENTIALS = {
        "jane_hr": {"password": "hr123", "name": "Jane Smith", "title": "HR Manager"},
        "mike_fin": {"password": "fin123", "name": "Mike Johnson", "title": "Finance Director"},
        "emma_mkt": {"password": "mkt123", "name": "Emma Davis", "title": "Marketing Manager"},
        "engr": {"password": "eng123", "name": "Alex Wilson", "title": "Senior Engineer"},
        "admin": {"password": "admin123", "name": "Peter Pandey", "title": "CEO"},
        "employee": {"password": "emp123", "name": "General User", "title": "Employee"}
    }

    USER_ROLE_MAP = {
        "jane_hr": ["hr"],
        "mike_fin": ["finance"],
        "emma_mkt": ["marketing"],
        "engr": ["engineering"],
        "admin": ["general", "marketing", "finance", "hr", "engineering"],  # C-Level access
        "employee": ["general"]  # Employee level access
    }

    @staticmethod
    def authenticate(username: str, password: str) -> bool:
        """Authenticate user credentials"""
        if username in UserAuth.USER_CREDENTIALS:
            return UserAuth.USER_CREDENTIALS[username]["password"] == password
        return False

    @staticmethod
    def get_user_roles(username: str) -> List[str]:
        """Get user's accessible roles/departments"""
        return UserAuth.USER_ROLE_MAP.get(username, [])

    @staticmethod
    def get_user_info(username: str) -> Dict:
        """Get user information"""
        if username in UserAuth.USER_CREDENTIALS:
            user_data = UserAuth.USER_CREDENTIALS[username].copy()
            user_data['roles'] = UserAuth.get_user_roles(username)
            return user_data
        return {}


class DocumentProcessor:
    """Handle document loading and processing"""

    DEPARTMENTS = ['hr', 'engineering', 'finance', 'general', 'marketing']

    @staticmethod
    def load_and_split_documents(dept: str) -> List:
        """Load documents from docs folder and split them into chunks"""
        all_docs = []
        folder_path = f'{BASE_DOC_PATH}/{dept}'

        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist!")
            return []

        files_in_dir = os.listdir(folder_path)
        print(f"Processing {dept}: {files_in_dir}")

        if dept == 'hr':
            # Handle CSV files for HR department
            for file in files_in_dir:
                if file.endswith('.csv'):
                    file_path = os.path.join(folder_path, file)
                    try:
                        loader = CSVLoader(file_path)
                        docs = loader.load()
                        all_docs.extend(docs)
                        print(f"Loaded {len(docs)} documents from {file}")
                    except Exception as e:
                        print(f"Error loading {file}: {e}")
        else:
            # Handle markdown and text files for other departments
            try:
                loader = DirectoryLoader(
                    folder_path,
                    glob='**/*.md',
                    loader_cls=TextLoader,
                    loader_kwargs={'encoding': 'utf-8'}
                )
                docs = loader.load()
                all_docs.extend(docs)

                if not docs:
                    # Fallback to all text files
                    loader = DirectoryLoader(
                        folder_path,
                        glob='**/*',
                        loader_cls=TextLoader,
                        loader_kwargs={'encoding': 'utf-8'},
                        silent_errors=True
                    )
                    docs = loader.load()
                    all_docs.extend(docs)

            except Exception as e:
                print(f"Error loading documents from {dept}: {e}")
                return []

        if not all_docs:
            return []

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(all_docs)
        return chunks

    @staticmethod
    def setup_vector_stores():
        """Initialize vector stores for all departments"""
        for dept in DocumentProcessor.DEPARTMENTS:
            chunks = DocumentProcessor.load_and_split_documents(dept)
            if chunks:
                persist_directory = f'{BASE_DB}/{dept}'
                os.makedirs(persist_directory, exist_ok=True)

                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding,
                    persist_directory=persist_directory
                )
                print(f'✓ Setup vector store for {dept}: {len(chunks)} chunks')


class RAGChatbot:
    """RAG-based chatbot with role-based access control"""

    def __init__(self):
        self.vector_stores = {}
        self.qa_chains = {}
        self._load_vector_stores()
        self._setup_qa_chains()

    def _load_vector_stores(self):
        """Load existing vector stores"""
        for dept in DocumentProcessor.DEPARTMENTS:
            persist_directory = f'{BASE_DB}/{dept}'
            if os.path.exists(persist_directory):
                try:
                    self.vector_stores[dept] = Chroma(
                        persist_directory=persist_directory,
                        embedding_function=embedding
                    )
                    print(f"✓ Loaded existing vector store for {dept}")
                except Exception as e:
                    print(f"Error loading vector store for {dept}: {e}")
            else:
                print(f"⚠️  No vector store found for {dept} at {persist_directory}")

    def _setup_qa_chains(self):
        """Setup QA chains for each department"""
        prompt_template = """
        You are an AI assistant for the company, powered by Osass. Use the following context to answer the question.
        Be helpful, accurate, and professional. Always mention the source of information when possible.

        Context: {context}

        Question: {question}

        Answer: """

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        for dept, vectorstore in self.vector_stores.items():
            self.qa_chains[dept] = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )

    def query(self, question: str, user_roles: List[str]) -> Dict:
        """Process user query based on their roles"""
        if not user_roles:
            return {
                "answer": "Access denied. No roles assigned to your account.",
                "sources": [],
                "accessible_departments": []
            }

        # Search across all accessible departments
        all_results = []
        sources = []

        for role in user_roles:
            if role in self.qa_chains:
                try:
                    result = self.qa_chains[role]({"query": question})
                    all_results.append({
                        "department": role,
                        "answer": result["result"],
                        "source_docs": result["source_documents"]
                    })

                    # Extract source information
                    for doc in result["source_documents"]:
                        source_info = {
                            "department": role,
                            "source": doc.metadata.get("source", "Unknown"),
                            "content_preview": doc.page_content[:100] + "..."
                        }
                        sources.append(source_info)

                except Exception as e:
                    print(f"Error querying {role}: {e}")

        if not all_results:
            return {
                "answer": "I couldn't find relevant information in your accessible departments. Please contact your administrator if you need access to additional resources.",
                "sources": [],
                "accessible_departments": user_roles
            }

        # Combine results from multiple departments
        combined_answer = self._combine_results(all_results, question)

        return {
            "answer": combined_answer,
            "sources": sources,
            "accessible_departments": user_roles
        }

    def _combine_results(self, results: List[Dict], question: str) -> str:
        """Combine results from multiple departments into a coherent answer"""
        if len(results) == 1:
            return results[0]["answer"]

        # For multiple departments, create a comprehensive response
        combined = f"Based on your access to multiple departments, here's what I found:\n\n"

        for result in results:
            if result["answer"] and "I don't have information" not in result["answer"]:
                combined += f"**{result['department'].title()} Department:**\n"
                combined += f"{result['answer']}\n\n"

        return combined


# Streamlit Application
def main():
    st.set_page_config(
        page_title="Company RAG Chatbot - Osass",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 Company RAG Chatbot")
    st.subheader("Role-Based Access Control System - Powered by Osass")

    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        with st.spinner("Initializing RAG system..."):
            st.session_state.chatbot = RAGChatbot()

    # Authentication
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.subheader("🔐 Login")

        col1, col2 = st.columns([1, 2])

        with col1:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Login"):
                if UserAuth.authenticate(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.user_info = UserAuth.get_user_info(username)
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")

        with col2:
            st.info("""
            **Demo Accounts:**
            - **jane_hr** / hr123 (HR Manager)
            - **mike_fin** / fin123 (Finance Director)  
            - **emma_mkt** / mkt123 (Marketing Manager)
            - **engr** / eng123 (Senior Engineer)
            - **admin** / admin123 (CEO - Full Access)
            - **employee** / emp123 (General Employee)
            """)

    else:
        # Main Chat Interface
        user_info = st.session_state.user_info

        # Sidebar with user info
        with st.sidebar:
            st.subheader("👤 User Profile")
            st.write(f"**Name:** {user_info['name']}")
            st.write(f"**Title:** {user_info['title']}")
            st.write(f"**Username:** {st.session_state.username}")
            st.write(f"**Access Level:** {', '.join(user_info['roles']).title()}")

            if st.button("Logout"):
                for key in ['authenticated', 'username', 'user_info', 'messages']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

        # Chat Interface
        st.subheader(f"💬 Hello {user_info['name']}! How can I help you today?")

        # Initialize chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 Sources"):
                        for source in message["sources"]:
                            st.write(f"**{source['department'].title()}:** {source['source']}")
                            st.caption(source['content_preview'])

        # Chat input
        if prompt := st.chat_input("Ask me anything about the company..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Searching company knowledge base..."):
                    result = st.session_state.chatbot.query(prompt, user_info['roles'])

                    st.markdown(result["answer"])

                    if result["sources"]:
                        with st.expander("📚 Sources"):
                            for source in result["sources"]:
                                st.write(f"**{source['department'].title()}:** {source['source']}")
                                st.caption(source['content_preview'])

                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"]
                    })


if __name__ == "__main__":
    # Only setup vector stores if they don't exist
    if not os.path.exists(BASE_DB):
        print("No existing vector stores found. Setting up new ones...")
        DocumentProcessor.setup_vector_stores()
    else:
        print("Using existing vector stores from:", BASE_DB)

    main()