# main.py - Complete RAG-based RBAC System with FAISS

import os
import streamlit as st
import asyncio  # 👈 Added to fix event loop issue on Streamlit Cloud

# Ensure asyncio event loop is available
try:
    asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

from typing import List, Dict, Optional

# --- FAISS Imports ---
from langchain_community.vectorstores import FAISS

# --- LangChain Imports ---
from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import hashlib

# --- LLM and Embedding Setup ---
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Configuration
BASE_DOC_PATH = './docs'
BASE_DB = './db'  # Now acts as a base folder for FAISS index files
EMBEDDING_MODEL = 'models/text-embedding-004'

# Initialize embeddings and LLM AFTER setting event loop
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
embedding = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)




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
        "admin": ["general", "marketing", "finance", "hr", "engineering"],
        "employee": ["general"]
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

        print(f"Processing {dept}: {os.listdir(folder_path)}")

        if dept == 'hr':
            for file in os.listdir(folder_path):
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

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(all_docs)
        return chunks

    @staticmethod
    def setup_vector_stores():
        """Initialize vector stores for all departments using FAISS"""
        for dept in DocumentProcessor.DEPARTMENTS:
            chunks = DocumentProcessor.load_and_split_documents(dept)
            if chunks:
                db_file_path = os.path.join(BASE_DB, f'{dept}.faiss')
                os.makedirs(BASE_DB, exist_ok=True)
                
                # Create a new FAISS index from documents
                vectorstore = FAISS.from_documents(chunks, embedding)
                
                # Save the FAISS index to the file
                vectorstore.save_local(db_file_path)
                print(f'✓ Setup and saved FAISS vector store for {dept}: {len(chunks)} chunks')


class RAGChatbot:
    """RAG-based chatbot with role-based access control"""

    def __init__(self):
        self.vector_stores = {}
        self.qa_chains = {}
        self._load_vector_stores()
        self._setup_qa_chains()

    def _load_vector_stores(self):
        """Load existing vector stores from FAISS files"""
        for dept in DocumentProcessor.DEPARTMENTS:
            db_file_path = os.path.join(BASE_DB, f'{dept}.faiss')
            if os.path.exists(db_file_path):
                try:
                    self.vector_stores[dept] = FAISS.load_local(
                        db_file_path,
                        embedding,
                        allow_dangerous_deserialization=True # This is required for security on some versions
                    )
                    print(f"✓ Loaded existing FAISS vector store for {dept}")
                except Exception as e:
                    print(f"Error loading FAISS store for {dept}: {e}")
            else:
                print(f"⚠️  No FAISS vector store found for {dept} at {db_file_path}")

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

    if 'chatbot' not in st.session_state:
        with st.spinner("Initializing RAG system..."):
            st.session_state.chatbot = RAGChatbot()

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
        user_info = st.session_state.user_info
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

        st.subheader(f"💬 Hello {user_info['name']}! How can I help you today?")
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 Sources"):
                        for source in message["sources"]:
                            st.write(f"**{source['department'].title()}:** {source['source']}")
                            st.caption(source['content_preview'])

        if prompt := st.chat_input("Ask me anything about the company..."):
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Searching company knowledge base..."):
                    result = st.session_state.chatbot.query(prompt, user_info['roles'])
                    st.markdown(result["answer"])
                    if result["sources"]:
                        with st.expander("📚 Sources"):
                            for source in result["sources"]:
                                st.write(f"**{source['department'].title()}:** {source['source']}")
                                st.caption(source['content_preview'])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"]
                    })

if __name__ == "__main__":
    if not os.path.exists(os.path.join(BASE_DB, 'hr.faiss')):
        print("No existing vector stores found. Setting up new ones...")
        DocumentProcessor.setup_vector_stores()
    else:
        print("Using existing vector stores from:", BASE_DB)

    main()

