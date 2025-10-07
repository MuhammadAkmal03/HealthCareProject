import os
import logging
import base64
import io
import tempfile
import sqlite3
from datetime import datetime
from typing import List, Dict
import pandas as pd

# LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain, load_summarize_chain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# Setup logger and database path
logger = logging.getLogger(__name__)
DB_PATH = "predictions.db"

class ChatbotService:
    _qa_chain = None
    _summarize_chain = None
    _db_connection = None
    _llm = None

    def __init__(self):
        if ChatbotService._qa_chain is None:
            logger.info("Initializing AI Assistant Service...")
            try:
                # 1. Initialize Embeddings and LLM
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                ChatbotService._llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
                
                # 2. Connect to Pinecone Vector Store
                index_name = "medicalbotdata" 
                vectorstore = PineconeVectorStore.from_existing_index(index_name, embeddings)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                logger.info(f"Successfully connected to Pinecone index '{index_name}'.")

                # 3. Build the Conversational RAG Chain
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                prompt_template = """You are a helpful and honest medical information assistant. Your task is to provide answers based on the provided context. You can also provide an answer based on your knowledge. Your answers should be clear and concise. Do not mention that you are getting the information from a provided text. IMPORTANT: Always end your response with a clear disclaimer: "This information is for educational purposes only. Please consult a healthcare professional for medical advice."
                Context: {context}
                Chat History: {chat_history}
                Question: {question}
                Helpful Answer:"""
                PROMPT = PromptTemplate(template=prompt_template, input_variables=["chat_history", "context", "question"])
                ChatbotService._qa_chain = ConversationalRetrievalChain.from_llm(llm=ChatbotService._llm, retriever=retriever, memory=memory, combine_docs_chain_kwargs={"prompt": PROMPT})
                logger.info("Conversational RAG chain created successfully.")

                # 4. Build the Summarization Chain
                summary_prompt_template = """
                You are an AI assistant that summarizes medical reports for patients in a friendly, simple paragraph.

                **CRITICAL INSTRUCTIONS:**
                1. BE EXTREMELY CONCISE. The entire summary must be a single, easy-to-read paragraph, no more than 4-5 sentences.
                2. Address the user directly using "your report".
                3. DO NOT use bullet points, lists, asterisks (*), or dashes (-). Write in plain paragraph format.
                4. DO NOT include any personal details like age or gender.
                5. Focus only on the most important findings that require discussion with a doctor.

                Medical Text:
                "{text}"

                Concise, friendly, single-paragraph summary:
                """
                summary_prompt = PromptTemplate(template=summary_prompt_template, input_variables=["text"])
                ChatbotService._summarize_chain = load_summarize_chain(llm=ChatbotService._llm, chain_type="stuff", prompt=summary_prompt)
                logger.info("Summarization chain created successfully.")

                # 5. Connect to Database and Create Tables for BOTH charts
                ChatbotService._db_connection = sqlite3.connect(DB_PATH, check_same_thread=False)
                cursor = ChatbotService._db_connection.cursor()
                # Table for the trend chart
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        diagnosis TEXT NOT NULL,
                        timestamp DATETIME NOT NULL
                    )
                """)
                # Table for the common queries chart
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chatbot_queries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        topic TEXT NOT NULL,
                        timestamp DATETIME NOT NULL
                    )
                """)
                ChatbotService._db_connection.commit()
                logger.info(f"Chatbot service connected to database and ensured tables exist.")

            except Exception as e:
                logger.error(f"CRITICAL ERROR: Could not initialize AI Assistant services: {e}", exc_info=True)

    def _save_query_topic(self, topic: str):
        if self._db_connection is None: return
        try:
            timestamp = datetime.now()
            cursor = self._db_connection.cursor()
            cursor.execute("INSERT INTO chatbot_queries (topic, timestamp) VALUES (?, ?)", (topic, timestamp))
            self._db_connection.commit()
            logger.info(f"Saved query topic '{topic}' to database.")
        except Exception as e:
            logger.error(f"Failed to save query topic: {e}", exc_info=True)
            
    def _extract_topic_with_llm(self, question: str) -> str or None:
        if not self._llm: return None
        try:
            extraction_prompt = PromptTemplate.from_template('Analyze the user question and extract the primary medical topic. Respond with ONLY the topic name or "None". Question: "{user_question}" Topic:')
            extraction_chain = extraction_prompt | self._llm
            extracted_topic = extraction_chain.invoke({"user_question": question}).content
            if "none" in extracted_topic.lower(): return None
            return extracted_topic.strip().capitalize()
        except Exception as e:
            logger.error(f"Error during topic extraction: {e}")
            return None

    def get_chat_response(self, question: str, history: list) -> dict:
        if not self._qa_chain: raise RuntimeError("Conversational QA chain is not available.")
        topic = self._extract_topic_with_llm(question)
        if topic: self._save_query_topic(topic)
        logger.info(f"Invoking RAG chain with question: {question}")
        return self._qa_chain.invoke({"question": question, "chat_history": history})

    def get_summary(self, pdf_base64: str = None, raw_text: str = None) -> dict:
        if not self._summarize_chain: raise RuntimeError("Summarization chain is not available.")
        docs_to_summarize = []
        if raw_text:
            docs_to_summarize = [Document(page_content=raw_text)]
        elif pdf_base64:
            try:
                pdf_bytes = base64.b64decode(pdf_base64)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(pdf_bytes)
                    tmp_path = tmp.name
                loader = PyPDFLoader(tmp_path)
                docs_to_summarize = loader.load()
                os.remove(tmp_path)
            except Exception as e:
                logger.error(f"Failed to process uploaded PDF: {e}")
                raise ValueError("Could not read the uploaded PDF file.")
        if not docs_to_summarize: raise ValueError("No content provided for summarization.")
        return self._summarize_chain.invoke(docs_to_summarize)

    def get_query_topics(self) -> Dict:
        if self._db_connection is None: raise RuntimeError("Database connection is not available.")
        logger.info("Fetching top 5 query topics from database.")
        try:
            query = "SELECT topic, COUNT(topic) as count FROM chatbot_queries GROUP BY topic ORDER BY count DESC LIMIT 5"
            df_topics = pd.read_sql_query(query, self._db_connection)
            if df_topics.empty:
                return {"labels": [], "datasets": []}
            chart_data = {
                "labels": df_topics['topic'].tolist(),
                "datasets": [{"label": "Common Queries", "data": df_topics['count'].tolist(), "backgroundColor": ['#22C55E', '#3B82F6', '#6366F1', '#EC4899', '#F97316']}]
            }
            return chart_data
        except Exception as e:
            logger.error(f"Error fetching query topics: {e}", exc_info=True)
            raise