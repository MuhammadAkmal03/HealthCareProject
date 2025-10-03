# import os
# import logging
# import base64
# import io
# from dotenv import load_dotenv

# # LangChain components
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_pinecone import PineconeVectorStore
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import ConversationalRetrievalChain, load_summarize_chain
# from langchain.memory import ConversationBufferMemory
# from langchain.prompts import PromptTemplate
# from langchain.docstore.document import Document

# # Setup logger
# logger = logging.getLogger(__name__)

# load_dotenv()  # Load environment variables from .env
# google_api_key = os.getenv("GOOGLE_API_KEY")

# class ChatbotService:
#     _qa_chain = None
#     _summarize_chain = None

#     def __init__(self):
#         if ChatbotService._qa_chain is None or ChatbotService._summarize_chain is None:
#             logger.info("Initializing AI Assistant Service...")
#             try:
#                 # --- 1. Initialize Embeddings and LLM ---
#                 embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#                 llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3,google_api_key=google_api_key)
                
#                 # --- 2. Connect to Pinecone Vector Store ---
#                 index_name = "medicalbotdata" # Make sure this matches your ingestion script
#                 vectorstore = PineconeVectorStore.from_existing_index(index_name, embeddings)
#                 retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
#                 logger.info(f"Successfully connected to Pinecone index '{index_name}'.")

#                 # --- 3. Build the Conversational RAG Chain ---
#                 memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#                 prompt_template = """
#                 You are a helpful medical assistant chatbot. Use the provided context to answer the question.
#                 If the question is a follow-up, consider the previous conversation.
#                 Context: {context}
#                 Chat History: {chat_history}
#                 Question: {question}
#                 Answer concisely and accurately. Always end with a disclaimer to consult a doctor.
#                 """
#                 PROMPT = PromptTemplate(template=prompt_template, input_variables=["chat_history", "context", "question"])
                
#                 ChatbotService._qa_chain = ConversationalRetrievalChain.from_llm(
#                     llm=llm,
#                     retriever=retriever,
#                     memory=memory,
#                     combine_docs_chain_kwargs={"prompt": PROMPT}
#                 )
#                 logger.info("Conversational RAG chain created successfully.")

#                 # --- 4. Build the Summarization Chain ---
#                 summary_prompt_template = """Write a concise summary of the following medical text: "{text}" CONCISE SUMMARY:"""
#                 summary_prompt = PromptTemplate(template=summary_prompt_template, input_variables=["text"])
#                 ChatbotService._summarize_chain = load_summarize_chain(
#                     llm=llm,
#                     chain_type="stuff", # Use "stuff" for simplicity with smaller texts
#                     prompt=summary_prompt
#                 )
#                 logger.info("Summarization chain created successfully.")

#             except Exception as e:
#                 logger.error(f"CRITICAL ERROR: Could not initialize AI Assistant services: {e}", exc_info=True)

#     def get_chat_response(self, question: str, history: list) -> dict:
#         """Handles the RAG chatbot conversation."""
#         if not self._qa_chain:
#             raise RuntimeError("Conversational QA chain is not available.")
#         logger.info(f"Invoking QA chain with question: {question}")
#         return self._qa_chain({"question": question, "chat_history": history})

#     def get_summary(self, pdf_base64: str = None, raw_text: str = None) -> dict:
#         """Handles document summarization."""
#         if not self._summarize_chain:
#             raise RuntimeError("Summarization chain is not available.")
        
#         docs_to_summarize = []
#         if raw_text:
#             logger.info("Summarizing raw text input.")
#             docs_to_summarize = [Document(page_content=raw_text)]
#         elif pdf_base64:
#             logger.info("Summarizing PDF file upload.")
#             try:
#                 # PyPDFLoader needs a file path, so we write the decoded PDF to a temporary file
#                 pdf_bytes = base64.b64decode(pdf_base64)
#                 temp_pdf_path = "temp_uploaded_report.pdf"
#                 with open(temp_pdf_path, "wb") as f:
#                     f.write(pdf_bytes)
                
#                 loader = PyPDFLoader(temp_pdf_path)
#                 docs_to_summarize = loader.load()
#                 os.remove(temp_pdf_path) # Clean up the temporary file
#             except Exception as e:
#                 logger.error(f"Failed to process uploaded PDF: {e}")
#                 raise ValueError("Could not read the uploaded PDF file.")

#         if not docs_to_summarize:
#             raise ValueError("No content provided for summarization.")
            
#         logger.info(f"Running summarization chain on {len(docs_to_summarize)} document(s).")
#         return self._summarize_chain.invoke(docs_to_summarize)