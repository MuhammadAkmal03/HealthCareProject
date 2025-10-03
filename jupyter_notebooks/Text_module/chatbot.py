import os
import logging
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# --- Pinecone Imports ---
from pinecone import Pinecone, ServerlessSpec

# --- Basic Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# ==============================================================================
# STEP 1: LOAD ENVIRONMENT VARIABLES
# ==============================================================================
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not PINECONE_API_KEY or not GOOGLE_API_KEY:
    raise ValueError("Pinecone and Google API keys must be set in the .env file.")

# Langsmith tracking (optional, but good practice)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Healthcare Chatbot"

# ==============================================================================
# STEP 2: DEFINE CONSTANTS
# ==============================================================================
INDEX_NAME = "medicalbot"
PDF_DOCS_PATH = "./documents/"  # The path to your PDF files from your first script
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ==============================================================================
# STEP 3: DATA INGESTION AND PROCESSING FUNCTIONS
# ==============================================================================
def load_pdf_documents(path: str):
    """Loads all PDF files from a directory."""
    logging.info(f"Loading PDF documents from path: {path}")
    loader = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    logging.info(f"Loaded {len(documents)} documents.")
    return documents


def split_documents_into_chunks(documents):
    """Splits documents into smaller chunks for processing."""
    logging.info("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    logging.info(f"Split documents into {len(text_chunks)} chunks.")
    return text_chunks


# ==============================================================================
# STEP 4: VECTOR STORE SETUP (PINECOME)
# ==============================================================================
def get_or_create_vector_store(text_chunks, embeddings):
    """Initializes Pinecone and creates/connects to a vector store."""
    logging.info("Initializing Pinecone client...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if INDEX_NAME not in pc.list_indexes().names():
        logging.info(f"Pinecone index '{INDEX_NAME}' not found. Creating a new one...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  # Dimension of the all-MiniLM-L6-v2 model
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        logging.info(
            "Creating embeddings and uploading to Pinecone. This may take a few minutes..."
        )
        vector_store = PineconeVectorStore.from_documents(
            documents=text_chunks, embedding=embeddings, index_name=INDEX_NAME
        )
        logging.info("Pinecone index created and populated successfully.")
    else:
        logging.info(f"Connecting to existing Pinecone index: '{INDEX_NAME}'")
        vector_store = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME, embedding=embeddings
        )
        logging.info("Successfully connected to Pinecone index.")

    return vector_store


# ==============================================================================
# STEP 5: CONVERSATIONAL CHAIN SETUP (GEMINI)
# ==============================================================================
custom_prompt_template = """
You are a helpful and honest medical information assistant.
Your task is to provide answers based on the provided context from medical documents.
If the context doesn't contain the answer, say that you don't have enough information from the documents but provide a general answer based on your own knowledge.
Always end your response with a clear disclaimer: "This information is for educational purposes only. Please consult a healthcare professional for medical advice."
Your answers should be clear, concise, and easy to understand for a general audience.

Context: {context}
Chat History: {chat_history}

Question: {question}
Helpful Answer:
"""


def create_custom_prompt():
    """Creates a custom prompt template for the RAG chain."""
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "chat_history", "question"],
    )


def get_conversational_rag_chain(vector_store):
    """Creates the main conversational retrieval chain using Google Gemini."""
    logging.info("Creating conversational RAG chain with Gemini...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.3
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    prompt = create_custom_prompt()

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        verbose=False,
    )
    logging.info("Conversational RAG chain created successfully.")
    return chain


# ==============================================================================
# STEP 6: MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # --- Data Ingestion and Vector Store Creation ---
    documents = load_pdf_documents(PDF_DOCS_PATH)
    text_chunks = split_documents_into_chunks(documents)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = get_or_create_vector_store(text_chunks, embeddings)

    # --- Create the QA Chain ---
    qa_chain = get_conversational_rag_chain(vector_store)

    # --- Interactive Chat Loop ---
    print("\n--- Medical Chatbot Initialized ---")
    print("Ask a question about your medical documents. Type 'exit' to quit.")
    while True:
        question = input("\nYou: ")
        if question.lower() == "exit":
            print("Chatbot session ended. Goodbye!")
            break

        # Get the result from the chain

        result = qa_chain.invoke({"question": question})

        # Print the answer
        print(f"\nAI: {result['answer']}")
