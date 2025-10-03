import os
import logging
import base64
import tempfile
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from app.core.schemas import (
    ChatRequest,
    ChatResponse,
    SummarizeRequest,
    SummarizeResponse,
)

from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain, load_summarize_chain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader

# ---------------------------
#  Setup logger
# ---------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------
#  Load environment variables
# ---------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY not found in environment variables!")


# ---------------------------
#  Initialize Chatbot Service
# ---------------------------
class ChatbotService:
    _qa_chain = None
    _summarize_chain = None

    def __init__(self):
        if ChatbotService._qa_chain is None or ChatbotService._summarize_chain is None:
            try:
                # Embeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                # LLM
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.3,
                )

                # Pinecone Vector Store
                index_name = "medicalbot"
                vectorstore = PineconeVectorStore.from_existing_index(
                    index_name, embeddings
                )
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

                # Memory
                memory = ConversationBufferMemory(
                    memory_key="chat_history", return_messages=True
                )

                # Chatbot Prompt
                chat_prompt_template = """
                You are a helpful and friendly medical assistant chatbot.
                try to answer from the provided medical documents to answer the user's question.

                Instructions:
                1. Answer only based on the information available in the context.
                2. If the context does not contain the answer, you can provide tha answer based on your knowledge.
                3. Format recommendations as bullet points (•) where appropriate.
                4. Keep the answer clear, concise, and easy to understand.
                5.If your document doesnt have that particular information dont mention it to the user

                Chat History:
                {chat_history}

                Context from documents:
                {context}

                Question:
                {question}

                Answer:
                """

                chat_PROMPT = PromptTemplate(
                    template=chat_prompt_template,
                    input_variables=["chat_history", "context", "question"],
                )

                # QA chain
                ChatbotService._qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=retriever,
                    memory=memory,
                    combine_docs_chain_kwargs={"prompt": chat_PROMPT},
                )

                # Summarization Prompt
                summary_prompt_template = """
You are a medical text summarizer. 
Summarize the following text in a friendly, readable style,
without mentioning names, ages, or other personal identifiers and you can mention like this In the provided document ....
Don't provide large summary
If there are recommendations, list them clearly as bullet points.

Text:
{text}

Friendly Summary:
"""
                summary_prompt = PromptTemplate(
                    template=summary_prompt_template, input_variables=["text"]
                )

                # Summarization chain
                ChatbotService._summarize_chain = load_summarize_chain(
                    llm=llm, chain_type="stuff", prompt=summary_prompt
                )

                logger.info("ChatbotService initialized successfully.")

            except Exception as e:
                logger.error(f"Failed to initialize ChatbotService: {e}", exc_info=True)

    def get_chat_response(self, question: str, history: list) -> dict:
        if not self._qa_chain:
            raise RuntimeError("Conversational QA chain not available")
        response = self._qa_chain({"question": question, "chat_history": history})

        # Optional: replace * with bullet points
        answer_text = response["answer"].replace("* ", "• ")
        return {"answer": answer_text}

    def get_summary(self, pdf_base64: str = None, raw_text: str = None) -> dict:
        if not self._summarize_chain:
            raise RuntimeError("Summarization chain not available")

        docs_to_summarize = []

        # Use raw text if provided
        if raw_text:
            docs_to_summarize = [Document(page_content=raw_text)]
        # Use PDF if provided
        elif pdf_base64:
            pdf_bytes = base64.b64decode(pdf_base64)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            docs_to_summarize = loader.load()
            os.remove(tmp_path)
        else:
            raise ValueError("No content provided for summarization")

        summary_text = self._summarize_chain.run(docs_to_summarize)
        summary_text = summary_text.replace("* ", "• ")
        return {"output_text": summary_text}


# ---------------------------
#  Initialize FastAPI router
# ---------------------------
router = APIRouter()

try:
    chatbot_service = ChatbotService()
except Exception as e:
    logger.error(f"Failed to initialize ChatbotService: {e}")
    chatbot_service = None


@router.post("/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    if not chatbot_service:
        raise HTTPException(status_code=503, detail="Chatbot service unavailable")
    try:
        result = chatbot_service.get_chat_response(
            request.question, request.chat_history
        )
        return ChatResponse(answer=result["answer"])
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing chat request")


@router.post("/summarize", response_model=SummarizeResponse)
async def handle_summarize(request: SummarizeRequest):
    if not chatbot_service:
        raise HTTPException(status_code=503, detail="Summarization service unavailable")
    try:
        result = chatbot_service.get_summary(
            pdf_base64=request.pdf_base64, raw_text=request.raw_text
        )
        return SummarizeResponse(summary=result["output_text"])
    except Exception as e:
        logger.error(f"Error in summarize endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Error processing summarization request"
        )
