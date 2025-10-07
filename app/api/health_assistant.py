from fastapi import APIRouter, HTTPException
from app.core.schemas import ChatRequest, ChatResponse, SummarizeRequest, SummarizeResponse
from app.services.chatbot_service import ChatbotService
import logging

# Initialize the router and logger
router = APIRouter()
logger = logging.getLogger(__name__)

#  creates a single, shared instance of the ChatbotService when the server starts.
try:
    chatbot_service = ChatbotService()
except Exception as e:
    logger.error(f"FATAL: Failed to initialize ChatbotService on startup: {e}", exc_info=True)
    chatbot_service = None

@router.post("/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    """
    Endpoint for the RAG chatbot.
    """
    if not chatbot_service:
        raise HTTPException(status_code=503, detail="Chatbot service is currently unavailable.")
    
    logger.info(f"Received chat request: '{request.question}'")
    try:
        result = chatbot_service.get_chat_response(request.question, request.chat_history)
        return ChatResponse(answer=result['answer'])
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while processing the chat request.")

@router.post("/summarize", response_model=SummarizeResponse)
async def handle_summarize(request: SummarizeRequest):
    """
    Endpoint for document summarization.
    """
    if not chatbot_service:
        raise HTTPException(status_code=503, detail="Summarization service is currently unavailable.")
    
    logger.info("Received summarization request.")
    try:
        result = chatbot_service.get_summary(pdf_base64=request.pdf_base64, raw_text=request.raw_text)
        return SummarizeResponse(summary=result['output_text'])
    except Exception as e:
        logger.error(f"Error in summarize endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during summarization.")

@router.get("/query_topics")
def get_common_queries():
    """
    Fetches aggregated query topic data for the bar chart.
    """
    if not chatbot_service:
        raise HTTPException(status_code=503, detail="Chatbot service is currently unavailable.")
        
    logger.info("Query topics endpoint called.")
    try:
        topics = chatbot_service.get_query_topics()
        return topics
    except Exception as e:
        logger.error(f"Error fetching query topics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not fetch query topic data.")