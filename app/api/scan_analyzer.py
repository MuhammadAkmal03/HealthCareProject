from fastapi import APIRouter, HTTPException
from app.core.schemas import (
    ScanAnalysisRequest,
    ScanAnalysisResponse,
)  
from app.services.image_service import ImageService
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

image_service = ImageService()


@router.post("/", response_model=ScanAnalysisResponse)
async def analyze_scan(request: ScanAnalysisRequest):
    """
    Takes a Base64 encoded image and returns an AI-powered analysis.
    """
    logger.info("Received request for image analysis.")
    try:
        prediction, confidence = await image_service.analyze(request.image_base64)
        return ScanAnalysisResponse(
            predicted_condition=prediction, confidence_score=confidence
        )
    except RuntimeError as e:
        logger.error(f"Service Error: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        logger.warning(f"Invalid image data: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"An unexpected internal error occurred: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal server error occurred during image analysis.",
        )
