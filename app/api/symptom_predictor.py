from fastapi import APIRouter, HTTPException
from app.core.schemas import SymptomPredictionRequest, SymptomPredictionResponse
from app.services.prediction_service import PredictionService
import logging

router = APIRouter()
logger = logging.getLogger(__name__)
prediction_service = PredictionService()


@router.post("/", response_model=SymptomPredictionResponse)
def predict_diagnosis(request: SymptomPredictionRequest):
    """
    This endpoint is unchanged.
    It receives patient data, gets a diagnosis, and the service saves it.
    """
    logger.info(f"Received prediction request for age: {request.Age}, gender: {request.Gender}")
    try:
        input_dict = request.model_dump(by_alias=True)
        diagnosis = prediction_service.predict(input_dict)
        return SymptomPredictionResponse(predicted_diagnosis=diagnosis)
    except RuntimeError as e:
        logger.error(f"Service Error: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        logger.warning(f"Invalid input data received: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid input data: {e}")
    except Exception as e:
        logger.error(f"An unexpected internal error occurred during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@router.get("/trends")
def get_prediction_trends():
    """
    Fetches aggregated prediction data to be displayed in a trend chart.
    """
    logger.info("Trend data endpoint called.")
    try:
        trends = prediction_service.get_trends()
        return trends
    except Exception as e:
        logger.error(f"Error fetching trends: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not fetch trend data.")
