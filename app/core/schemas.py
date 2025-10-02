from pydantic import BaseModel, Field
from typing import List, Dict

# ==============================================================================
# Schemas for the Symptom Prediction Module
# ==============================================================================
class SymptomPredictionRequest(BaseModel):
    Age: int
    Gender: str
    Heart_Rate_bpm: int
    Body_Temperature_C: float
    oxygen_saturation_percent: float = Field(..., alias="Oxygen_Saturation_%")
    Systolic_BP: int
    Diastolic_BP: int
    symptoms: List[str]
    model_config = { "populate_by_name": True }

class SymptomPredictionResponse(BaseModel):
    predicted_diagnosis: str = Field(..., example="Flu")


# ==============================================================================
# Schemas for the Scan Analyzer (Deep Learning) Module (ADD THIS PART)
# ==============================================================================
class ScanAnalysisRequest(BaseModel):
    """Defines the input for the medical scan analysis."""
    image_base64: str = Field(..., description="Base64 encoded string of the medical scan image.")

class ScanAnalysisResponse(BaseModel):
    """Defines the output for the scan analysis."""
    predicted_condition: str = Field(..., example="Pneumonia")
    confidence_score: float = Field(..., example=0.92)