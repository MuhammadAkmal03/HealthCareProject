from pydantic import BaseModel, Field, model_validator
from typing import List, Optional


# --- Schemas for Symptom Prediction ---
class SymptomPredictionRequest(BaseModel):
    Age: int
    Gender: str
    Heart_Rate_bpm: int
    Body_Temperature_C: float
    oxygen_saturation_percent: float = Field(..., alias="Oxygen_Saturation_%")
    Systolic_BP: int
    Diastolic_BP: int
    symptoms: List[str]
    model_config = {"populate_by_name": True}


class SymptomPredictionResponse(BaseModel):
    predicted_diagnosis: str


# --- Schemas for Scan Analyzer ---
class ScanAnalysisRequest(BaseModel):
    image_base64: str


class ScanAnalysisResponse(BaseModel):
    predicted_condition: str
    confidence_score: float


# SCHEMAS FOR THE AI ASSISTANT MODULE 

# --- Chatbot Schemas ---
class ChatRequest(BaseModel):
    question: str
    chat_history: List[List[str]] = []  # Default to empty list


class ChatResponse(BaseModel):
    answer: str


# --- Summarizer Schemas ---
class SummarizeRequest(BaseModel):
    pdf_base64: Optional[str] = None
    raw_text: Optional[str] = None

    @model_validator(mode="after")
    def check_one_input(self):
        if sum(x is not None for x in [self.pdf_base64, self.raw_text]) != 1:
            raise ValueError(
                'Exactly one of "pdf_base64" or "raw_text" must be provided.'
            )
        return self


class SummarizeResponse(BaseModel):
    summary: str
