import pytest
from fastapi.testclient import TestClient
import os
import sys
import base64
from PIL import Image
import io

# Add the project root to the Python path to allow imports from 'app'
# This ensures that when pytest runs, it can find your 'app' module.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app

# Create a single client to make requests to your app in all tests
client = TestClient(app)


# --- Test 1: General Health Check ---
def test_health_check():
    """Tests if the root endpoint is running and returns a 200 OK status."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "Welcome to the Healthcare AI API!"}


# --- Test 2: Symptom Predictor Module ---
def test_symptom_prediction_success():
    """
    Tests the /predict/ endpoint with a valid payload (the "happy path").
    """
    valid_payload = {
        "Age": 52,
        "Gender": "Female",
        "Heart_Rate_bpm": 90,
        "Body_Temperature_C": 38.5,
        "Oxygen_Saturation_%": 94.0,
        "Systolic_BP": 140,
        "Diastolic_BP": 90,
        "symptoms": ["Cough", "Fever", "Body ache"]
    }
    response = client.post("/predict/", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_diagnosis" in data
    assert isinstance(data["predicted_diagnosis"], str)


def test_symptom_prediction_validation_error():
    """
    Tests the /predict/ endpoint with an invalid payload (the "sad path").
    """
    invalid_payload = { "Age": "fifty-two", "Gender": "Female", "symptoms": [] } # Incomplete payload
    response = client.post("/predict/", json=invalid_payload)
    assert response.status_code == 422 # 422 Unprocessable Entity


# --- Test 3: Scan Analyzer (Deep Learning) Module ---
def create_dummy_image_base64() -> str:
    """Helper function to create a simple black image and encode it as Base64."""
    # Create a small, black 10x10 pixel image
    img = Image.new('RGB', (10, 10), color = 'black')
    # Save the image to an in-memory buffer
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    # Encode the bytes from the buffer into a Base64 string
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_base64

def test_scan_analyzer_success():
    """
    Tests the /analyze/ endpoint with a valid Base64 image payload.
    """
    # Create a dummy image payload
    image_payload = {
        "image_base64": create_dummy_image_base64()
    }
    response = client.post("/analyze/", json=image_payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_condition" in data
    assert "confidence_score" in data
    assert isinstance(data["predicted_condition"], str)
    assert isinstance(data["confidence_score"], float)


# --- Test 4: AI Assistant (Chatbot) Module ---
def test_ai_assistant_chat_success():
    """
    Tests the /assistant/chat endpoint with a valid question.
    Note: Your endpoint returns a StreamingResponse, so we check for a 200 status
    and that the response content is not empty.
    """
    # This test assumes you have a non-streaming endpoint for simplicity of testing JSON
    # If your endpoint is streaming text, the test needs to be different.
    # Let's assume you have an endpoint that returns a simple JSON as it's easier to test.
    # We will need to update the router for this test to pass.
    
    # Let's adjust the test to work with your CURRENT streaming router.
    chat_payload = {
        "question": "What is a fever?",
        "chat_history": []
    }
    
    response = client.post("/assistant/chat/", json=chat_payload)
    
    # For a streaming response, we check for success and that we received content
    assert response.status_code == 200
    # The 'text' attribute will contain the concatenated streamed content
    assert len(response.text) > 0 
    assert isinstance(response.text, str)


