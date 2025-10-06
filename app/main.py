import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import symptom_predictor, scan_analyzer, health_assistant ,analytics

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Healthcare AI API",
    description="API for all healthcare modules.",
    version="1.0.0",
)

origins = ["http://localhost", "http://127.0.0.1", "http://127.0.0.1:5500", "null"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Routers ---
app.include_router(
    symptom_predictor.router, prefix="/predict", tags=["Symptom Predictor"]
)

app.include_router(scan_analyzer.router, prefix="/analyze", tags=["Scan Analyzer"])

app.include_router(health_assistant.router, prefix="/assistant", tags=["AI Assistant"])

app.include_router(analytics.router, prefix="/analytics")


@app.get("/", tags=["Health Check"])
def read_root():
    logger.info("Health check endpoint was called.")
    return {"status": "ok", "message": "Welcome to the Healthcare AI API!"}
