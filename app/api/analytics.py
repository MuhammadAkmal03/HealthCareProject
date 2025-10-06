from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session
from collections import defaultdict
import os

from app.database import SessionLocal
from app.models.prediction_record import PredictionRecord

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Serve the chart HTML
@router.get("/trends")
def get_trends_page():
    return FileResponse(os.path.join("frontend", "trendchart.html"))

# Serve JSON data
@router.get("/trends/data")
def get_trends_data(db: Session = Depends(get_db)):
    records = db.query(PredictionRecord).all()
    
    trends = defaultdict(lambda: defaultdict(int))
    for r in records:
        day = r.timestamp.date().isoformat()
        trends[day][r.disease] += 1

    return JSONResponse(trends)
