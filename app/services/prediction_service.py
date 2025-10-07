import pandas as pd
import joblib
import logging
import sqlite3
from datetime import datetime
from typing import Dict
from threading import Lock

logger = logging.getLogger(__name__)

DB_PATH = "predictions.db"

class PredictionService:
    _model_pipeline = None
    _symptom_binarizer = None
    _db_lock = Lock()  # Add a lock for thread-safe database operations

    def __init__(self):
        if PredictionService._model_pipeline is None:
            logger.info("Initializing PredictionService...")
            try:
                model_path = "ml_models/best_pipeline_LogisticRegression.joblib"
                symptom_path = "ml_models/symptom_binarizer.joblib"
                PredictionService._model_pipeline = joblib.load(model_path)
                PredictionService._symptom_binarizer = joblib.load(symptom_path)
                logger.info("Successfully loaded model artifacts.")
            except Exception as e:
                logger.error(f"CRITICAL ERROR loading model artifacts: {e}", exc_info=True)

            # Initialize the database table
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        diagnosis TEXT NOT NULL,
                        timestamp DATETIME NOT NULL
                    )
                """)
                conn.commit()
                conn.close()
                logger.info(f"Successfully initialized database at '{DB_PATH}'.")
            except Exception as e:
                logger.error(f"CRITICAL ERROR initializing database: {e}", exc_info=True)

    def _get_db_connection(self):
        """Create a new database connection for each operation."""
        return sqlite3.connect(DB_PATH)

    def _save_prediction(self, diagnosis: str):
        """Saves a prediction and timestamp to the SQLite database."""
        with self._db_lock:
            try:
                timestamp = datetime.now()
                conn = self._get_db_connection()
                cursor = conn.cursor()
                cursor.execute("INSERT INTO predictions (diagnosis, timestamp) VALUES (?, ?)", 
                             (diagnosis, timestamp))
                conn.commit()
                conn.close()
                logger.info(f"Saved prediction '{diagnosis}' to database.")
            except Exception as e:
                logger.error(f"Failed to save prediction to database: {e}", exc_info=True)

    def predict(self, input_data: Dict) -> str:
        """Takes user input, makes a prediction, and saves the result."""
        if self._model_pipeline is None: 
            raise RuntimeError("Model is not available.")
        
        try:
            logger.info("Preparing data for prediction...")
            df = pd.DataFrame([input_data])
            
            if "symptoms" not in df.columns:
                raise ValueError("'symptoms' field is missing from the input data.")
            
            symptoms = df.pop("symptoms")
            symptom_encoded = self._symptom_binarizer.transform(symptoms)
            symptom_df = pd.DataFrame(symptom_encoded, columns=self._symptom_binarizer.classes_, index=df.index)
            final_df = pd.concat([df, symptom_df], axis=1)
            
            required_features = self._model_pipeline.feature_names_in_
            for col in required_features:
                if col not in final_df.columns:
                    final_df[col] = 0
            final_df = final_df[required_features]
            
            logger.info(f"\n--- DATA SENT TO MODEL ---\n{final_df.to_string()}\n--------------------------")
            prediction = self._model_pipeline.predict(final_df)
            result = prediction[0]
            logger.info(f"Prediction successful. Result: {result}")
            
            # Save prediction to database
            self._save_prediction(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}", exc_info=True)
            raise

    def get_trends(self) -> Dict:
        """Fetches and aggregates prediction data for the trend chart."""
        with self._db_lock:
            logger.info("Fetching prediction trends from database.")
            try:
                conn = self._get_db_connection()
                query = "SELECT diagnosis, timestamp FROM predictions"
                df_trends = pd.read_sql_query(query, conn, parse_dates=['timestamp'])
                conn.close()
                
                if df_trends.empty:
                    return {"labels": [], "datasets": []}
                
                df_trends['date'] = df_trends['timestamp'].dt.date
                daily_counts = df_trends.groupby(['date', 'diagnosis']).size().unstack(fill_value=0)
                all_days = pd.date_range(start=daily_counts.index.min(), end=daily_counts.index.max(), freq='D')
                daily_counts = daily_counts.reindex(all_days, fill_value=0)
                
                chart_data = {"labels": [d.strftime('%Y-%m-%d') for d in daily_counts.index], "datasets": []}
                colors = {"Flu": "#F97316", "Cold": "#4169E1", "Pneumonia": "#EF4444", 
                         "Bronchitis": "#9333EA", "Healthy": "#22C55E"}
                
                for diagnosis in daily_counts.columns:
                    chart_data["datasets"].append({
                        "label": diagnosis, 
                        "data": daily_counts[diagnosis].tolist(),
                        "borderColor": colors.get(diagnosis, "#6B7280"), 
                        "fill": False, 
                        "tension": 0.1
                    })
                
                return chart_data
                
            except Exception as e:
                logger.error(f"Error fetching trend data: {e}", exc_info=True)
                raise