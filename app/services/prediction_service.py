import pandas as pd
import joblib
import logging
from typing import Dict

# Setup a logger for this service module
logger = logging.getLogger(__name__)

class PredictionService:
    """
    This service is responsible for the core machine learning logic.
    It loads the model artifacts and performs predictions.
    """
    _model_pipeline = None
    _symptom_binarizer = None

    def __init__(self):
        # This method is called only ONCE when the FastAPI app starts.
        if PredictionService._model_pipeline is None:
            logger.info("Loading model artifacts for the first time...")
            try:
                # IMPORTANT: Ensure your model files are in the 'ml_models' directory
                # and the filenames match what you saved from your notebook.
                model_path = "ml_models/best_pipeline_LogisticRegression.joblib"
                symptom_path = "ml_models/symptom_binarizer.joblib"
                
                PredictionService._model_pipeline = joblib.load(model_path)
                PredictionService._symptom_binarizer = joblib.load(symptom_path)
                
                logger.info("Successfully loaded prediction pipeline and symptom binarizer.")

            except FileNotFoundError as e:
                logger.error(f"CRITICAL ERROR: Model artifact not found. The application will not be able to make predictions. {e}")
            except Exception as e:
                logger.error(f"An unexpected error occurred during model loading: {e}", exc_info=True)

    def predict(self, input_data: Dict) -> str:
        """
        Takes a dictionary of user input, preprocesses it, and returns a diagnosis prediction.
        """
        if self._model_pipeline is None or self._symptom_binarizer is None:
            logger.error("Prediction failed because models are not loaded.")
            raise RuntimeError("Prediction models are not available. Please check server logs.")

        try:
            logger.info("Preparing data for prediction...")
            
            df = pd.DataFrame([input_data])
            
            if 'symptoms' not in df.columns:
                 raise ValueError("'symptoms' field is missing from the input data.")
            symptoms = df.pop('symptoms')

            symptom_encoded = self._symptom_binarizer.transform(symptoms)
            symptom_df = pd.DataFrame(symptom_encoded, columns=self._symptom_binarizer.classes_, index=df.index)

            final_df = pd.concat([df, symptom_df], axis=1)
            
            # Ensure column order and presence matches the model's training data
            required_features = self._model_pipeline.feature_names_in_
            for col in required_features:
                if col not in final_df.columns:
                    final_df[col] = 0
            final_df = final_df[required_features]

            logger.info("Data preprocessed successfully. Making prediction...")
            logger.info(f"\n--- DATA SENT TO MODEL ---\n{final_df.to_string()}\n--------------------------")
            
            prediction = self._model_pipeline.predict(final_df)
            
            result = prediction[0]
            logger.info(f"Prediction successful. Result: {result}")
            
            return result

        except Exception as e:
            logger.error(f"An error occurred during the prediction process: {e}", exc_info=True)
            raise