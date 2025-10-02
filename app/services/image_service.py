import base64
import io
import logging
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Setup logger for this service
logger = logging.getLogger(__name__)

class ImageService:
    """
    This service loads the trained deep learning model and performs image analysis.
    Includes enhanced logging for debugging.
    """
    _model = None
    _img_size = (150, 150)
    _class_names = ["NORMAL", "Pneumonia"]

    def __init__(self):
        # This code runs only ONCE when the FastAPI application starts.
        if ImageService._model is None:
            logger.info("Attempting to load deep learning model for the first time...")
            try:
                # Construct the full path to the model file
                model_path = os.path.join("ml_models", "cnn19.h5")
                logger.info(f"Looking for model at path: {os.path.abspath(model_path)}")

                if not os.path.exists(model_path):
                    logger.error(f"CRITICAL ERROR: Model file not found at '{model_path}'.")
                    raise FileNotFoundError(f"Model file not found at {model_path}")

                ImageService._model = tf.keras.models.load_model(model_path)
                logger.info(f"Successfully loaded image analysis model from: {model_path}")

            except Exception as e:
                # Log the full error with traceback for detailed debugging
                logger.error(f"CRITICAL ERROR: Could not load image model. {e}", exc_info=True)
                # Ensure the model is set to None so we know it failed
                ImageService._model = None

    def _preprocess_image_bytes(self, image_bytes: bytes) -> np.ndarray:
        """
        Preprocesses an image from a byte stream to be ready for the model.
        """
        logger.info("Opening image from byte stream...")
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        logger.info(f"Resizing image to {self._img_size}...")
        img = img.resize(self._img_size)
        
        logger.info("Converting image to array and normalizing...")
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0
        
        logger.info("Expanding dimensions for batch prediction...")
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array

    async def analyze(self, image_base64: str) -> (str, float):
        """
        Decodes a base64 image, preprocesses it, and returns a prediction.
        """
        if self._model is None:
             logger.error("Analysis attempted but the image model is not loaded.")
             raise RuntimeError("Image analysis model is not available. Check server startup logs for errors.")

        try:
            logger.info("Decoding base64 image string...")
            image_bytes = base64.b64decode(image_base64)
            
            processed_image = self._preprocess_image_bytes(image_bytes)
            
            logger.info("Making prediction on the preprocessed image...")
            prediction = self._model.predict(processed_image)
            
            # Interpret the sigmoid output from the model
            score = float(prediction[0][0])
            
            if score > 0.5:
                predicted_class = self._class_names[1] # Pneumonia
                confidence = score
            else:
                predicted_class = self._class_names[0] # Normal
                confidence = 1 - score
            
            logger.info(f"Image prediction successful. Class: {predicted_class}, Confidence: {confidence:.4f}")
            return predicted_class, confidence

        except Exception as e:
            logger.error(f"An error occurred during the image analysis process: {e}", exc_info=True)
            # This error will be sent back to the frontend.
            raise ValueError(f"Failed to process image. It might be corrupted or in an unsupported format.")