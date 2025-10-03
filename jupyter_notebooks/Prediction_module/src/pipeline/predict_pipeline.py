
# import os
# import sys
# from dataclasses import dataclass

# from sklearn.ensemble import (
#     RandomForestClassifier,
# )
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.preprocessing import LabelEncoder
# import numpy as np
# from imblearn.over_sampling import SMOTE

# from Prediction_module.src.exception import CustomException
# from Prediction_module.src.logger import logging

# from Prediction_module.src.utils import save_object, evaluate_models, load_object

# @dataclass
# class ModelTrainerConfig:
#     trained_model_file_path = os.path.join("Prediction_module", "artifacts", "model.pkl")

# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config = ModelTrainerConfig()

#     def initiate_model_trainer(self, train_array, test_array):
#         try:
#             logging.info("Split training and test input data")
#             X_train, y_train = train_array[:, :-1], train_array[:, -1]
#             X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
#             # --- Apply SMOTE to the preprocessed training data ---
#             logging.info("Applying SMOTE to training data...")
#             smote = SMOTE(random_state=42)
#             X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
#             logging.info("SMOTE application complete.")
            
#             # Define models and their hyperparameters
#             models = {
#                 "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
#                 "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear', max_iter=1000, class_weight='balanced'),
#             }
            
#             params = {
#                 "Random Forest": {
#                     'n_estimators': [50, 100],
#                     'max_depth': [10, None],
#                 },
#                 "Logistic Regression": {
#                     'C': [0.1, 1.0],
#                 }
#             }

#             model_report: dict = evaluate_models(
#                 X_train=X_train_resampled, y_train=y_train_resampled, X_test=X_test, y_test=y_test,
#                 models=models, param=params
#             )
            
#             logging.info(f"Model Training Report: {model_report}")
            
#             best_model_score = max(sorted(model_report.values()))
#             best_model_name = list(model_report.keys())[
#                 list(model_report.values()).index(best_model_score)
#             ]
#             best_model = models[best_model_name]

#             if best_model_score < 0.6:
#                 raise CustomException("No best model found with sufficient performance")
            
#             logging.info(f"Best found model on both training and testing dataset: {best_model_name}")

#             save_object(
#                 file_path=self.model_trainer_config.trained_model_file_path,
#                 obj=best_model
#             )

#             # Load the LabelEncoder to decode predictions
#             le = load_object(os.path.join("Prediction_module", "artifacts", "label_encoder.pkl"))
            
#             predicted = best_model.predict(X_test)
#             predicted_decoded = le.inverse_transform(predicted.astype(int))
#             y_test_decoded = le.inverse_transform(y_test.astype(int))

#             accuracy = accuracy_score(y_test_decoded, predicted_decoded)
#             print("\nClassification Report (SMOTE-trained model):")
#             print(classification_report(y_test_decoded, predicted_decoded))
            
#             return accuracy

#         except Exception as e:
#             raise CustomException(e, sys)

import os
import sys
import pandas as pd
from Prediction_module.src.exception import CustomException
from Prediction_module.src.logger import logging
from Prediction_module.src.utils import load_object

class PredictPipeline:
    def __init__(self):
        """
        Initializes the prediction pipeline by loading the preprocessor and trained model.
        """
        # Define paths to artifacts
        self.preprocessor_path = os.path.join("Prediction_module", "artifacts", "preprocessor.pkl")
        self.model_path = os.path.join("Prediction_module", "artifacts", "model.pkl")
        self.label_encoder_path = os.path.join("Prediction_module", "artifacts", "label_encoder.pkl")
        
        # Load the artifacts
        try:
            self.preprocessor = load_object(self.preprocessor_path)
            self.model = load_object(self.model_path)
            self.label_encoder = load_object(self.label_encoder_path)
            logging.info("Successfully loaded preprocessor, model, and label encoder.")
        except Exception as e:
            raise CustomException(f"Failed to load artifacts: {e}", sys)

    def predict(self, features: pd.DataFrame):
        """
        Predicts the disease diagnosis for new, unseen data.
        
        Args:
            features (pd.DataFrame): A DataFrame containing the new data to predict.
        
        Returns:
            list: A list of predicted disease diagnoses (decoded from labels).
        """
        try:
            logging.info("Starting prediction process.")
            
            # Apply feature engineering to the new data
            if 'Blood_Pressure_mmHg' in features.columns:
                features[['Systolic_BP', 'Diastolic_BP']] = features['Blood_Pressure_mmHg'].str.split('/', expand=True)
                features['Systolic_BP'] = pd.to_numeric(features['Systolic_BP'])
                features['Diastolic_BP'] = pd.to_numeric(features['Diastolic_BP'])
                features.drop('Blood_Pressure_mmHg', axis=1, inplace=True)
            
            if 'Patient_ID' in features.columns:
                features.drop('Patient_ID', axis=1, inplace=True)
            
            # Transform the features using the loaded preprocessor
            # Note: We do not use fit_transform here, only transform.
            # The preprocessor is already fitted on the training data.
            preprocessed_data = self.preprocessor.transform(features)
            
            logging.info("Data preprocessed successfully.")
            
            # Make predictions with the loaded model
            predictions_encoded = self.model.predict(preprocessed_data)
            
            # Decode the predictions back to their original string labels
            predictions_decoded = self.label_encoder.inverse_transform(predictions_encoded)
            
            logging.info("Predictions made successfully.")
            return predictions_decoded.tolist()
            
        except Exception as e:
            raise CustomException(e, sys)

# Example usage for a standalone test of the prediction pipeline
class CustomData:
    def __init__(self, Age: int, Gender: str, Heart_Rate_bpm: int, Body_Temperature_C: float,
                 Oxygen_Saturation: float, Systolic_BP: int, Diastolic_BP: int,
                 Symptom_1: str, Symptom_2: str, Symptom_3: str):
        self.Age = Age
        self.Gender = Gender
        self.Heart_Rate_bpm = Heart_Rate_bpm
        self.Body_Temperature_C = Body_Temperature_C
        self.Oxygen_Saturation = Oxygen_Saturation
        self.Systolic_BP = Systolic_BP
        self.Diastolic_BP = Diastolic_BP
        self.Symptom_1 = Symptom_1
        self.Symptom_2 = Symptom_2
        self.Symptom_3 = Symptom_3
        
    def get_data_as_dataframe(self):
        custom_data_input_dict = {
            "Age": [self.Age],
            "Gender": [self.Gender],
            "Heart_Rate_bpm": [self.Heart_Rate_bpm],
            "Body_Temperature_C": [self.Body_Temperature_C],
            "Oxygen_Saturation_%": [self.Oxygen_Saturation],
            "Systolic_BP": [self.Systolic_BP],
            "Diastolic_BP": [self.Diastolic_BP],
            "Symptom_1": [self.Symptom_1],
            "Symptom_2": [self.Symptom_2],
            "Symptom_3": [self.Symptom_3]
        }
        return pd.DataFrame(custom_data_input_dict)

if __name__ == "__main__":
    # Create an instance of the pipeline
    predict_pipe = PredictPipeline()

    # Create a new data point to predict
    data = CustomData(
        Age=35,
        Gender='Female',
        Heart_Rate_bpm=80,
        Body_Temperature_C=37.5,
        Oxygen_Saturation=97,
        Systolic_BP=120,
        Diastolic_BP=80,
        Symptom_1='Cough',
        Symptom_2='Fatigue',
        Symptom_3='Fever'
    )
    
    # Get data as a DataFrame
    new_data_df = data.get_data_as_dataframe()
    print("New data for prediction:")
    print(new_data_df)
    
    # Make a prediction
    predictions = predict_pipe.predict(new_data_df)
    print("\nPredicted Diagnosis:")
    print(predictions)