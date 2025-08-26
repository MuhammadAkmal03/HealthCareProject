# import sys
# import os
# import pandas as pd
# from Prediction_module.src.exception import CustomException
# from Prediction_module.src.utils import load_object

# # Add project root to sys.path for correct module imports
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# if project_root not in sys.path:
#     sys.path.append(project_root)

# class PredictPipeline:
    
#     """
#     A class for making predictions on new data using a pre-trained model and preprocessor.
#     """
    
#     def __init__(self):
#         self.preprocessor_path = os.path.join("Prediction_module", "artifacts", "preprocessor.pkl")
#         self.model_path = os.path.join("Prediction_module", "artifacts", "model.pkl")

#     def predict(self, features):
        
#         """
#         Loads the saved preprocessor and model, and makes a prediction on new data.
        
#         Args:
#             features (DataFrame): A pandas DataFrame containing the new data to be predicted.
#                                   It must have the same columns as the training data.
        
#         Returns:
#             str: The predicted diagnosis as a string label.
#         """
        
#         try:
#             preprocessor = load_object(file_path=self.preprocessor_path)
#             model = load_object(file_path=self.model_path)
            
#             # --- Feature Engineering for new data ---
#             if 'Blood_Pressure_mmHg' in features.columns:
#                 features[['Systolic_BP', 'Diastolic_BP']] = features['Blood_Pressure_mmHg'].str.split('/', expand=True)
#                 features['Systolic_BP'] = pd.to_numeric(features['Systolic_BP'])
#                 features['Diastolic_BP'] = pd.to_numeric(features['Diastolic_BP'])
#                 features.drop('Blood_Pressure_mmHg', axis=1, inplace=True)
#             if 'Patient_ID' in features.columns:
#                 features.drop('Patient_ID', axis=1, inplace=True)
            
#             # --- Remove Severity and Treatment_Plan from new data ---
#             if 'Severity' in features.columns:
#                 features.drop('Severity', axis=1, inplace=True)
#             if 'Treatment_Plan' in features.columns:
#                 features.drop('Treatment_Plan', axis=1, inplace=True)

#             data_scaled = preprocessor.transform(features)
            
#             preds = model.predict(data_scaled)
            
#             label_mapping = {
#                 0: 'Bronchitis',
#                 1: 'Cold',
#                 2: 'Flu',
#                 3: 'Healthy',
#                 4: 'Pneumonia'
#             }
            
#             predicted_label = label_mapping.get(preds[0], "Unknown")
            
#             return predicted_label
        
#         except Exception as e:
#             raise CustomException(e, sys)

# # The following code is for demonstration purposes.
# if __name__ == "__main__":
#     new_data = {
#         'Patient_ID': [1001],
#         'Age': [35],
#         'Gender': ['Female'],
#         'Symptom_1': ['Cough'],
#         'Symptom_2': ['Fever'],
#         'Symptom_3': ['Fatigue'],
#         'Blood_Pressure_mmHg': ['120/80'],
#         'Heart_Rate_bpm': [75],
#         'Body_Temperature_C': [37.2],
#         'Oxygen_Saturation_%': [98],
#     }

#     new_df = pd.DataFrame(new_data)
    
#     print("New Data to Predict:")
#     print(new_df)

#     predict_pipeline = PredictPipeline()
#     predicted_diagnosis = predict_pipeline.predict(new_df)
    
#     print(f"\nThe predicted diagnosis is: {predicted_diagnosis}")
import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
from imblearn.over_sampling import SMOTE

from Prediction_module.src.exception import CustomException
from Prediction_module.src.logger import logging

from Prediction_module.src.utils import save_object, evaluate_models, load_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("Prediction_module", "artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            # --- Apply SMOTE to the preprocessed training data ---
            logging.info("Applying SMOTE to training data...")
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            logging.info("SMOTE application complete.")
            
            # Define models and their hyperparameters
            models = {
                "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
                "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear', max_iter=1000, class_weight='balanced'),
            }
            
            params = {
                "Random Forest": {
                    'n_estimators': [50, 100],
                    'max_depth': [10, None],
                },
                "Logistic Regression": {
                    'C': [0.1, 1.0],
                }
            }

            model_report: dict = evaluate_models(
                X_train=X_train_resampled, y_train=y_train_resampled, X_test=X_test, y_test=y_test,
                models=models, param=params
            )
            
            logging.info(f"Model Training Report: {model_report}")
            
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient performance")
            
            logging.info(f"Best found model on both training and testing dataset: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Load the LabelEncoder to decode predictions
            le = load_object(os.path.join("Prediction_module", "artifacts", "label_encoder.pkl"))
            
            predicted = best_model.predict(X_test)
            predicted_decoded = le.inverse_transform(predicted.astype(int))
            y_test_decoded = le.inverse_transform(y_test.astype(int))

            accuracy = accuracy_score(y_test_decoded, predicted_decoded)
            print("\nClassification Report (SMOTE-trained model):")
            print(classification_report(y_test_decoded, predicted_decoded))
            
            return accuracy

        except Exception as e:
            raise CustomException(e, sys)