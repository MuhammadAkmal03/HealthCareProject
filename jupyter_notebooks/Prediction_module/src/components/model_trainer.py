
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
            
#             # --- New: Store classification report in log file ---
#             report_str = classification_report(y_test_decoded, predicted_decoded)
#             logging.info("\n" + report_str)
#             print(report_str)
            
#             return accuracy

#         except Exception as e:
#             raise CustomException(e, sys)

import os
import sys
from dataclasses import dataclass
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from Prediction_module.src.components.data_transformation import DataTransformation

from Prediction_module.src.exception import CustomException
from Prediction_module.src.logger import logging  # Make sure this import is correct
from Prediction_module.src.utils import save_object, load_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("Prediction_module", "artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_df: pd.DataFrame, test_df: pd.DataFrame, preprocessor_path: str, label_encoder_path: str):
        """
        Trains and tunes multiple models, logs them with MLflow, and saves the best one.
        
        Args:
            train_df (pd.DataFrame): Training data.
            test_df (pd.DataFrame): Testing data.
            preprocessor_path (str): Path to save the preprocessor object.
            label_encoder_path (str): Path to the saved label encoder.
        
        Returns:
            float: The accuracy of the best model on the test data.
        """
        try:
            logging.info("Separating features and target from dataframes")
            X_train = train_df.drop(columns=['Diagnosis'], axis=1)
            y_train = train_df['Diagnosis']
            X_test = test_df.drop(columns=['Diagnosis'], axis=1)
            y_test = test_df['Diagnosis']

            # Get the preprocessors from DataTransformation
            data_transformer = DataTransformation()
            linear_model_preprocessor, tree_model_preprocessor = data_transformer.get_data_transformer_object()
            
            # Save the tree-based preprocessor as it's used for the best model (LightGBM)
            save_object(file_path=preprocessor_path, obj=tree_model_preprocessor)
            
            logging.info("Preprocessing objects obtained and saved.")
            
            mlflow.set_experiment("Disease Diagnosis Comparison Final")
            
            # Define pipelines to test
            pipelines = {
                'Logistic Regression': Pipeline(steps=[
                    ('preprocessor', linear_model_preprocessor),
                    ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
                ]),
                'Random Forest': Pipeline(steps=[
                    ('preprocessor', tree_model_preprocessor),
                    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
                ]),
                'LightGBM': Pipeline(steps=[
                    ('preprocessor', tree_model_preprocessor),
                    ('classifier', LGBMClassifier(random_state=42, class_weight='balanced'))
                ])
            }
            
            best_model_name = None
            best_model_score = -1
            best_model = None

            # Loop through and train each model
            for name, pipeline in pipelines.items():
                logging.info(f"--- Training {name} ---")
                
                with mlflow.start_run(run_name=f"{name} (class_weight)"):
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)
                    
                    report_dict = classification_report(y_test, y_pred, output_dict=True)
                    accuracy = report_dict['accuracy']
                    
                    # Log metrics to MLflow
                    mlflow.log_param("model_name", name)
                    mlflow.log_param("balancing_strategy", "class_weight")
                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.log_metric("macro_avg_f1_score", report_dict['macro avg']['f1-score'])
                    
                    # Log the trained model
                    mlflow.sklearn.log_model(pipeline, "model")

                    # Log the classification report to the log file
                    logging.info(f"Classification Report for {name}:\n" + classification_report(y_test, y_pred))
                    
                    logging.info(f"Run for {name} logged to MLflow.")
                    
                    if accuracy > best_model_score:
                        best_model_score = accuracy
                        best_model_name = name
                        best_model = pipeline

            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient performance")

            logging.info(f"Best model selected: {best_model_name} with accuracy: {best_model_score}")
            
            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            logging.info(f"Best model saved to {self.model_trainer_config.trained_model_file_path}")
            
            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)