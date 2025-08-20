import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from Prediction_module.src.exception import CustomException
from Prediction_module.src.logger import logging

from Prediction_module.src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    
    """
    A class to train, evaluate, and save the best-performing machine learning model.
    """
    
    trained_model_file_path = os.path.join("Prediction_module", "artifacts", "model.pkl")

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        
        """
        Trains and evaluates multiple models, selects the best one,
        and saves it to disk.

        Args:
            train_array (np.ndarray): The pre-processed training data array.
            test_array (np.ndarray): The pre-processed testing data array.

        Returns:
            float: The accuracy score of the best-performing model on the test set.
        """
        
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test)

            models = {
                "Random Forest": RandomForestClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear'),
            }
            
            params = {
                "Random Forest": {
                    'n_estimators': [50, 100],
                    'max_depth': [10, None]
                },
                "Logistic Regression": {
                    'C': [0.1, 1.0]
                }
            }

            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train_encoded, X_test=X_test, y_test=y_test_encoded,
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

            predicted = best_model.predict(X_test)
            predicted_decoded = le.inverse_transform(predicted)

            accuracy = accuracy_score(y_test, predicted_decoded)
            return accuracy

        except Exception as e:
            raise CustomException(e, sys)