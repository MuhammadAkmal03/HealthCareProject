import os
import sys
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from Prediction_module.src.exception import CustomException

def save_object(file_path, obj):
    """
    Saves a Python object to a binary file using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Loads a Python object from a binary file using pickle.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Trains and evaluates multiple models using GridSearchCV.
    """
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]
            
            gs = GridSearchCV(model, para, cv=3, verbose=1)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            y_test_pred = model.predict(X_test)
            test_model_score = accuracy_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
        return report

    except Exception as e:
        raise CustomException(e, sys)