import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder

from Prediction_module.src.exception import CustomException
from Prediction_module.src.logger import logging
import os
from Prediction_module.src.utils import save_object, load_object

@dataclass
class DataTransformationConfig:
    """
    Configuration class for data transformation paths.
    """
    preprocessor_obj_file_path = os.path.join('Prediction_module', 'artifacts', "preprocessor.pkl")
    label_encoder_path = os.path.join('Prediction_module', 'artifacts', "label_encoder.pkl")

class DataTransformation:
    """
    A class to handle data preprocessing, including feature engineering,
    scaling, and encoding of features.
    """
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates and returns a data preprocessor object with appropriate
        transformers for numerical, ordinal, and nominal features.

        Returns:
            ColumnTransformer: A preprocessor object ready to be fitted to data.
        """
        try:
            # Define features that a user can provide
            numerical_features = ['Age', 'Heart_Rate_bpm', 'Body_Temperature_C', 'Oxygen_Saturation_%', 'Systolic_BP', 'Diastolic_BP']
            nominal_categorical_features = ['Gender', 'Symptom_1', 'Symptom_2', 'Symptom_3']

            numerical_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )
            nominal_cat_pipeline = Pipeline(
                steps=[
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore'))
                ]
            )
            
            logging.info(f"Numerical columns: {numerical_features}")
            logging.info(f"Nominal categorical columns: {nominal_categorical_features}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", numerical_pipeline, numerical_features),
                    ("nom_cat_pipelines", nominal_cat_pipeline, nominal_categorical_features)
                ],
                remainder='passthrough'
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        """
        Performs data transformation by applying feature engineering and
        the preprocessor to the training and testing datasets.

        Args:
            train_path (str): The file path to the training data.
            test_path (str): The file path to the testing data.
        
        Returns:
            tuple: A tuple containing the pre-processed data arrays and the preprocessor file path.
                   (train_arr, test_arr, preprocessor_obj_file_path)
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Starting feature engineering on dataframes")

            for df in [train_df, test_df]:
                if 'Blood_Pressure_mmHg' in df.columns:
                    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood_Pressure_mmHg'].str.split('/', expand=True)
                    df['Systolic_BP'] = pd.to_numeric(df['Systolic_BP'])
                    df['Diastolic_BP'] = pd.to_numeric(df['Diastolic_BP'])
                    df.drop('Blood_Pressure_mmHg', axis=1, inplace=True)
                if 'Patient_ID' in df.columns:
                    df.drop('Patient_ID', axis=1, inplace=True)
                
                if 'Severity' in df.columns:
                    df.drop('Severity', axis=1, inplace=True)
                if 'Treatment_Plan' in df.columns:
                    df.drop('Treatment_Plan', axis=1, inplace=True)

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Diagnosis"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes.")
            
            # Save the fitted LabelEncoder for the prediction pipeline
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(target_feature_train_df)
            y_test_encoded = le.transform(target_feature_test_df)
            save_object(self.data_transformation_config.label_encoder_path, obj=le)

            # Transform features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, y_train_encoded]
            test_arr = np.c_[input_feature_test_arr, y_test_encoded]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)