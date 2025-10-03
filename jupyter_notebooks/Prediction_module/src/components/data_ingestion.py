# import os
# import sys
# from dataclasses import dataclass
# import pandas as pd
# from sklearn.model_selection import train_test_split

# # --- FIX: Add project root to sys.path to resolve ModuleNotFoundError ---
# # This block makes the script runnable from any directory
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# if project_root not in sys.path:
#     sys.path.append(project_root)

# # Now, import your local modules using the full path
# from Prediction_module.src.exception import CustomException
# from Prediction_module.src.logger import logging
# from Prediction_module.src.components.data_transformation import DataTransformation
# from Prediction_module.src.components.model_trainer import ModelTrainer

# @dataclass
# class DataIngestionConfig:
#     """
#     Configuration class for data ingestion paths.
#     """
#     artifact_dir: str = os.path.join('Prediction_module', 'artifacts')
#     train_data_path: str = os.path.join(artifact_dir, "train.csv")
#     test_data_path: str = os.path.join(artifact_dir, "test.csv")
#     raw_data_path: str = os.path.join(artifact_dir, "data.csv")

# class DataIngestion:
#     """
#     A class for ingesting and splitting the raw data.
#     """
#     def __init__(self):
#         self.ingestion_config = DataIngestionConfig()

#     def initiate_data_ingestion(self):
#         """
#         Reads the raw data, performs a train-test split,
#         and saves the raw, training, and testing datasets to disk.

#         Returns:
#             tuple: A tuple containing the paths to the saved training and testing data.
#                    (train_data_path, test_data_path)
#         """
#         logging.info("Entered the data ingestion method or component")
#         try:
#             os.makedirs(self.ingestion_config.artifact_dir, exist_ok=True)
#             logging.info(f"Created or verified artifacts directory at {self.ingestion_config.artifact_dir}")

#             df = pd.read_csv('Prediction_module/Data/disease_diagnosis.csv')
#             logging.info('Read the dataset as a dataframe')

#             df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
#             logging.info(f"Saved raw data to {self.ingestion_config.raw_data_path}")

#             logging.info("Train test split initiated")
#             train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Diagnosis'])

#             train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
#             test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
#             logging.info(f"Saved train data to {self.ingestion_config.train_data_path}")
#             logging.info(f"Saved test data to {self.ingestion_config.test_data_path}")

#             logging.info("Ingestion of the data is completed")

#             return (
#                 self.ingestion_config.train_data_path,
#                 self.ingestion_config.test_data_path
#             )
#         except Exception as e:
#             raise CustomException(e, sys)

# if __name__ == "__main__":
#     obj = DataIngestion()
#     train_data, test_data = obj.initiate_data_ingestion()

#     data_transformation = DataTransformation()
#     train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

#     modeltrainer = ModelTrainer()
#     print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
import os
import sys
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

# --- FIX: Add project root to sys.path to resolve ModuleNotFoundError ---
# This block makes the script runnable from any directory
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if project_root not in sys.path:
    sys.path.append(project_root)

# Now, import your local modules using the full path
from Prediction_module.src.exception import CustomException
from Prediction_module.src.logger import logging


@dataclass
class DataIngestionConfig:
    """
    Configuration class for data ingestion paths.
    """

    artifact_dir: str = os.path.join("Prediction_module", "artifacts")
    train_data_path: str = os.path.join(artifact_dir, "train.csv")
    test_data_path: str = os.path.join(artifact_dir, "test.csv")
    raw_data_path: str = os.path.join(artifact_dir, "data.csv")


# data_ingestion.py


class DataIngestion:
    """
    A class for ingesting and splitting the raw data.
    """

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Reads the raw data, performs a train-test split,
        and saves the raw, training, and testing datasets to disk.

        Returns:
            tuple: A tuple containing the paths to the saved training and testing data.
                   (train_data_path, test_data_path)
        """
        logging.info("Entered the data ingestion method or component")
        try:
            os.makedirs(self.ingestion_config.artifact_dir, exist_ok=True)
            logging.info(
                f"Created or verified artifacts directory at {self.ingestion_config.artifact_dir}"
            )

            # FIX: Get the path to the data file relative to the script's location
            script_dir = os.path.dirname(__file__)
            data_file_path = os.path.join(
                script_dir, "..", "..", "Data", "disease_data1.csv"
            )

            df = pd.read_csv(data_file_path)
            logging.info("Read the dataset as a dataframe")

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Saved raw data to {self.ingestion_config.raw_data_path}")

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(
                df, test_size=0.25, random_state=42, stratify=df["Diagnosis"]
            )

            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )
            logging.info(f"Saved train data to {self.ingestion_config.train_data_path}")
            logging.info(f"Saved test data to {self.ingestion_config.test_data_path}")

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
