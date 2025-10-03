Prediction_module README
This README provides an overview of the Prediction_module, a machine learning project designed to predict disease diagnoses based on patient vitals and symptoms. The module is built as a production-ready pipeline, separating different stages of the machine learning workflow into dedicated scripts.

1. Project Structure 
The project is organized into a modular structure to ensure maintainability and scalability.

Prediction_module/
├── Data/
│   └── disease_diagnosis.csv  # Raw dataset
├── artifacts/
│   ├── data.csv                 # Raw data copy
│   ├── train.csv                # Training data after split
│   ├── test.csv                 # Testing data after split
│   ├── preprocessor.pkl         # Saved preprocessing object
│   └── model.pkl                # Saved trained model
└── src/
    ├── components/
    │   ├── __init__.py
    │   ├── data_ingestion.py    # Reads and splits data
    │   ├── data_transformation.py # Handles feature engineering and preprocessing
    │   └── model_trainer.py     # Trains and evaluates models
    ├── pipeline/
    │   ├── __init__.py
    │   └── predict_pipeline.py  # Makes predictions on new data
    ├── __init__.py
    ├── exception.py           # Custom exception handling
    ├── logger.py              # Logging setup
    └── utils.py               # Utility functions (save/load objects, evaluate models)
2. Setup and Installation 
Follow these steps to set up the project and install the required dependencies.

Create a Virtual Environment (recommended to avoid conflicts):

Bash

python -m venv venv
Activate the Virtual Environment:

Bash

# On Windows
.\venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
Install Required Libraries:
You'll need pandas, scikit-learn, numpy, and seaborn for this project.

Bash

pip install pandas scikit-learn numpy seaborn matplotlib
3. Usage: Training and Prediction 
a. Training the Model
To train the machine learning pipeline and save the trained model and preprocessor, run the main data_ingestion.py script. This single command executes the entire pipeline from start to finish.

Place the disease_diagnosis.csv file in the Prediction_module/Data/ directory.

Run the script from your project's root directory (E:\HealthCareProject).

Bash

python Prediction_module/src/components/data_ingestion.py
After execution, the artifacts folder will be populated with the saved model (model.pkl) and the data preprocessor (preprocessor.pkl).

b. Making Predictions
To use the trained model for making predictions on new data, run the predict_pipeline.py script.

Ensure you have already run the training script to generate the artifacts folder.

Run the script from your project's root directory.

Bash

python Prediction_module/src/pipeline/predict_pipeline.py
The script includes an example data point and will print the predicted diagnosis to the terminal.

4. Key Components Explained 
data_ingestion.py: The starting point of the pipeline. It loads the raw data, performs an 80/20 train-test split, and saves the data files to the artifacts folder.

data_transformation.py: This module handles all data preprocessing. It performs feature engineering (like splitting the blood pressure column) and fits a ColumnTransformer to scale numerical features and encode categorical ones. The fitted preprocessor is saved for future use in the prediction pipeline.

model_trainer.py: This component trains and evaluates multiple classification models (e.g., Random Forest and Logistic Regression) and uses a hyperparameter search to find the best-performing one. The best model is then saved as model.pkl.

predict_pipeline.py: This is the inference pipeline. It loads the saved preprocessor and model, applies the same preprocessing steps to new data, and uses the model to make and return a final prediction.

utils.py: A helper module containing generic functions to save and load Python objects using pickle, and to evaluate the performance of multiple models.