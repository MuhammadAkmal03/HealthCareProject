import sys
import os
import pandas as pd
from Prediction_module.src.exception import CustomException
from Prediction_module.src.utils import load_object

# Add project root to sys.path for correct module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

class PredictPipeline:
    
    """
    A class for making predictions on new data using a pre-trained model and preprocessor.
    """
    
    def __init__(self):
        self.preprocessor_path = os.path.join("Prediction_module", "artifacts", "preprocessor.pkl")
        self.model_path = os.path.join("Prediction_module", "artifacts", "model.pkl")

    def predict(self, features):
        
        """
        Loads the saved preprocessor and model, and makes a prediction on new data.
        
        Args:
            features (DataFrame): A pandas DataFrame containing the new data to be predicted.
                                  It must have the same columns as the training data.
        
        Returns:
            str: The predicted diagnosis as a string label.
        """
        
        try:
            preprocessor = load_object(file_path=self.preprocessor_path)
            model = load_object(file_path=self.model_path)
            
            # --- Feature Engineering for new data ---
            if 'Blood_Pressure_mmHg' in features.columns:
                features[['Systolic_BP', 'Diastolic_BP']] = features['Blood_Pressure_mmHg'].str.split('/', expand=True)
                features['Systolic_BP'] = pd.to_numeric(features['Systolic_BP'])
                features['Diastolic_BP'] = pd.to_numeric(features['Diastolic_BP'])
                features.drop('Blood_Pressure_mmHg', axis=1, inplace=True)
            if 'Patient_ID' in features.columns:
                features.drop('Patient_ID', axis=1, inplace=True)
            
            # --- Remove Severity and Treatment_Plan from new data ---
            if 'Severity' in features.columns:
                features.drop('Severity', axis=1, inplace=True)
            if 'Treatment_Plan' in features.columns:
                features.drop('Treatment_Plan', axis=1, inplace=True)

            data_scaled = preprocessor.transform(features)
            
            preds = model.predict(data_scaled)
            
            label_mapping = {
                0: 'Bronchitis',
                1: 'Cold',
                2: 'Flu',
                3: 'Healthy',
                4: 'Pneumonia'
            }
            
            predicted_label = label_mapping.get(preds[0], "Unknown")
            
            return predicted_label
        
        except Exception as e:
            raise CustomException(e, sys)

# The following code is for demonstration purposes.
if __name__ == "__main__":
    new_data = {
        'Patient_ID': [1001],
        'Age': [35],
        'Gender': ['Female'],
        'Symptom_1': ['Cough'],
        'Symptom_2': ['Fever'],
        'Symptom_3': ['Fatigue'],
        'Blood_Pressure_mmHg': ['120/80'],
        'Heart_Rate_bpm': [75],
        'Body_Temperature_C': [37.2],
        'Oxygen_Saturation_%': [98],
    }

    new_df = pd.DataFrame(new_data)
    
    print("New Data to Predict:")
    print(new_df)

    predict_pipeline = PredictPipeline()
    predicted_diagnosis = predict_pipeline.predict(new_df)
    
    print(f"\nThe predicted diagnosis is: {predicted_diagnosis}")