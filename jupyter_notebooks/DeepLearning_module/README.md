Image Prediction Module

Overview

The image_prediction module is designed to classify brain tumor MRI images using deep learning models.
It supports custom CNN models as well as pre-trained architectures like VGG16 and ResNet for medical image classification.

Directory Structure
image_prediction/
├── notebook/             # Jupyter notebooks for experimentation
│   ├── brain-tumor-custom.ipynb
│   ├── brain-tumor-vgg.ipynb
│   └── braintumor-mri_resnet.ipynb
├── trained_models/       # Saved trained models (.h5 / joblib)
├── logs/                 # Training logs
├── exception.py          # Custom exception handling
├── logger.py             # Logging utility
├── image_prediction.ipynb# Main prediction notebook
├── VGG16.ipynb           # VGG16 specific implementation
└── requirements.txt      # Dependencies for this module

Features

Custom CNN: Train and test your own convolutional network for MRI image classification.

Pre-trained Models: Use VGG16 or ResNet models for transfer learning.

Data Preprocessing: Resize, normalize, and augment MRI images.

Evaluation: Accuracy, confusion matrix, and classification report.

Logging & Exception Handling: Track model training and handle errors.

Getting Started

Install dependencies:

pip install -r requirements.txt


Run a Jupyter notebook for your preferred model:

jupyter notebook brain-tumor-custom.ipynb


Train your model and save it to trained_models/:

model.save('trained_models/custom_cnn_model.h5')


Make predictions on new MRI images:

from image_prediction import predict_image
predict_image('path_to_new_image.jpg')

Usage

Place your MRI images in a folder or provide paths directly in the notebook.

Use the pre-trained models or train your custom model.

Output includes predicted class and confidence score.

Logging

Logs are saved in the logs/ directory.

Errors and exceptions are handled via exception.py.

Dependencies

Python 3.8+

TensorFlow / Keras

NumPy, Pandas, Matplotlib, Seaborn

scikit-learn