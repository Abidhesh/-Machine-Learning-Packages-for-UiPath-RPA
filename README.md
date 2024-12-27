# -Machine-Learning-Packages-for-UiPath-RPA
A custom machine learning package for UiPath, designed to implement Linear Regression in RPA workflows. This package allows users to perform predictive modeling, analyze trends, and automate data-driven decisions with ease. Fully integrated with UiPath, it supports input from CSV, Excel, or other structured data sources.

# -Linear Regression Package for UiPath
# -Overview
This repository provides a custom machine learning package that integrates Linear Regression models into UiPath workflows. It includes scripts to train, evaluate, and deploy linear regression models, enabling intelligent and automated decision-making in RPA processes.

# -Features
Model Training: Train a Linear Regression model using structured data.
Model Evaluation: Assess the accuracy of the trained model.
Prediction: Use the trained model for predictions with JSON input.
Data Handling: Automatically preprocesses numeric, categorical, and date features for seamless training and evaluation.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
   
# -Setup and Installation

1.Clone the repository:

git clone <repository_url>
cd <repository_directory>

2. Install dependencies:

pip install -r requirements.txt

3. Set up environment variables:

artifacts_directory: Directory to store training artifacts (default: ./artifacts).
keep_training: Set to 'True' to enable incremental training.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# -Training the Model:

Run train.py to train a Linear Regression model:

python train.py

Place your training data (your.csv) in the specified training directory.

# -Evaluating the Model:

Use the evaluate() method in train.py to evaluate model performance on a test dataset.

# -Making Predictions:

Run main.py to use the trained model for predictions:

python main.py
Provide input data in JSON format to the predict() method.

HAPPY CODING 😊


