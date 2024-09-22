# SecureBank Transaction Fraud Detection System
This is a transaction fraud detection system designed to identify and prevent fraudulent activities with a precision and recall score of 54% and 69%, respectively, which is a significant improvement over its predecessor. Leveraging a supervised ensemble learning model, the Random Forest Classifier, this system ensures that legitimate transactions are processed seamlessly while suspicious activities are flagged for further investigation.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#clone-the-repository)
  - [Build the Docker Image](#build-the-docker-image)
  - [Run the Docker Container](#run-the-docker-container)
- [Usage](#usage)
  - [API Endpoints](#api-endpoints)
    - [/health Endpoint](#health-endpoint)
    - [/predict Endpoint](#predict-endpoint)
  - [Making Predictions](#making-predictions)
  - [Administrator Functions](#administrator-functions)
    - [Generate New Training Dataset](#generate-new-training-dataset)
    - [Select Pre-trained Models](#select-pre-trained-models)
    - [Audit System Performance](#audit-system-performance)
- [How the System Meets Requirements](#how-the-system-meets-requirements)
  - [R1: Improved Model Performance](#r1-improved-model-performance)
  - [R2: Transaction Legitimacy Prediction](#r2-transaction-legitimacy-prediction)
  - [R3: Generate New Training Dataset](#r3-generate-new-training-dataset)
  - [R4: Select from Pre-trained Models](#r4-select-from-pre-trained-models)
  - [R5: Audit System Performance](#r5-audit-system-performance)
- [Project Structure](#project-structure)
- [Logging](#logging)
- [License](#license)

## Features
- Fraud Detection: Analyze transactions to identify potential fraud.
- Scalable Architecture: Docker-enabled for easy deployment and scalability.
- Model Versioning: Support for multiple pre-trained models, allowing administrators to select and switch between them seamlessly.
- Comprehensive Logging: Detailed logs for monitoring system performance and auditing purposes.
- Versatile Toolset: Includes capabilities for dataset generation, model selection, and performance auditing to support various machine learning workflows.

## Requirements
Before setting up the system, ensure that you have the following:

- Docker: Version 20.10+
- Git: For cloning the repository.
- JSON Payloads: Properly formatted JSON files for making predictions.

## Installation
### Prerequisites
1. Install Docker:

    - Windows & macOS:
        - Download Docker Desktop.
        - Follow the installation instructions.
    - Linux:
        - Follow the official Docker installation guide for your distribution.

2. Install Git:

    - Download and install Git.

### Clone the Repository
Copy and paste the following code into your command terminal:
`git clone https://github.com/creating-ai-enabled-systems-fall-2024/chang-waldemar.git`
`cd securebank`

### Build the Docker Image
Ensure you're in the root directory `securebank` of the cloned repository, which contains the Dockerfile, the Flask app, and other important files.

Copy and paste the following code into your command terminal:
`docker build -t securebank .`

### Run the Docker Container
Copy and paste the following code into your command terminal:
`docker run -d -p 5000:5000 --name securebank-container securebank`

Explanation:
`-d`: Run the container in detached mode.
`-p 5000:5000`: Map port 5000 of the host to port 5000 of the container.
`--name securebank-container`: Assign a name to the running container.
`securebank`: Specify the image to run.

## Usage
Once the Docker container is running, you can interact with the system through its API endpoints.

### API Endpoints
#### /health Endpoint
- Description: Health check to verify if the server is running.
- Method: GET
- URL: http://localhost:5000/health
- Response:
    {
        "status": "running"
    }

#### /predict Endpoint
- Description: Predicts whether a transaction is legitimate or fraudulent.
- Method: POST
- URL: http://localhost:5000/predict
- Headers:
    - Content-Type: application/json
- Body: JSON object containing transaction details.
- Response:
    {
        "prediction": "fraud" | "legitimate"
    }

## Making Predictions
### Prepare a JSON Payload:

Create a `test.json` file with the following structure:
    {
        "unix_time": "1.314248",
        "merchant": "0.001097",
        "amt": "5.363884",
        "merch_lat": "-0.521162",
        "merch_long": "0.193988",
        "transaction_hour": "23",
        "time_since_last_trans": "0"
    }

Another acceptable, but suboptimal, structure (from Assignment 3) is as follows:
    {
        "trans_date_trans_time": "2024-09-21 10:00:00",
        "cc_num": "1234567890123456",
        "unix_time": 1716240000,
        "merchant": "MerchantXYZ",
        "category": "electronics",
        "amt": "250.75",
        "merch_lat": "37.7749",
        "merch_long": "-122.4194",
    }

Ideally, `time_since_last_trans` should be included, if available.

##### Field Descriptions:
- trans_date_trans_time: Date and time of the transaction.
- cc_num: Credit card number.
- unix_time: Unix timestamp of the transaction.
- merchant: Merchant name.
- category: Transaction category.
- amt: Transaction amount.
- merch_lat: Merchant latitude.
- merch_long: Merchant longitude.
- time_since_last_trans: Time in seconds since the last transaction.

##### Send a Prediction Request Using PowerShell:
Copy and paste the following code into your command terminal:
`Invoke-WebRequest -Uri 'http://localhost:5000/predict' -Method POST -ContentType 'application/json' -Body (Get-Content -Path 'test.json' -Raw)`

##### Expected Response:
    {
        "prediction": "fraud"
    }
or
    {
        "prediction": "legitimate"
    }

### Administrator Functions
#### Generate New Training Dataset
Administrators can generate a new training dataset by aggregating data from various sources.

1. Run the Dataset Generation Function:

Copy and paste the following code into your command terminal:
`docker exec -it securebank-container python -c "from pipeline import Pipeline; p = Pipeline(); p.create_data()"`

Description: This script pulls data from `customer_release.csv`, `transactions_release.parquet`, and `fraud_release.json`, processes it, and generates a new training dataset.

2. Verify Dataset Creation:

Check the logs or output directory to ensure the new dataset has been created successfully.

#### Select Pre-trained Models
Administrators can choose from a catalog of pre-trained models to deploy.

1. List Available Models:

`docker exec -it securebank-container ls storage/models/artifacts/`

##### Example Output:

Dummy.joblib
Extra Trees.joblib
Logistic Regression.joblib
Random Forest.joblib
Trained.joblib

2. Select a Model:

To select a specific model version, specify the version as a parameter when instantiating the Pipeline class. This can be done by running a Python command inside the Docker container.

`docker exec -it securebank-container python -c "from pipeline import Pipeline; p = Pipeline(model_version='Advanced'); p.load_model()"`

3. Restart the Container to Apply Changes:

`docker restart securebank-container`

##### Audit System Performance
Administrators can audit the system's performance by reviewing logs and performance metrics.

1. Access Logs:

`docker exec -it securebank-container cat model_training.log`

Description: This command displays the contents of the `model_training.log` file, which records details about the training process, including performance metrics and any issues encountered.

2. Access Additional Output Files:

The `train_eval_model` and `get_history` functions within pipeline.py generate various output files such as classification reports and precision-recall curves. These files are saved in the root directory of the Docker container.

##### Generated Files Include:

- Dummy_classification_report.txt
- Dummy_confusion_matrix.txt
- Dummy_precision_recall_curve.png
- Dummy_precision_recall_curve_with_optimal_threshold.png

##### Accessing These Files:

To view or retrieve these files, you can use the `docker cp` command to copy them from the container to your local machine.

##### Example:

`docker cp securebank-container:/securebank/Dummy_classification_report.txt ./Dummy_classification_report.txt`
`docker cp securebank-container:/securebank/Dummy_precision_recall_curve.png ./Dummy_precision_recall_curve.png`


## How the System Meets Requirements
### R1: Improved Model Performance
- Implementation:
    - Utilized advanced preprocessing techniques and optimized model hyperparameters.
    - Implemented SMOTE for handling class imbalance, enhancing the model's ability to detect fraudulent transactions.
- Result:
    - Achieved higher precision and recall (by 24% and 9%, respectively) compared to the previous iteration, resulting in a final precision of 54% and a final recall of 69%.
### R2: Transaction Legitimacy Prediction
- Implementation:
    - Developed a Flask API with a /predict endpoint that analyzes transaction data.
    - The system returns clear predictions indicating whether a transaction is "legitimate" or "fraud".
- Usage:
    - Users can send transaction details via JSON payloads to receive immediate fraud assessments.
### R3: Generate New Training Dataset
- Implementation:
    - Created the create_data function within pipeline.py that aggregates data from multiple sources (customer_release.csv, transactions_release.parquet, fraud_release.json).
    - Ensured data preprocessing steps are consistent and reproducible.
- Usage:
    - Administrators can run the dataset generation function to refresh and resample the training data, ensuring the model stays updated with the latest transaction patterns.
### R4: Select from Pre-trained Models
- Implementation:
    - Maintained a catalog of pre-trained models stored in `storage/models/artifacts/`.
    - Enabled model selection by specifying the `version` parameter during the instantiation of the `Pipeline` class, allowing administrators to switch models without modifying the codebase.
- Usage:
    - Administrators can list available models and select the desired one by setting the appropriate `model_version` parameter and restarting the Docker container.
### R5: Audit System Performance
- Implementation:
    - Integrated comprehensive logging mechanisms that record prediction results, system status, and errors.
    - The `train_eval_model` and `get_history functions` within pipeline.py generate additional output files (classification reports and precision-recall curves) for performance evaluation.
- Usage:
    - Administrators can access `model_training.log` directly to review training details.
    - Additional output files can be retrieved using `docker cp` for further analysis.

## Project Structure

securebank/
├── Dockerfile
├── Dummy_classification_report.txt
├── Dummy_confusion_matrix.txt
├── Dummy_precision_recall_curve.png
├── Dummy_precision_recall_curve_with_optimal_threshold.png
├── app.py
├── dataset_design.py
├── feature_extractor.py
├── model_training.log
├── pipeline.py
├── raw_data_handler.py
├── requirements.txt
├── test.json
├── analysis/
│   ├── data_analysis.ipynb  
│   └── model_performance.ipynb      
├── modules/
│   ├── __pycache__/
│   └── Data_Pipeline_Design.md
└── README.md

- app.py: Flask application defining API endpoints.
- Dockerfile: Instructions to build the Docker image.
- dataset_design.py: Script for designing and preprocessing datasets.
- feature_extractor.py: Module for extracting features from raw data.
- model_training.log: Log file capturing details of the model training process.
- pipeline.py: Contains the Pipeline class handling model loading, training, prediction, and dataset generation via the create_data function.
- raw_data_handler.py: Module for handling and processing raw input data.
- requirements.txt: Python dependencies.
- test.json: Sample JSON payload for making predictions.
- analysis/: Directory containing Jupyter notebooks for data analysis and model performance evaluation.
- data_analysis.ipynb: Notebook for analyzing raw and processed data from Assignment 1.
- model_performance.ipynb: Notebook for evaluating model performance metrics from Assignment 3.
- modules/: Directory containing the original (deprecated) modules and documentation that have not been updated.
- pycache/: Compiled Python files.
- Data_Pipeline_Design.md: Original (deprecated) documentation detailing the data pipeline design.
- README.md: Current project documentation.

## Logging
The system employs comprehensive logging to facilitate monitoring and auditing:
- Model Training Logs (model_training.log):
    - Records details about the training process, including performance metrics and any issues encountered.
- Additional Output Files:
    - {version}_classification_report.txt: Classification report detailing precision, recall, F1-score, etc.
    - {version}_confusion_matrix.txt: Confusion matrix showing actual vs. predicted labels.
    - {version}_precision_recall_curve.png: Plot of the Precision-Recall Curve.
    - {version}_precision_recall_curve_with_optimal_threshold.png: Precision-Recall Curve annotated with the optimal threshold.

These additional output files are generated by the `train_eval_model` and `get_history` functions within `pipeline.py` and are saved in the root directory of the Docker container.

##### Accessing Logs and Output Files:

- Access model_training.log:

`docker exec -it securebank-container cat model_training.log`

- Access Additional Output Files:

To retrieve these files from the container, use the `docker cp` command:

`docker cp securebank-container:/securebank/Dummy_classification_report.txt ./Dummy_classification_report.txt`
`docker cp securebank-container:/securebank/Dummy_precision_recall_curve.png ./Dummy_precision_recall_curve.png`

(You may need to adjust the paths as necessary based on your Docker setup.)

## License
This project is licensed under the MIT License.