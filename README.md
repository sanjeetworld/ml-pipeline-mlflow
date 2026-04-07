# End-to-End ML Pipeline with MLflow

## Overview

This project demonstrates a complete machine learning pipeline including data generation, model training, experiment tracking, and deployment using MLflow and FastAPI.
It follows an industry-level workflow for building, tracking, and serving ML models.

---

## Features

* Synthetic dataset generation
* Model training using Scikit-learn
* Experiment tracking with MLflow
* Model evaluation using accuracy
* Model versioning and storage
* REST API deployment using FastAPI

---

## Tech Stack

* Language: Python
* Libraries: Scikit-learn, Pandas, NumPy
* Tools: MLflow, FastAPI, Uvicorn

---

## Project Structure

```
ml-pipeline-project/
│
├── data/
│   └── data.csv
├── mlruns/              
├── generate_data.py     
├── train.py             
├── predict.py           
├── app.py               
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```
git clone https://github.com/your-username/ml-pipeline-project.git
cd ml-pipeline-project
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

## Usage

### Step 1: Generate Dataset

```
python generate_data.py
```

### Step 2: Train Model

```
python train.py
```

### Step 3: Launch MLflow UI

```
python -m mlflow ui
```

Open in browser:

```
http://127.0.0.1:5000
```

### Step 4: Run Prediction Script

```
python predict.py
```

### Step 5: Run FastAPI Server

```
uvicorn app:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## Model Details

* Algorithm: Random Forest Classifier
* Features:

  * Age
  * Salary
  * Experience
  * Education Level
  * City Tier
* Target:

  * Purchased (0/1)

---

## MLflow Tracking

MLflow is used to:

* Log parameters (e.g., number of estimators)
* Track performance metrics (accuracy)
* Store trained models
* Manage experiment runs

---

## API Endpoint

### GET /predict

Returns prediction based on input sample.

Example response:

```
{
  "prediction": [1]
}
```

---

## Future Improvements

* Add multiple model comparison
* Hyperparameter tuning
* Docker deployment
* CI/CD integration
* Real-world dataset integration

---

## Author

Sanjeet Kumar
AI/ML Engineer
