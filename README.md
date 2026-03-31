# End-to-End ML Pipeline with MLflow 🚀

This project demonstrates a complete ML pipeline including model training, experiment tracking, and deployment using FastAPI.

## 🔥 Features
- Model training using Scikit-learn
- Experiment tracking with MLflow
- Model versioning
- FastAPI deployment
- REST API for predictions

## 🛠️ Tech Stack
- Python
- Scikit-learn
- MLflow
- FastAPI

## ⚙️ Setup

```bash
pip install -r requirements.txt
python train.py
mlflow ui
uvicorn app:app --reload
