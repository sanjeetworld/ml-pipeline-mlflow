from fastapi import FastAPI
import mlflow.sklearn

app = FastAPI()

# Load model (update path accordingly)
model = mlflow.sklearn.load_model("mlruns/0/<run_id>/artifacts/RandomForest")

@app.get("/")
def home():
    return {"message": "ML Model API Running"}

@app.get("/predict")
def predict():
    sample = [[30, 60000, 5, 1, 2]]
    result = model.predict(sample)
    return {"prediction": result.tolist()}