from mlflow import sklearn as mlflow_sklearn
import numpy as np

# Load model (update run_id properly)
model = mlflow_sklearn.load_model("mlruns/0/<run_id>/artifacts/RandomForest")

# Correct input (5 features)
sample = np.array([[30, 60000, 5, 1, 2]])

prediction = model.predict(sample)

print(prediction)