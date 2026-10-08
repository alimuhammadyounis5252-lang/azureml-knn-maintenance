
import os, json
import mlflow.pyfunc
import pandas as pd

def init():
    global model
    base = os.getenv("AZUREML_MODEL_DIR")
    model_path = base
    for root, _, files in os.walk(base):
        if "MLmodel" in files:
            model_path = root
            break
    model = mlflow.pyfunc.load_model(model_path)

def run(raw_data):
    data = json.loads(raw_data)["input_data"]
    if isinstance(data, dict):
        df = pd.DataFrame(data["data"], columns=data.get("columns"))
    else:
        df = pd.DataFrame(data)
    return model.predict(df).tolist()
