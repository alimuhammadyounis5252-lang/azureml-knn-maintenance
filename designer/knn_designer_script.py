import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)

URL = "https://raw.githubusercontent.com/alimuhammadyounis5252-lang/azureml-knn-maintenance/main/data/ai4i_clean.csv"
TARGET = "machine_failure"
NUM_COLS = ["air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min"]
CAT_COLS = ["type"]
K = 1
P = 1
WEIGHTS = "uniform"

def azureml_main(dataframe1=None, dataframe2=None):
    df = pd.read_csv(URL)
    df[NUM_COLS] = df[NUM_COLS].astype(float)
    X, y = df[NUM_COLS + CAT_COLS], df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    model = Pipeline([("prep", prep),
                      ("knn", KNeighborsClassifier(n_neighbors=K, p=P, weights=WEIGHTS))])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    scored = X_test.copy()
    scored["actual"] = y_test.values
    scored["predicted_failure"] = pred
    scored["failure_probability"] = prob.round(3)

    metrics = pd.DataFrame({
        "metric": ["k", "p", "accuracy", "precision", "recall", "f1", "auc"],
        "value": [K, P, accuracy_score(y_test, pred),
                  precision_score(y_test, pred, zero_division=0),
                  recall_score(y_test, pred), f1_score(y_test, pred),
                  roc_auc_score(y_test, prob)],
    })
    print(metrics)
    return scored, metrics