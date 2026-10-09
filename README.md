# Predictive Maintenance with KNN on Azure ML

## Overview
Binary classification of machine failure using the AI4I 2020 Predictive
Maintenance dataset. The KNN model is built three ways on Azure Machine
Learning (notebook, AutoML, Designer), tracked with MLflow, registered, and
the notebook model is deployed as a managed online endpoint.

## Repository Structure
- `data/`: cleaned dataset and prepared data
- `notebooks/`: data prep, KNN training, endpoint deployment
- `automl/`: AutoML experiment
- `designer/`: Designer KNN script and pipeline
- `deployment/`: scoring script and environment files
- `screenshots/`: lab evidence

## Model Input Schema
| Column | Type |
|---|---|
| air_temp_k | double |
| process_temp_k | double |
| rpm | double |
| torque_nm | double |
| tool_wear_min | double |
| type | string (L / M / H) |

Target: `machine_failure` (0 = no failure, 1 = failure)

## Results
- Accuracy: 0.964 (96.4%)
- Confusion matrix: see `notebooks/confusion_matrix.png`
- Endpoint test: two sample requests returned `[0, 1]`

## Deployment
- Managed online endpoint, deployment `blue` (100% traffic)
- Environment: scikit-learn 1.7.2, custom `score.py` using `mlflow.pyfunc`
- The endpoint was deleted after testing to avoid cost.
