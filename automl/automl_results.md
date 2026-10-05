# Automated ML results (KNN only)
- Job name: knn-automl-ali886-v2
- Best algorithm (scaler + model): StandardScaler + KNeighborsClassifier
- n_neighbors: 100
- weights: distance
- metric: l1 (Manhattan)
- AUC weighted (5-fold CV): 0.951
- Recall (failure class, 5-fold CV): 0.004 (1 of 271 failures caught)
- F1 (failure class, 5-fold CV): ~0.007
- Note 1: first attempt failed ("all allow-listed algorithms are incompatible
  with sparse datasets") because KNN does not support one-hot (sparse) data.
  Fixed by ordinal-encoding type (L=0, M=1, H=2) before AutoML.
- Note 2: AutoML also produced a VotingEnsemble (AUC 0.953); the best pure KNN
  is the StandardScaler model above.
- Observation: AutoML chose K=100, distance weighting, Manhattan distance
  (notebook: K=1, uniform, p=1). AutoML optimised AUC, so ranking quality is
  high (0.95), but at the default 0.5 threshold it almost never predicts
  failure (balanced accuracy ~0.50). Optimising recall/F1 or lowering the
  
  threshold would be needed.