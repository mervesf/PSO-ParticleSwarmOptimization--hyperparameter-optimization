import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from pyswarm import pso

df = pd.read_csv('taxi_rides.csv')
df1=pd.read_csv('taxi_rides_clean.csv')
def objective_function(hyperparameters):
    n_estimators, contamination = hyperparameters
    model = IsolationForest(
        n_estimators=int(n_estimators),
        contamination=contamination,
        random_state=42
    )
    model.fit(df[['value']])
    df['anomaly'] = model.predict(df[['value']])
    TP = df[(df['anomaly'] == -1) & (df['true_label'] == -1)].shape[0]
    FP = df[(df['anomaly'] == -1) & (df['true_label'] == 1)].shape[0]
    FN = df[(df['anomaly'] == 1) & (df['true_label'] == -1)].shape[0]
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return -f1_score


hyperparameter_ranges = (np.array([50, 0.01]), np.array([150, 0.5]))

best_hyperparameters, best_objective_value = pso(objective_function, hyperparameter_ranges[0], hyperparameter_ranges[1])
print("En iyi F1 skoru:", -best_objective_value)

