import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# xgboost
from xgboost import XGBClassifier
from xgboost.dask import predict

df_data = pd.read_csv("titanic_prepared.csv")

# TRAIN and TEST partition (0.9 and 0.1)
df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df_data.drop("label", axis=1),
                                                                df_data["label"],
                                                                test_size=0.1,
                                                                random_state=42,
                                                                stratify=df_data["label"],
                                                                shuffle=True)

# NaN fillingscaled_X_val = pd.DataFrame(scaler.fit_transform(prep_df_X_val))
prep_df_X_train = df_X_train.fillna(df_X_train.median())
prep_df_X_test = df_X_test.fillna(df_X_test.median())
# normalization
scaler = MinMaxScaler()
scaled_X_train = pd.DataFrame(scaler.fit_transform(prep_df_X_train))
scaled_X_test = pd.DataFrame(scaler.fit_transform(prep_df_X_test))

models_all_params = {"LR": LogisticRegression(C=1),
                     "DT": DecisionTreeClassifier(max_depth=5),
                     "XGB": XGBClassifier(max_depth=5, n_estimators=50)}

for name in models_all_params:
    models_all_params[name].fit(scaled_X_train, df_y_train)
    print(f"{name} accuracy:", accuracy_score(df_y_test, models_all_params[name].predict(scaled_X_test)))

print("======================")

feature_importances = models_all_params["DT"].feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]

selected_features = sorted_indices[:2]
selected_X_train = scaled_X_train.iloc[:, selected_features]
selected_X_test = scaled_X_test.iloc[:, selected_features]

models_two_params = {"LR": LogisticRegression(C=1),
                     "DT": DecisionTreeClassifier(max_depth=5),
                     "XGB": XGBClassifier(max_depth=5, n_estimators=50)}

for name in models_two_params:
    models_two_params[name].fit(selected_X_train, df_y_train)
    print(f"{name} accuracy:", accuracy_score(df_y_test, models_two_params[name].predict(selected_X_test)))
