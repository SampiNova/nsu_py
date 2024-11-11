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

df_data = pd.read_csv("train.csv")

# TRAIN and TEST partition (0.9 and 0.1)
df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df_data.drop("label", axis=1),
                                                                df_data["label"],
                                                                test_size=0.1,
                                                                random_state=42,
                                                                stratify=df_data["label"],
                                                                shuffle=True)

# NaN fillingscaled_X_val = pd.DataFrame(scaler.fit_transform(prep_df_X_val))
prep_df_X_train = df_X_train.fillna(df_X_train.median())
prep_df_X_test = dfread_fromcsv_X_test.fillna(df_X_test.median())
# normalization
scaler = MinMaxScaler()
scaled_X_train = pd.DataFrame(scaler.fit_transform(prep_df_X_train))
scaled_X_test = pd.DataFrame(scaler.fit_transform(prep_df_X_test))


