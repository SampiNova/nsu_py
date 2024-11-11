import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# xgboost
from xgboost import XGBClassifier
from xgboost.dask import predict

df_data = pd.read_csv("train.csv")
df_data = df_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# TRAIN and TEST partition (0.8 and 0.2)
df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df_data.drop("Survived", axis=1),
                                                                df_data["Survived"],
                                                                test_size=0.2,
                                                                random_state=42,
                                                                stratify=df_data["Survived"],
                                                                shuffle=True)

# VAL and TEST partition (0.5 and 0.5)
df_X_val, df_X_test, df_y_val, df_y_test = train_test_split(df_X_test,
                                                            df_y_test,
                                                            test_size=0.5,
                                                            random_state=42,
                                                            stratify=df_y_test,
                                                            shuffle=True)

# df_X_train - train data
# df_X_val & df_y_val - validation data
# df_X_test & df_y_test - testing data


def prepare_num(df):
    df_num = df.drop(['Sex', 'Embarked', 'Pclass'], axis=1)
    df_sex = pd.get_dummies(df['Sex'])
    df_emb = pd.get_dummies(df['Embarked'], prefix='Emb')
    df_pcl = pd.get_dummies(df['Pclass'], prefix='Pclass')

    df_num = pd.concat((df_num, df_sex, df_emb, df_pcl), axis=1)
    return df_num


# preparing
prep_df_X_train = prepare_num(df_X_train)
prep_df_X_val = prepare_num(df_X_val)
prep_df_X_test = prepare_num(df_X_test)
# NaN filling
prep_df_X_train = prep_df_X_train.fillna(prep_df_X_train.median())
prep_df_X_val = prep_df_X_val.fillna(prep_df_X_val.median())
prep_df_X_test = prep_df_X_test.fillna(prep_df_X_test.median())
# normalization
scaler = MinMaxScaler()
scaled_X_train = scaler.fit_transform(prep_df_X_train)
scaled_X_val = scaler.fit_transform(prep_df_X_val)
scaled_X_test = scaler.fit_transform(prep_df_X_test)


def find_hyper_params(model, param_grid, x_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(x_train, y_train)

    return grid_search.best_params_

# k-NN
knn_model = KNeighborsClassifier()

hyper_params = find_hyper_params(knn_model,
                                 {'n_neighbors': list(range(3, 10)),
                                            'weights': ['uniform', 'distance'],
                                            'metric': ['euclidean', 'manhattan']}, scaled_X_val, df_y_val)

knn_model = KNeighborsClassifier(n_neighbors=hyper_params["n_neighbors"],
                                 weights=hyper_params["weights"],
                                 metric=hyper_params["metric"])
knn_model.fit(scaled_X_train, df_y_train)
knn_predict = knn_model.predict(scaled_X_test)

print(hyper_params)
print("KNN accuracy:", accuracy_score(df_y_test, knn_predict))

# Logistic Regression
lr_model = LogisticRegression()
hyper_params = find_hyper_params(lr_model,
                                 {'C': np.linspace(0.01, 1, 100),
                                            'penalty': ['l1', 'l2'],
                                            'solver': ['liblinear', 'saga']},
                                 scaled_X_val, df_y_val)

lr_model = LogisticRegression(C=hyper_params["C"],
                              penalty=hyper_params["penalty"],
                              solver=hyper_params["solver"])
lr_model.fit(scaled_X_train, df_y_train)
lr_predict = lr_model.predict(scaled_X_test)

print(hyper_params)
print("LR accuracy:", accuracy_score(df_y_test, lr_predict))

# random forest classifier
rfc_model = RandomForestClassifier()
hyper_params = find_hyper_params(rfc_model,
                                 {
                                     'n_estimators': list(range(3, 300, 10)),
                                     'max_depth': list(range(3, 10)),
                                     'max_features': ['sqrt', 'log2']
                                 },
                                 scaled_X_val, df_y_val)

rfc_model = RandomForestClassifier(n_estimators=hyper_params["n_estimators"],
                                   max_depth=hyper_params["max_depth"],
                                   max_features=hyper_params["max_features"])
rfc_model.fit(scaled_X_train, df_y_train)
rfc_predict = rfc_model.predict(scaled_X_test)

print(hyper_params)
print("RFC accuracy:", accuracy_score(df_y_test, rfc_predict))

# xgboost
xgb_model = XGBClassifier()
hyper_params = find_hyper_params(xgb_model,
                                 {
                                     'n_estimators': list(range(3, 300, 10)),
                                     'max_depth': list(range(3, 10)),
                                     'subsample': [0.7, 0.8, 0.9]
                                 },
                                 scaled_X_val, df_y_val)

xgb_model = XGBClassifier(n_estimators=hyper_params["n_estimators"],
                          max_depth=hyper_params["max_depth"],
                          subsample=hyper_params["subsample"])
xgb_model.fit(scaled_X_train, df_y_train)
xgb_predict = rfc_model.predict(scaled_X_test)

print(hyper_params)
print("XGB accuracy:", accuracy_score(df_y_test, xgb_predict))
