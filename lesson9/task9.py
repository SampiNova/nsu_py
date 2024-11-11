import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# sklearn
from sklearn.metrics import accuracy_score
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


models = {
    'RandomForest': (RandomForestClassifier(), {
        'n_estimators': [50, 100],
        'max_depth': [None, 5, 10]}),
    'XGBoost': (XGBClassifier(), {
        'n_estimators': [50, 100],
        'max_depth': [3, 4, 5]}),
    'LogisticRegression': (LogisticRegression(), {'C': [0.1, 1, 10]}),
    'KNN': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]})
}

best_models = {}
for model_name, (model, params) in models.items():
    grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy')
    grid_search.fit(scaled_X_val, df_y_val)
    best_models[model_name] = grid_search.best_estimator_
    print(f"Best {model_name}: {grid_search.best_params_}")

print("==============================")
for model_name, model in best_models.items():
    y_pred = model.predict(scaled_X_test)
    accuracy = accuracy_score(df_y_test, y_pred)
    print(f"{model_name} Test Accuracy:", accuracy)
