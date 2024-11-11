import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# sklearn
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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
scaled_X_train = pd.DataFrame(scaler.fit_transform(prep_df_X_train))
scaled_X_val = pd.DataFrame(scaler.fit_transform(prep_df_X_val))
scaled_X_test = pd.DataFrame(scaler.fit_transform(prep_df_X_test))


rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(scaled_X_train, df_y_train)

feature_importances = rf_model.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]

for n_features in [2, 4, 8]:
    selected_features = sorted_indices[:n_features]
    selected_X_train = scaled_X_train.iloc[:, selected_features]
    selected_X_val = scaled_X_val.iloc[:, selected_features]
    selected_X_test = scaled_X_test.iloc[:, selected_features]

    rf_model_selected = RandomForestClassifier(random_state=42)
    rf_model_selected.fit(selected_X_train, df_y_train)
    y_pred = rf_model_selected.predict(selected_X_test)
    accuracy = accuracy_score(df_y_test, y_pred)
    print(f"Random Forest (Top {n_features} features) Test Accuracy:", accuracy)

print("==========================")
for i in selected_features:
    print(scaled_X_train.columns[i])
