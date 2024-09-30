import pandas as pd
import numpy as np
import datetime as dtime

# 1
df = pd.DataFrame(np.random.random((10, 5)))

df["mean"] = df[:].apply(lambda x: np.mean(np.extract(x > 0.3, x)), axis=1)
print(df)

# 2


def date_func(row):
    a = np.datetime64(row[3])
    b = np.datetime64(row[4])
    return int((a - b) / np.timedelta64(30, 'D'))


df_wi = pd.read_csv("wells_info.csv")
df_wi["Time"] = df_wi[:].apply(date_func, axis=1)
print(df_wi)

# 3
df_na = pd.read_csv('wells_info_na.csv')

df_na_filled = df_na.fillna({
    'LatWGS84': df_na['LatWGS84'].median(),
    'LonWGS84': df_na['LonWGS84'].median(),
    'BasinName': df_na['BasinName'].mode()[0],
    'StateName': df_na['StateName'].mode()[0],
    'CountyName': df_na['CountyName'].mode()[0]
})

df_na_filled.to_csv('wells_info_filled.csv', index=False)
