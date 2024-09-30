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
    print(a, b, a - b)
    return ()


df_wi = pd.read_csv("wells_info.csv")
df_wi["Time"] = df_wi[:].apply(date_func, axis=1)
print(df_wi)
