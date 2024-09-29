import pandas as pd
import numpy as np
import datetime as dtime

# 1
df = pd.DataFrame(np.random.random((10, 5)))

df["mean"] = df[:].apply(lambda x: np.mean(np.extract(x > 0.3, x)), axis=1)
print(df)

# 2


def date_func(row):
    print(row[3], row[4][:11])
    a = tuple(map(int, row[3].split("-")))
    a = dtime.date(year=a[0], month=a[1], day=a[2])
    b = tuple(map(int, row[4][:11].split("-")))
    b = dtime.date(year=b[0], month=b[1], day=b[2])
    return a - b


df_wi = pd.read_csv("wells_info.csv")
df_wi["Time"] = df_wi[:].apply(date_func, axis=1)
print(df_wi)
