import pandas as pd
import numpy as np

df = pd.read_csv("titanic_with_labels.csv")

print(df)
df = ~df["sex"].isna()
print(df)
