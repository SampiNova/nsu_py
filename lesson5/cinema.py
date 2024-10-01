import pandas as pd
import numpy as np

df = pd.read_csv("titanic_with_labels.csv", delimiter=" ")

print(df)
# df = df.loc[df["sex"] != "-"]
# df = df.loc[df["sex"] != "Не указан"]
df["sex"] = df["sex"].map({"М": 0, "м": 0, "Ж": 1, "ж": 1})
df = df.loc[~df["sex"].isna()]
df = df.fillna({"row_number": df["row_number"].mode()[0]})
# ===================================================================
q_low, q_high = df["liters_drunk"].quantile([0.25, 0.75])
iqr = q_high - q_low
low, high = q_low - 1.5 * iqr, q_high + 1.5 * iqr
ld = df["liters_drunk"]
df["liters_drunk"] = np.where((ld < low) | (ld > high), ld.mean(), ld)
print(df)
