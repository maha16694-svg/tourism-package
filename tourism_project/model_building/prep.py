
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset

dataset = load_dataset("maha16694-svg/tourism-package")

df = dataset["train"].to_pandas()

df = df.drop(columns=["CustomerID"])

df.fillna(method="ffill", inplace=True)

le = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

X = df.drop("ProdTaken", axis=1)
y = df["ProdTaken"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv("tourism_project/data/train.csv", index=False)
test_df.to_csv("tourism_project/data/test.csv", index=False)

print("Data preparation completed")
