import pandas as pd
import joblib
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/iris.csv")
X = df.drop(columns="target")
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

joblib.dump(model, "models/model.pkl")

mlflow.log_param("model", "logistic_regression")
mlflow.log_artifact("models/model.pkl")
