import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("student_exam_performance_dataset.csv")
df = df.drop(["student_id", "final_exam_score", "grade_category"], axis=1)

X = df.drop("pass_fail", axis=1)
y = df["pass_fail"]

num_cols = X.select_dtypes(include="number").columns
cat_cols = X.select_dtypes(exclude="number").columns

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols)
])

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000)
}

results = {name: {"acc": []} for name in models}

for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    for name, m in models.items():
        pipe = Pipeline([
            ("preprocess", preprocess),
            ("classifier", m)
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results[name]["acc"].append(accuracy_score(y_test, pred))

avg_acc = np.mean(results["LogisticRegression"]["acc"])
print(f"\n{name}")
print(f"accuracy: {avg_acc}")

names = list(models.keys())
acc_data = [results[name]["acc"] for name in names]

plt.boxplot(acc_data, tick_labels=names)
plt.title('accuracy distribution')
plt.show()
