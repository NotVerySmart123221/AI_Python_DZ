import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import r2_score

df = pd.read_csv("Student Performance and Behaviour.csv")

X = df.drop(["Final_Exam_Score", "Student_ID", "Semester_ID"], axis=1)
y = df["Final_Exam_Score"]

num_cols = X.select_dtypes(include="number").columns
cat_cols = X.select_dtypes(exclude="number").columns

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

models = {
    "linear": LinearRegression(),
    "lasso": Lasso(alpha=1.0),
    "ridge": Ridge(alpha=1.0),
    "elasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5)
}

colors = ['blue', 'green', 'orange', 'purple']
scores = []

plt.figure(figsize=(13, 8))
for (name, m), color in zip(models.items(), colors):
    pipe = Pipeline([
        ("prep", preprocess),
        ("reg", m)
    ])
    
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    
    score = r2_score(y_test, pred)
    scores.append(score)
    plt.scatter(y_test, pred, alpha=0.2, s=3, c=color)

plt.plot([y.min(), y.max()], [y.min(), y.max()])
plt.xlabel("true")
plt.ylabel("pred")
plt.show()

plt.bar(models.keys(), scores, color=colors)
plt.show()
