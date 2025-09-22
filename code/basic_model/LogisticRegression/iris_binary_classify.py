from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# loading iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Only take the first two classes for binary classification task
X = X[y != 2]
y = y[y != 2]

# devide dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train model (Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# predict on test set
y_pred = model.predict(X_test)

# evaluate model
print(f"classify accuracy: {accuracy_score(y_test, y_pred):.2f}")