from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# A simple dataset of house prices
data = {
    'area': [50, 60, 80, 100, 120],
    'price': [150, 180, 240, 300, 350]
}
df = pd.DataFrame(data)

# features and target
X = df[['area']]
y = df['price']

# devide dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train model (Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# predict on test set
y_pred = model.predict(X_test)

print(f"预测的房价: {y_pred}")