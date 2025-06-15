import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


data = pd.read_csv('Cleaned_Data_Final.csv')





data['From Date'] = pd.to_datetime(data['From Date'], errors='coerce')
data['To Date'] = pd.to_datetime(data['To Date'], errors='coerce')



X = data.drop(columns=['AQI', 'From Date', 'To Date'])
y = data['AQI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


svm_model = SVR(kernel='rbf', C=150, gamma=0.6, epsilon=0.1)

svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R^2 Score: {r2}")
