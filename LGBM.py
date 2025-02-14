import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np



data = pd.read_csv("/content/Cleaned_Data_Final.csv")

X = data.drop(columns=['AQI', 'From Date', 'To Date'])
y = data['AQI']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


lgb_model = lgb.LGBMRegressor(random_state=42)
lgb_model.fit(X_train, y_train)
lgb_y_pred = lgb_model.predict(X_test)


lgb_mae = mean_absolute_error(y_test, lgb_y_pred)
lgb_mse = mean_squared_error(y_test, lgb_y_pred)
lgb_r2 = r2_score(y_test, lgb_y_pred)
lgb_rmse = np.sqrt(lgb_mse)

print("LightGBM Results:")
print(f"RMSE: {lgb_rmse}")
print(f"MAE: {lgb_mae}")
print(f"MSE: {lgb_mse}")
print(f"R²: {lgb_r2}")


plt.scatter(y_test, lgb_y_pred, alpha=0.5, label='LightGBM Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', label='Ideal Fit')
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI (LightGBM)")
plt.legend()
plt.show()
