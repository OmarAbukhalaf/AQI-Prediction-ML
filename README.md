# 🌍 Air Quality Prediction Using Machine Learning

A machine learning-based system for predicting the **Air Quality Index (AQI)** using pollution and meteorological data. This project applies several ML models, including ensemble methods and deep learning (LSTM), to forecast AQI values with high accuracy, helping mitigate health risks from air pollution.



## 📝 Abstract

The Air Quality Index (AQI) is crucial for monitoring environmental and public health. This project implements a range of machine learning models including:

- Linear Regression  
- Random Forest  
- Support Vector Machines (SVM)  
- XGBoost  
- LightGBM  
- Long Short-Term Memory (LSTM) Neural Network  

**Results**:  
- 🥇 **XGBoost** achieved the best performance with an **RMSE of 2.79** and **R² of 0.99**  
- 🥈 **LightGBM** followed closely with **RMSE of 3.32** and **R² of 0.98**

---

## 📊 Dataset

- **Source**: Central Pollution Board - Central Control Room for Air (CPCBCRR), Visakhapatnam, India  
- **Size**: ~1900 daily readings across 5 years  
- **Features**: 20+ including PM2.5, NO₂, CO, SO₂, temperature, humidity, wind speed, etc.

### ⚙️ Preprocessing

- Handled missing data with **mean imputation**  
- Addressed skewed distributions with **log/sqrt/Yeo-Johnson transforms**  
- Performed **feature scaling using Min-Max normalization**  
- Applied **correlation-based feature selection**

---

## 🔍 Models & Evaluation

| Model        | RMSE | MSE   | MAE  | R²   |
|--------------|------|-------|------|------|
| Linear Reg.  | 11.81| 139.6 | 7.27 | 0.87 |
| Random Forest| 3.31 | 10.98 | 1.15 | 0.98 |
| SVM          | 9.22 | 85.09 | 4.04 | 0.92 |
| **XGBoost**  | **2.79** | **7.79** | **1.38** | **0.99** |
| LightGBM     | 3.32 | 11.08 | 1.72 | 0.98 |
| LSTM         | 20.92| 437.8 |16.03 | 0.44 |

> 📌 GridSearchCV was used for hyperparameter tuning  
> 🧠 LSTM was designed for time-series forecasting using 85-day windows



