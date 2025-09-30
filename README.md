# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
```

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

data = pd.read_csv("Amazon Sale Report.csv", usecols=["Date", "Amount"], low_memory=False)
data["Date"] = pd.to_datetime(data["Date"], infer_datetime_format=True)
data.set_index("Date", inplace=True)

weekly_amount = data["Amount"].resample("W").sum()

adf_result = adfuller(weekly_amount.dropna())
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")
if adf_result[1] < 0.05:
    print("The data is stationary.")
else:
    print("The data is non-stationary.")
print("GIVEN DATA")
print(data.head())
train_size = int(len(weekly_amount) * 0.8)
train, test = weekly_amount[:train_size], weekly_amount[train_size:]

safe_acf_lags = min(10, len(train)-1)
safe_pacf_lags = min(7, len(train)//2)

fig, ax = plt.subplots(2, figsize=(10, 8))
plot_acf(train.dropna(), ax=ax[0], lags=safe_acf_lags)
ax[0].set_title("Autocorrelation Function (ACF)")
ax[0].set_xlabel("Lag")
ax[0].set_ylabel("ACF")

plot_pacf(train.dropna(), ax=ax[1], lags=safe_pacf_lags)
ax[1].set_title("Partial Autocorrelation Function (PACF)")
ax[1].set_xlabel("Lag")
ax[1].set_ylabel("PACF")
plt.tight_layout()
plt.show()

lag_order = min(3, len(train)-1)
ar_model = AutoReg(train.dropna(), lags=lag_order).fit()

ar_pred = ar_model.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label='Test Data')
plt.plot(test.index, ar_pred, label='AR Model Prediction', color='red', linestyle='--')
plt.xlabel("Date")
plt.ylabel("Amount")
plt.title("AR Model Prediction vs Test Data")
plt.legend()
plt.grid()
plt.show()

mse = mean_squared_error(test, ar_pred)
print(f"Mean Squared Error (MSE): {mse}")

plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Train Data')
plt.plot(test.index, test, label='Test Data')
plt.plot(test.index, ar_pred, label='AR Model Prediction', color='red', linestyle='--')
plt.xlabel("Date")
plt.ylabel("Amount")
plt.title("Train, Test, and AR Model Prediction")
plt.legend()
plt.grid()
plt.show()

```
### OUTPUT:

GIVEN DATA & ADF test result: 

![alt text](<Screenshot 2025-09-30 141941.png>)

ACF

![alt text](<Screenshot 2025-09-30 141952.png>)

PACF 

![alt text](<Screenshot 2025-09-30 142002.png>)

PREDICTION

![alt text](<Screenshot 2025-09-30 142027.png>)

FINIAL PREDICTION

![alt text](<Screenshot 2025-09-30 142040.png>)


### RESULT:
Thus we have successfully implemented the auto regression function using python.
