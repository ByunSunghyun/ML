from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
import numpy as np

boston = load_boston()

x = boston.data
y = boston.target

train_size = int(len(x) * 0.7)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

lin_reg = LinearRegression()

lin_reg.fit(x_train, y_train)

y_pred = lin_reg.predict(x_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Linear Regressor RMSE: ", rmse)