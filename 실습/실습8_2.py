from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
import numpy as np

boston = load_boston()
x,y = boston.data, boston.target

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=42)

model = MLPRegressor(hidden_layer_sizes=(100,50), max_iter=1000, random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("MLP Regressor RMSE: ", rmse)