import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Load the data
train_data = pd.read_csv('titanic/train.csv')
test_data = pd.read_csv('titanic/test.csv')
# i want make a model that predicts whether a passenger survived the Titanic shipwreck or not.


# Prepare the data
X = train_data.drop('Survived', axis=1).select_dtypes(include=['number'])
y = train_data['Survived']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# # Create and fit the model
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X_train, y_train)

# # Evaluate the model
# print(knn.score(X_val, y_val))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

print(knn.score(X_val_scaled, y_val))