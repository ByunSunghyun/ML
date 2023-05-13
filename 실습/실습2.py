import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#데이터 셋 로드
iris=load_iris()
X, y = iris.data[:, 1:3], iris.target
#train-test split ratio
ratios = np.linspace(0.1,0.9,num=9)
#모델별로 정확도를 저장할 리스트
nb_accs, rf_accs = [],[]

#작성 내용
for ratio in ratios:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=42)
    
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    
    nb_pred = nb_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)
    
    nb_accuracy = accuracy_score(y_test, nb_pred)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    nb_accs.append(nb_accuracy)
    rf_accs.append(rf_accuracy)

plt.plot(ratios, nb_accs, label="Navie Bayes")
plt.plot(ratios, rf_accs, label="Random Forest")
plt.title("Accuracy by test set ratio")
plt.xlabel("Test set ratio")
plt.ylabel("Accuracy")
plt.legend()
plt.show()