from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#데이터 셋 생성 
X, y=make_moons(n_samples=1000, noise=0.1, random_state=42)

#학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.4, random_state=42)

#선형 SVM 모델 생성 및 학습
svm_linear = SVC(kernel='linear', C=1, gamma='auto')
svm_linear.fit(X_train, y_train)

#비선형 SVM 모델 생성 및 학습
svm_rbf = SVC(kernel='rbf', C=1, gamma=0.1)
svm_rbf.fit(X_train, y_train)

#테스트 데이터 예측 및 정확도 평가
y_pred_linear = svm_linear.predict(X_test)
accuracy_linear = accuracy_score(y_test, y_pred_linear)

y_pred_rbf = svm_rbf.predict(X_test)
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)

print('Linear SVM accuracy: %2f'%accuracy_linear)
print('RBF SVM accuracy: %2f'%accuracy_rbf)