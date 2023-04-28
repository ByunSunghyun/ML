import numpy as np                  
import pandas as pd                   
import matplotlib.pyplot as plt
import seaborn as sns
# 머신러닝 모델들
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

# 경고 제거
import warnings
warnings.filterwarnings(action='ignore')

# 데이터 불러오기
train = pd.read_csv("titanic/train.csv")
test_data = pd.read_csv("titanic/test.csv")
compare_data = pd.read_csv("titanic/gender_submission.csv")

print("train_data columns = ",train.columns.values)
print("test_data columns = ",test_data.columns.values)
print("compare_data columns = ", compare_data.columns.values)

#데이터 변수 확인
# print(train.head()) # 주피터 용
# print(train.columns.values) # vscode 용

# 훈련 데이터와 테스트 데이터의 정보 확인
print('train_data.info')
train.info()
print('test_data.info')
test_data.info()
print('compare_data.info')
compare_data.info()

# 위의 코드를 보면 Null 값이 있는 것을 확인할 수 있다.
# Null 값을 없애기 위해 데이터 전처리
train['Age'].fillna(train['Age'].mean(), inplace=True)  # Age의 평균값으로 결측치 대체
train['Cabin'].fillna('N', inplace=True)  # 'N'으로 대체
train['Embarked'].fillna('N', inplace=True)  # 'N'값으로 대체
test_data['Age'].fillna(train['Age'].mean(), inplace=True)  # Age의 평균값으로 결측치 대체
test_data['Cabin'].fillna('N', inplace=True)  # 'N'으로 대체
test_data['Fare'].fillna(train['Fare'].mean(), inplace=True)  # Age의 평균값으로 결측치 대체
print('데이터 세트 Null값 개수 : ', train.isnull().sum().sum() + test_data.isnull().sum().sum())

# 훈련 자료를 통해 데이터 분석

sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='rainbow')
plt.title("distribution of survivors")
plt.show()

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='rainbow')
plt.title('Distribution of survivors by gender')
plt.show()

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
plt.title('Survivor distribution by room rating')
plt.show()

sns.displot(train['Age'].dropna(),kde=False,color='darkred',bins=30)
plt.title('Age distribution')
plt.show()

sns.countplot(x='SibSp',data=train)
plt.title('Number of siblings and spouses')
plt.show()

train['Fare'].hist(color='blue',bins=40,figsize=(8,4))
plt.title('Fare distribution')
plt.show()

plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
plt.title('Age distribution by room rating')
plt.show()

# 객실등급에 따른 생존률
grid = sns.FacetGrid(train, col='Survived', row='Pclass', hue="Pclass", height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20) 
grid.add_legend();
plt.show()

#승선지에 따른 생존률
grid = sns.FacetGrid(train, row='Embarked', height=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep', order = [1, 2, 3], hue_order = ["male", "female"])
grid.add_legend()
plt.show()


# 불필요한 칼럼 제거
train = train.drop(['Ticket', 'Cabin', 'Name', 'PassengerId'], axis=1)
test_data = test_data.drop(['Ticket', 'Cabin', 'Name', 'PassengerId'], axis=1)

# 칼컴을 레이블 인코딩 하기
def encode_features(dataDF) :
    features = ['Sex', 'Embarked']
    for feature in features :
        le = preprocessing.LabelEncoder()
        le.fit(dataDF[feature])
        dataDF[feature] = le.transform(dataDF[feature])
    
    return dataDF

train = encode_features(train)
train.head()
test_data = encode_features(test_data)
test_data.head()

train.info()
test_data.info()

X = train.drop('Survived', axis=1).select_dtypes(include=['number'])
y = train['Survived']

# 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN 모델 생성
clf = KNN(n_neighbors = 6)

# 모델 학습
clf.fit(X_train, y_train)

# 예측 + f1 score 계산 + 정확도 계산
test_predict_KNN = clf.predict(X_test)
knn_k = f1_score(test_predict_KNN, y_test)
acc = accuracy_score(test_predict_KNN, y_test)
# 결과값 출력
print('KNN')
print('F1 Score:', knn_k )
print('Accuracy:', acc)

# Creating instance of Logistic Regression
log_reg = LogisticRegression(random_state=42)

# 모델 학습
log_reg.fit(X_train, y_train)

# 예측 + f1 score 계산 + 정확도 계산
test_predict_log = log_reg.predict(X_test)
k_log = f1_score(test_predict_log, y_test)
acc = accuracy_score(test_predict_log, y_test)

# 결과값 출력
print('Logistic Regression')
print('F1 Score:', k_log )
print('Accuracy:', acc)

# Creating instance of Random Forest Classifier
r_f = RandomForestClassifier(random_state=42)

# 모델 학습
r_f.fit(X_train, y_train)

# 예측 + f1 score 계산 + 정확도 계산
test_predict_rf = r_f.predict(X_test)
rf_log = f1_score(test_predict_rf, y_test)
acc = accuracy_score(test_predict_rf, y_test)

# 결과값 출력
print('Random Forest')
print('F1 Score:', rf_log )
print('Accuracy:', acc)

# Creating instance of Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# 모델 학습
clf.fit(X_train, y_train)

# 예측 + f1 score 계산 + 정확도 계산
test_predict_dt = clf.predict(X_test)
k_dt = f1_score(test_predict_dt, y_test)
acc = accuracy_score(test_predict_dt, y_test)

# 결과값 출력
print('Decision Tree')
print('F1 Score:', k_dt )
print('Accuracy:', acc)

#K-fold 교차 검증
kf = KFold(n_splits = 5, shuffle = True, random_state = 50)
accuracy_history = []
# K-fold 교차 검증으로 모델 성능을 측정
for train_index, test_index in kf.split(train):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = LogisticRegression(random_state=42) # 모델 선언
    model.fit(X_train, y_train) # 모델 학습

    y_pred = model.predict(X_test) # 예측 라벨
    accuracy_history.append(accuracy_score(y_pred, y_test)) # 정확도 측정 및 기록

print("각 분할의 정확도 기록 :", accuracy_history)
print("평균 정확도 :", np.mean(accuracy_history))

# 최종 모델을 로지스틱 회귀로 정하고 학습시키기 
final_model = LogisticRegression(random_state=42)
final_model.fit(X_train, y_train)

# 예측값과 실제값 비교하기
final_predict = final_model.predict(test_data)
tmp = 0
print("최종 모델")
print("예측 값/ 실제 값 / 정답유무")
for i in range(len(final_predict)) :
    print(final_predict[i], " / " , str(compare_data['Survived'][i]) , " / " , str(final_predict[i] == compare_data['Survived'][i]))
    if final_predict[i] == compare_data['Survived'][i] :
        tmp += 1
        
print("정답률 : " , tmp / len(final_predict) * 100 , "%")
final_f1_score = f1_score(final_predict, compare_data['Survived'])
fianl_acc = accuracy_score(final_predict, compare_data['Survived'])
print('F1 Score:', final_f1_score )
print('Accuracy:', fianl_acc)
