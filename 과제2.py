import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
import warnings

color = sns.color_palette()

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

warnings.filterwarnings(action = 'ignore')

# Read the data
df_train = pd.read_csv('titanic/train.csv')
df_test = pd.read_csv('titanic/test.csv')
df_submission  = pd.read_csv('titanic/gender_submission.csv')

# Shape of the data
print('Shape of the train data: %s', df_train.shape)
print('Shape of the test data: %s', df_test.shape)
print('Shape of the submission data: %s', df_submission.shape)

# Sample train data
print(df_train.head())

# Smaple of test data
print(df_test.head())

# Sample submission data
print(df_submission.head())

# Data types
print(df_train.dtypes)

# Summary statistics
print(df_train.describe())

# Let's create a copy of the train and test data to perform data cleaning
df_train_copy = df_train.copy()
df_test_copy = df_test.copy()

# Missing values in train data
df_train_copy.isna().sum()

# Missing values in test data
df_test_copy.isna().sum()

# Impute missing values in Age column with median
df_train_copy['Age'] = df_train_copy['Age'].fillna(df_train_copy['Age'].median())
df_test_copy['Age'] = df_test_copy['Age'].fillna(df_test_copy['Age'].median())

# Impute missing values in Embarked column with mode
df_train_copy['Embarked'] = df_train_copy['Embarked'].fillna(df_train_copy['Embarked'].mode()[0])
df_test_copy['Embarked'] = df_test_copy['Embarked'].fillna(df_test_copy['Embarked'].mode()[0])

# Check if there is any relations between the missing values in Cabin column and Survived column
df_train_copy[df_train_copy['Cabin'].isna()]['Survived'].value_counts()

# Impute missing values in Cabin column with 'Missing'
df_train_copy['Cabin'] = df_train_copy['Cabin'].fillna('Missing')
df_test_copy['Cabin'] = df_test_copy['Cabin'].fillna('Missing')

# Let see if we still have any missing values in the train data
df_train_copy.isna().sum()

# Imputing missing values in Fare column with median in test data
df_test_copy['Fare'] = df_test_copy['Fare'].fillna(df_test_copy['Fare'].median())

# Let see if we still have any missing values in test data
df_test_copy.isna().sum()

# Check for duplicates in train data
df_train_copy.duplicated().sum()

# Check for duplicates in test data
df_test_copy.duplicated().sum()

# Convert Name, Survived, Pclass, Sex, SibSp, Parch, Embarked, Ticket, Cabin to categorical variables in train data
df_train_copy['Survived'] = df_train_copy['Survived'].astype('category')
df_train_copy['Pclass'] = df_train_copy['Pclass'].astype('category')
df_train_copy['Sex'] = df_train_copy['Sex'].astype('category')
df_train_copy['SibSp'] = df_train_copy['SibSp'].astype('category')
df_train_copy['Parch'] = df_train_copy['Parch'].astype('category')
df_train_copy['Embarked'] = df_train_copy['Embarked'].astype('category')
df_train_copy['Ticket'] = df_train_copy['Ticket'].astype('category')
df_train_copy['Cabin'] = df_train_copy['Cabin'].astype('category')
df_train_copy['Name'] = df_train_copy['Name'].astype('category')

# Convert Name, Pclass, Sex, SibSp, Parch, Embarked, Ticket, Cabin to categorical variables in test data
df_test_copy['Pclass'] = df_test_copy['Pclass'].astype('category')
df_test_copy['Sex'] = df_test_copy['Sex'].astype('category')
df_test_copy['SibSp'] = df_test_copy['SibSp'].astype('category')
df_test_copy['Parch'] = df_test_copy['Parch'].astype('category')
df_test_copy['Embarked'] = df_test_copy['Embarked'].astype('category')
df_test_copy['Ticket'] = df_test_copy['Ticket'].astype('category')
df_test_copy['Cabin'] = df_test_copy['Cabin'].astype('category')
df_test_copy['Name'] = df_test_copy['Name'].astype('category')

# Function for plolting the distribution of categorical variables
def plot_cat(df, col, x_label, y_label, plot_title):
    sns.set_style('darkgrid')
    plt.figure(figsize=(12,8))
    sns.countplot(data=df, x=col, color='dodgerblue')
    plt.title(plot_title, fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.show()

# Plot distribution of Survived column
plot_cat(df_train_copy, 'Survived', 'Survived', 'Count', 'Distribution of Survived column')

# Plotting distribution of Pclass column
plot_cat(df_train_copy, 'Pclass', 'Pclass', 'Count', 'Distribution of Pclass column')

# Plotting distribution of Sex column
plot_cat(df_train_copy, 'Sex', 'Sex', 'Count', 'Distribution of Sex column')

# Plottting distribution of SibSp column
plot_cat(df_train_copy, 'SibSp', 'SibSp', 'Count', 'Distribution of SibSp column')

# Plottting distribution of Parch column
plot_cat(df_train_copy, 'Parch', 'Parch', 'Count', 'Distribution of Parch column')

# Plottting distribution of Embarked column
plot_cat(df_train_copy, 'Embarked', 'Embarked', 'Count', 'Distribution of Embarked column')
# Function for calculating descriptives of numeric variable and plotting the distribution
def plot_dist(df, col, x_label, y_label, plot_title):
    _min = df[col].min()
    _max = df[col].max()
    ran = df[col].max()-df[col].min()
    mean = df[col].mean()
    median = df[col].median()
    st_dev = df[col].std()
    skew = df[col].skew()
    kurt = df[col].kurtosis()

    # calculating points of standard deviation
    points = mean-st_dev, mean+st_dev
    sns.set_style('darkgrid')
    plt.figure(figsize=(12,8))
    sns.histplot(data=df, x=col, bins=30, kde=True, color='dodgerblue')
    sns.lineplot(x=points, y=[0,0], color = 'black', label = "std_dev")
    sns.scatterplot(x=[_min,_max], y=[0,0], color = 'orange', label = "min/max")
    sns.scatterplot(x=[mean], y=[0], color = 'red', label = "mean")
    sns.scatterplot(x=[median], y=[0], color = 'blue', label = "median")
    plt.title(plot_title, fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)

    # Creating a DataFrame for the descriptive statistics
    variable_stats = pd.DataFrame({'Statistics': ['Minimum Value', 'Maximum Value', 'Range', 'Mean', 
                                                    'Median', 'Standard Deviation', 'Skewness', 'Kurtosis'], 
                                        'Value': [_min, _max, ran, mean, median, st_dev, skew, kurt]})
    
    plt.show()

    print(tabulate(variable_stats, headers='keys', showindex=False, tablefmt='html'))
    
# Plotting distribution of Age column
plot_dist(df_train, 'Age', 'Age', 'Count', 'Distribution of Age column')

# Let's see how Fare column is distributed
plot_dist(df_train, 'Fare', 'Fare', 'Count', 'Distribution of Fare column')

# Function for plotting the distribution of numeric variables against the target variable
# Here target variable is assumed to be categorical
def plot_num_vs_target(df, col, target, x_label, y_label, plot_title):
    sns.set_style('darkgrid')
    plt.figure(figsize=(12,8))
    sns.boxplot(data=df, x=target, y=col, color='dodgerblue')
    plt.title(plot_title, fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.show()
    
# Relationship between Survived and Age
plot_num_vs_target(df_train_copy, 'Age', 'Survived', 'Survived', 'Age', 'Relationship between Survived and Age')

# Relationship between Survived and Fare
plot_num_vs_target(df_train_copy, 'Fare', 'Survived', 'Survived', 'Fare', 'Relationship between Survived and Fare')

# Function for plotting the distribution of categorical variables against the target variable
# Here target variable and categorical variable are assumed to be categorical
def plot_cat_vs_target(df, col, target, x_label, y_label, plot_title):
    sns.set_style('darkgrid')
    plt.figure(figsize=(12,8))
    sns.countplot(data=df, x=col, hue=target, palette='Set1')
    plt.title(plot_title, fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.show()
    
# Relationship between Survived and Pclass
plot_cat_vs_target(df_train_copy, 'Pclass', 'Survived', 'Pclass', 'Count', 'Relationship between Survived and Pclass')

# Relationship between Survived and Sex
plot_cat_vs_target(df_train_copy, 'Sex', 'Survived', 'Sex', 'Count', 'Relationship between Survived and Sex')

# Relationship between Survived and SibSp
plot_cat_vs_target(df_train_copy, 'SibSp', 'Survived', 'SibSp', 'Count', 'Relationship between Survived and SibSp')

# Relationship between Survived and Parch
plot_cat_vs_target(df_train_copy, 'Parch', 'Survived', 'Parch', 'Count', 'Relationship between Survived and Parch')

# Relationship between Survived and Embarked
plot_cat_vs_target(df_train_copy, 'Embarked', 'Survived', 'Embarked', 'Count', 'Relationship between Survived and Embarked')

# Function to encode categorical variables, we will use scikit-learn's LabelEncoder for label encoding and pandas get_dummies for one-hot encoding
from sklearn.preprocessing import LabelEncoder

def encode_cat(df, col, encoding_type):
    if encoding_type == 'label':
        label_encoder = LabelEncoder()
        df[col] = label_encoder.fit_transform(df[col])
    elif encoding_type == 'onehot':
        df = pd.get_dummies(df, columns=[col], prefix=[col])
    return df

# Encoding variables in the training dataset and create a new dataframe called df_train_encoded
df_train_encoded = df_train_copy.copy()
df_train_encoded = encode_cat(df_train_encoded, 'Survived', 'label')
df_train_encoded = encode_cat(df_train_encoded, 'Cabin', 'label')
df_train_encoded = encode_cat(df_train_encoded, 'Pclass', 'label')
df_train_encoded = encode_cat(df_train_encoded, 'Sex', 'label')
df_train_encoded = encode_cat(df_train_encoded, 'SibSp', 'label')
df_train_encoded = encode_cat(df_train_encoded, 'Parch', 'label')
df_train_encoded = encode_cat(df_train_encoded, 'Embarked', 'label')

# Encoding variables in the test dataset and create a new dataframe called df_test_encoded
df_test_encoded = df_test_copy.copy()
df_test_encoded = encode_cat(df_test_encoded, 'Cabin', 'label')
df_test_encoded = encode_cat(df_test_encoded, 'Pclass', 'label')
df_test_encoded = encode_cat(df_test_encoded, 'Sex', 'label')
df_test_encoded = encode_cat(df_test_encoded, 'SibSp', 'label')
df_test_encoded = encode_cat(df_test_encoded, 'Parch', 'label')
df_test_encoded = encode_cat(df_test_encoded, 'Embarked', 'label')

# Check the processed training set
print(df_train_encoded.head())

# Check the processed test set
print(df_test_encoded.head())

# Function to plot correlation between variables
def plot_corr(df, size=10):
    corr = df.corr()
#     print(corr)
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(corr, annot=True, linewidths=.5, ax=ax, cmap='crest')
    plt.show()
    
# Correlation between variables in the training set
plot_corr(df_train_encoded.drop(['PassengerId', 'Name', 'Ticket'], axis=1))


# Function to plot correlation of variables with the target variable as a barplot
def plot_corr_target(df, target, size=10):
    corr = df.corr()
    corr_target = corr[target]
    corr_target = corr_target.sort_values(ascending=False)
    corr_target = corr_target.drop(target)
    plt.figure(figsize=(size, size))
    corr_target.plot.barh()
    plt.show()
    
# Check correlation of variables with the target variable
plot_corr_target(df_train_encoded.drop(['Name', 'Ticket', 'PassengerId'], axis=1), 'Survived')

# We will first separate the target variable from the features
y = df_train_encoded['Survived']
x = df_train_encoded.drop(['Survived', 'Name', 'Ticket', 'PassengerId'], axis=1)
x.shape, y.shape

## Importing the MinMax Scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

x = pd.DataFrame(x_scaled, columns = x.columns)

# Check data after scaling
x.head()

# Importing the train test split function
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y, random_state = 56, stratify=y)

# Import KNN classifier and metric F1 score
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import f1_score

# Creating instance of KNN
clf = KNN(n_neighbors = 10)

# Fitting the model
clf.fit(train_x, train_y)

# Predicting over the Train Set and calculating F1
test_predict = clf.predict(test_x)
k = f1_score(test_predict, test_y)
print('Test F1 Score    ', k )


# Function to find the best value of K
def Elbow(K):
    #initiating empty list
    test_error = []
   
    #training model for evey value of K
    for i in K:
        #Instance oh KNN
        clf = KNN(n_neighbors = i)
        clf.fit(train_x, train_y)
        # Appending F1 scores to empty list claculated using the predictions
        tmp = clf.predict(test_x)
        tmp = f1_score(tmp,test_y)
        error = 1-tmp
        test_error.append(error)
    
    return test_error

#Defining K range
k = range(6, 20, 2)

# calling above defined function
test = Elbow(k)

# plotting the Curves
plt.plot(k, test)
plt.xlabel('K Neighbors')
plt.ylabel('Test error')
plt.title('Elbow Curve for test')
plt.show()

# Creating instance of KNN
clf = KNN(n_neighbors = 6)

# Fitting the model
clf.fit(train_x, train_y)

# Predicting over the Test Set and calculating F1
test_predict = clf.predict(test_x)
k = f1_score(test_predict, test_y)
print('KNN Test F1 Score    ', k )

submission_predictions = clf.predict(df_test_encoded.drop(['Name', 'Ticket', 'PassengerId'], axis=1))

print(submission_predictions)

df_submission['Survived'] = submission_predictions

df_submission.head()

df_submission.to_csv('submission_KNN.csv', index=False)

# Importing Logistic Regression
from sklearn.linear_model import LogisticRegression

# Creating instance of Logistic Regression
log_reg = LogisticRegression()

# Fitting the model
log_reg.fit(train_x, train_y)

# Predicting over the Test Set and calculating F1
test_predict_log = log_reg.predict(test_x)
k_log = f1_score(test_predict_log, test_y)

print('Logistic Regression Test F1 Score    ', k_log )

submission_predictions_log = log_reg.predict(df_test_encoded.drop(['Name', 'Ticket', 'PassengerId'], axis=1))

print(submission_predictions_log)

# Combine predics with df_submission and save to csv
df_submission['Survived'] = submission_predictions_log
df_submission.to_csv('submission_log.csv', index=False)

# Importing Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

# Creating instance of Decision Tree Classifier
clf = DecisionTreeClassifier()

# Fitting the model
clf.fit(train_x, train_y)

# Predicting over the Test Set and calculating F1
test_predict_dt = clf.predict(test_x)
k_dt = f1_score(test_predict_dt, test_y)

print('Decision Tree Test F1 Score    ', k_dt )

submission_predictions_dt = clf.predict(df_test_encoded.drop(['Name', 'Ticket', 'PassengerId'], axis=1))

# Combine predictions with df_submission and save to csv
df_submission['Survived'] = submission_predictions_dt
df_submission.to_csv('submission_dt.csv', index=False)

#https://www.kaggle.com/code/rishabhvijay/surviving-the-titanic

# part 1
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# #데이터 불러오기
# titanic_df = pd.read_csv('titanic/train.csv')
# titanic_df.head(3)

# #컬럼 속성 확인하기
# print('\n ### 학습 데이터 정보 ###\n')
# print(titanic_df.info())

# # 결측치 없애기
# titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace = True) # Age의 평균값으로 결측치 대체
# titanic_df['Cabin'].fillna('N', inplace = True) # 'N'으로 대체
# titanic_df['Embarked'].fillna('N', inplace = True) # 'N'값으로 대체
# print('데이터 세트 Null값 개수 : ', titanic_df.isnull().sum().sum())

# print('\n ### 학습 데이터 정보 ###\n')
# print(titanic_df.info())

# # 값 분류 확인하기
# print('Sex 값 분포 : \n', titanic_df['Sex'].value_counts())
# print('\n')
# print('Cabin 값 분포 : \n', titanic_df['Cabin'].value_counts())
# print('\n')
# print('Embarked 값 분포 : \n', titanic_df['Embarked'].value_counts())

# titanic_df['Cabin'] = titanic_df['Cabin'].str[:1] # str은 글자를 추출하기위한 메소드
# print(titanic_df['Cabin'].head(3))

# # 성별에 따른 생존자 수 확인하기
# print(titanic_df.groupby(['Sex', 'Survived'])['Survived'].count())

# sns.barplot(x = 'Sex', y = 'Survived', data = titanic_df) 
# sns.barplot(x = 'Pclass', y = 'Survived', hue = 'Sex', data = titanic_df) # hue는 x를 더욱 세부적으로 나누기 위한 특성


# # 카테고리값 할당을 위한 함수
# def get_category(age) :
#     cat = ''
#     if age <= -1 : cat = 'Unknown'
#     elif age <= 5 : cat = 'Baby'
#     elif age <= 12 : cat = 'Child'
#     elif age <= 18 : cat = 'Teenager'
#     elif age <= 25 : cat = 'Student'
#     elif age <= 35 : cat = 'Young Child'
#     elif age <= 60 : cat = 'Adult'
#     else : cat = 'Elderly'
    
#     return cat


# # 막대 그래프의 크기 figure를 더 크게 설정
# plt.figure(figsize = (10, 6))

# # X축의 값을 순차적으로 표시하기 위한 설정
# group_names = ['UnKnown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Elderly']

# # lambda 식에 위에서 생성한 get_category()함수를 반환값으로 지정
# # get_category(X)는 입력값으로 'Age'칼럼 값을 받아서 해당하는 cat 반환
# titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x : get_category(x))
# sns.barplot(x = 'Age_cat', y = 'Survived', hue = 'Sex', data = titanic_df, order = group_names)
# titanic_df.drop('Age_cat', axis = 1, inplace = True)


# from sklearn import preprocessing

# # 여러 칼컴을 레이블 인코딩 하기
# def encode_features(dataDF) :
#     features = ['Cabin', 'Sex', 'Embarked']
#     for feature in features :
#         le = preprocessing.LabelEncoder()
#         le.fit(dataDF[feature])
#         dataDF[feature] = le.transform(dataDF[feature])
    
#     return dataDF

# titanic_df = encode_features(titanic_df)
# print(titanic_df.head())

# plt.show()

# # Null 처리 함수
# def fillna(df) : 
#     df['Age'].fillna(df['Age'].mean(), inplace = True)
#     df['Cabin'].fillna('N', inplace = True)
#     df['Embarked'].fillna('N', inplace = True)
#     df['Fare'].fillna(0, inplace = True)
#     return df

# # 머신러닝 알고리즘에 불필요한 속성 제거
# def drop_features(df) :
#     df.drop(['PassengerId', 'Name', 'Ticket'], axis = 1, inplace = True) # 모두 단순 구분을 위한 컬럼들
#     return df

# # 레이블 인코딩 수행
# def format_features(df) :
#     df['Cabin'] = df['Cabin'].str[:1]
#     features = ['Cabin', 'Sex', 'Embarked']
#     for feature in features :
#         le = preprocessing.LabelEncoder()
#         le.fit(df[feature])
#         df[feature] = le.transform(df[feature])
    
#     return df

# # 앞 3가지 함수를 모두 통함
# def transform_features(df) :
#     df = fillna(df)
#     df = drop_features(df)
#     df = format_features(df)
#     return df

# # 데이터를 재로딩하고, 피처 데이터 세트와 레이블 데이터 세트 추출
# titanic_df = pd.read_csv("titanic/train.csv")
# y_titanic_df = titanic_df['Survived']
# X_titanic_df = titanic_df.drop('Survived', axis = 1, inplace = False)

# X_titanic_df = transform_features(X_titanic_df)

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, test_size = 0.2, random_state = 11)

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

# # 각 모델에 대한 Classifier 클래스 생성
# dt_clf = DecisionTreeClassifier(random_state = 11)
# rf_clf = RandomForestClassifier(random_state = 11)
# lr_clf = LogisticRegression(random_state = 11)

# # DecisionTreeClassfier 학습/예측평가
# dt_clf.fit(X_train, y_train)
# dt_pred = dt_clf.predict(X_test)
# print('DecisionTreeClassifier 정확도 : {0:.4f}'.format(accuracy_score(y_test, dt_pred)))

# # RandomForestClassifier 학습/예측/평가
# rf_clf.fit(X_train, y_train)
# rf_pred = rf_clf.predict(X_test)
# print('RandomForestClassifier 정확도 : {0:.4f}'.format(accuracy_score(y_test, rf_pred)))

# # LogisticRegression 학습/예측/평가
# lr_clf.fit(X_train, y_train)
# lr_pred = lr_clf.predict(X_test)
# print('LogisticRegression 정확도 : {0:.4f}'.format(accuracy_score(y_test, lr_pred)))

# from sklearn.model_selection import KFold

# def exec_kfold(clf, folds = 5) :
#     # 폴드 세트가 5개인 KFold 객체 생성. 폴드 수만큼 예측결과 저장 위한 리스트 생성
#     kfold = KFold(n_splits = folds)
#     scores = []
    
#     # KFold 교차 검증 수행
#     for iter_count, (train_index, test_index) in enumerate(kfold.split(X_titanic_df)) :
#         # X_titanic_df 데이터에서 교차 검증별로 학습과 검증 데이터를 가리키는 index 생성
#         X_train, X_test = X_titanic_df.values[train_index], X_titanic_df.values[test_index] # values를 통해 df를 ndarray로 변환
#         y_train, y_test = y_titanic_df.values[train_index], y_titanic_df.values[test_index]
        
#         # Classifier 학습/예측/평가
#         clf.fit(X_train, y_train)
#         clf_pred = clf.predict(X_test)
#         accuracy = accuracy_score(y_test, clf_pred)
#         scores.append(accuracy)
#         print("교차 검증 {0} 정확도 : {1:.4f}".format(iter_count, accuracy))
        
#     # 5개의 fold에서 평균 계산
#     mean_score = np.mean(scores)
#     print("평균 정확도: {0:.4f}".format(mean_score))

# #exec_fold 호출
# exec_kfold(dt_clf, folds = 5)

# from sklearn.model_selection import GridSearchCV

# parameters = {'max_depth':[2, 3, 5, 10],
#              'min_samples_split':[2, 3, 5],
#              'min_samples_leaf':[1, 5, 8]}
# grid_dclf = GridSearchCV(dt_clf, param_grid = parameters, scoring = 'accuracy', cv = 5)
# grid_dclf.fit(X_train, y_train)

# print('GridSearchCV 최적 하이퍼 파라미터 : ', grid_dclf.best_params_)
# print('GridSearchCV 최고 정확도: {0:.4f}'.format(grid_dclf.best_score_))
# best_dclf = grid_dclf.best_estimator_

# #GridSearchCV의 최적 하이퍼 파라미터로 학습된 Estimator로 예측 및 평가 수행
# dpredictions = best_dclf.predict(X_test)
# accuracy = accuracy_score(y_test, dpredictions)
# print('테스트 세트에서의 DecisionTreeClassifier 정확도 : {0:.4f}'.format(accuracy))

# # https://aiclaudev.tistory.com/m/11

# train_data = pd.read_csv('titanic/train.csv')
# test_data = pd.read_csv('titanic/test.csv')
# # i want make a model that predicts whether a passenger survived the Titanic shipwreck or not.


# # Prepare the data
# X = train_data.drop('Survived', axis=1).select_dtypes(include=['number'])
# y = train_data['Survived']

# # Split the data into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# # # Create and fit the model
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X_train, y_train)

# # # Evaluate the model
# # print(knn.score(X_val, y_val))

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_val_scaled = scaler.transform(X_val)

# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X_train_scaled, y_train)

# print(knn.score(X_val_scaled, y_val))



# part2
# import numpy as np                    # linear algebra
# import pandas as pd                   # data processing, CSV file I/O (e.g. pd.read_csv)
# import matplotlib.pyplot as plt
# import seaborn as sns

# train = pd.read_csv("titanic/train.csv")
# test_data = pd.read_csv("titanic/test.csv") 

# print(train.head())

# sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# plt.show()

# sns.set_style('whitegrid')
# sns.countplot(x='Survived',data=train,palette='RdBu_r')
# plt.show()

# sns.set_style('whitegrid')
# sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
# plt.show()

# sns.set_style('whitegrid')
# sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
# plt.show()


# sns.displot(train['Age'].dropna(),kde=False,color='darkred',bins=30)
# plt.show()

# train['Age'].hist(bins=30,color='darkred',alpha=0.7)
# plt.show()

# sns.countplot(x='SibSp',data=train)
# plt.show()


# train['Fare'].hist(color='green',bins=40,figsize=(8,4))
# plt.show()

# plt.figure(figsize=(12, 7))
# sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
# plt.show()

# def impute_age(cols):
#     Age = cols[0]
#     Pclass = cols[1]    
#     if pd.isnull(Age):
#         if Pclass == 1:
#             return 37
#         elif Pclass == 2:
#             return 29
#         else:
#             return 24
#     else:
#         return Age
    
# train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
# test_data['Age'] = test_data[['Age','Pclass']].apply(impute_age,axis=1)

# sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# plt.show()

# train.drop('Cabin',axis=1,inplace=True)
# test_data.drop('Cabin',axis=1,inplace=True)

# #for train data
# sex = pd.get_dummies(train['Sex'],drop_first=True)
# embark = pd.get_dummies(train['Embarked'],drop_first=True)
# train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
# train = pd.concat([train,sex,embark],axis=1)

# #for test data
# test_sex = pd.get_dummies(test_data['Sex'],drop_first=True)
# test_embark = pd.get_dummies(test_data['Embarked'],drop_first=True)
# test_data.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
# test_data = pd.concat([test_data,test_sex,test_embark],axis=1)

# #fill null value of fare column with 0
# test_data.Fare.fillna(0 ,inplace = True)

# #Train Test Split
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived','PassengerId'],axis=1), 
#                                                     train['Survived'], test_size=0.30, 
#                                                     random_state=101)

# from sklearn.linear_model import LogisticRegression
# logmodel = LogisticRegression()
# logmodel.fit(X_train,y_train)

# predictions = logmodel.predict(X_test)

# from sklearn.metrics import classification_report
# print(classification_report(y_test,predictions))

# id = test_data['PassengerId']
# predictions = logmodel.predict(test_data.drop('PassengerId', axis=1))
# result = pd.DataFrame({ 'PassengerId' : id, 'Survived': predictions })
# result.head()


# #writing the output in csv 
# result.to_csv('titanic/titanic-predictions.csv', index = False)