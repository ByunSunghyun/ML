from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# Lets plot our decision regions to visualize how well the classification worked
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings


def versiontuple(v):
    return tuple(map(int, (v.split("."))))


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='000000',
                    alpha=1.0,
                    edgecolor='black',
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')

## Load Iris Dataset and use two features: Sepal Length and Petal Length

iris = pd.read_csv("iris.csv")
X = iris.iloc[:, [1,3]].values
X
Y = iris.iloc[:, 5].values
Y
class_mapping = {label:idx+1 for idx,label in 
                 enumerate(np.unique(Y))}
iris['Species'] = iris['Species'].map(class_mapping)
Y = iris.iloc[:,5].values
Y

#Split into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=0)

#Running Logistic Regression with unstandardized components

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lr = LogisticRegression(C=1, random_state=0) 
lr.fit(X_train, Y_train)
#Checking Accuracy

Y_pred = lr.predict(X_test)
# Lets see how good we did
print('Misclassified samples: %d' %(Y_test != Y_pred).sum())
from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(Y_test,Y_pred))
X_combined_std = np.vstack((X_train, X_test))
Y_combined = np.hstack((Y_train, Y_test))
Y_combined
plot_decision_regions(X_combined_std, Y_combined, classifier=lr, test_idx=range(105, 150))
plt.title('Logistic Regression')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('logistic_regression.png', dpi=300)
plt.show()
# Running Logistic Regression with Standardied Components

## Use StandardScalar to scale the features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

lr = LogisticRegression(C=1, random_state=0) 
lr.fit(X_train_std, Y_train)

Y_pred = lr.predict(X_test_std)
# Lets see how good we did
print('Misclassified samples: %d' %(Y_test != Y_pred).sum())
from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(Y_test,Y_pred))
X_combined_std = np.vstack((X_train_std, X_test_std))
Y_combined = np.hstack((Y_train, Y_test))
Y_combined
plot_decision_regions(X_combined_std, Y_combined,
                      classifier=lr, test_idx=range(105, 150))
plt.title('Logistic Regression on Standardized features')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('logistic_regression.png', dpi=300)
plt.show()
# Plotting explained variance ratio for PCA with 4 Components

## Load Iris Dataset and use all four features

iris = pd.read_csv("iris.csv")
X = iris.iloc[:, [1,2,3,4]].values
X
Y = iris.iloc[:, 5].values
Y
class_mapping = {label:idx+1 for idx,label in 
                 enumerate(np.unique(Y))}
iris['Species'] = iris['Species'].map(class_mapping)
Y = iris.iloc[:,5].values

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
from sklearn.decomposition import PCA

pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_
plt.bar(range(1, 5), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(range(1, 5), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.show()
# Running Logistic Regression on 2 Principal Components
pca = PCA(n_components=2)
X_train_pca_scikit = pca.fit_transform(X_train_std)
X_test_pca_scikit = pca.transform(X_test_std)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(Y_train), colors, markers):
    plt.scatter(X_train_pca_scikit[Y_train == l, 0], 
                X_train_pca_scikit[Y_train == l, 1], 
                c=c, label=l, marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
#plt.savefig('pca-scikit.png', dpi=300)
plt.show()
## Implement PCA and use all components 
## giving the explained variance of each Principal Component
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lr = LogisticRegression(C=1, random_state=1) # we will see the parameter C below
lr.fit(X_train_pca_scikit, Y_train)
X_combined_std = np.vstack((X_train_pca_scikit, X_test_pca_scikit))
Y_combined = np.hstack((Y_train, Y_test))
Y_combined
plot_decision_regions(X_combined_std, Y_combined,
                      classifier=lr, test_idx=range(105, 150))
plt.title('Logistic Regression on Principal Components, 2 Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('logistic_regression.png', dpi=300)
plt.show()
# Use the LR model to predict the test data
Y_pred = lr.predict(X_test_pca_scikit)
# Lets see how good we did
print('Misclassified samples: %d' %(Y_test != Y_pred).sum())
from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(Y_test,Y_pred))