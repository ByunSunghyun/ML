from sklearn.datasets import make_moons
from matplotlib import pyplot as plt
import seaborn as sns

# Generate a dataset
X, y = make_moons(n_samples=500, noise=0.1)

# Plot the data as a scatter plot
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y)
plt.show()

# Plot a histogram of the first feature of the data
sns.histplot(x=X[:, 0])
plt.show()

# Plot a histogram of the squared values of the first feature of the data
sns.histplot(x=X[:, 0] ** 2)
plt.show()