from sklearn.datasets import make_moons
from matplotlib import pyplot as plt
import seaborn as sns
#데이터 셋 생성
x, y = make_moons(n_samples=500, noise=0.3, random_state=42)
#산점도 그리기
sns.scatterplot(x=x[:,0],y=x[:,1], hue=y, palette='rainbow')
plt.title('Scatter plot of origin Moons dataset')
plt.show()

#원본 데이터의 첫번째 특성에 대한 히스토그램
plt.hist(x[:,0], bins=25)
plt.title('Histogram of feature 1 of origin Moons dataset')
plt.show()

#첫 번째 특성을 제곱한 값에 대한 히스토그램
plt.hist(x[:,0]**2, bins=25)
plt.title('Histogram of feature 1 squared of Moons dataset')
plt.show()
