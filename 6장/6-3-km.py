import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from custom_functions.draw_fruits import draw_fruits

fruits = np.load('../data/fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)

pca = PCA(n_components=0.5)
pca.fit(fruits_2d)
fruits_pca = pca.transform(fruits_2d)

print(pca.n_components_)

km = KMeans(n_clusters=3, n_init='auto')
km.fit(fruits_pca)

for label in range(0, 3):
    draw_fruits(fruits[km.labels_ == label])

for label in range(0, 3):
    data = fruits_pca[km.labels_ == label]
    plt.scatter(data[:, 0], data[:, 1])
plt.show()
