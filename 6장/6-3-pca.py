import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from custom_functions.draw_fruits import draw_fruits

fruits = np.load('../data/fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)

pca = PCA(n_components=50)
pca.fit(fruits_2d)

draw_fruits(pca.components_.reshape(-1, 100, 100))

fruits_pca = pca.transform(fruits_2d)

fruits_inverse = pca.inverse_transform(fruits_pca)

fruits_reconstruct = fruits_inverse.reshape(-1, 100, 100)

for start in [0, 100, 200]:
    draw_fruits(fruits_reconstruct[start:start+100])

print(np.sum(pca.explained_variance_ratio_))
print(pca.explained_variance_ratio_)

plt.plot(pca.explained_variance_ratio_)
plt.show()
