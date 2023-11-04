import numpy as np
from sklearn.cluster import KMeans
from custom_functions.draw_fruits import draw_fruits

fruits = np.load('../data/fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)

km = KMeans(n_clusters=3, n_init='auto')
km.fit(fruits_2d)

# draw_fruits(fruits[km.labels_ == 0])
# draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio=3)

print(km.transform(fruits_2d[100:101]))
print(km.predict(fruits_2d[100:101]))

draw_fruits(fruits[100:101])

print(km.n_iter_)
