import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from data.bream_list import *
from data.smelt_list import *

length_data = bream_length + smelt_length
weight_data = bream_weight + smelt_weight

fish_data = [[l, w] for l, w in zip(length_data, weight_data)]
fish_target = ['bream']*len(bream_length) + ['smelt']*len(smelt_length)

kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target)

score = kn.score(fish_data, fish_target)
predict = kn.predict([[30, 600]])

for n in range(5, 50):
    kn.n_neighbors = n
    n_score = kn.score(fish_data, fish_target)

    print(f'{n}th score: {n_score}')

plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.scatter(30, 600, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
