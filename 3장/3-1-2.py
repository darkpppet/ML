import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from data.perch_length_weight import *

perch_length = perch_length.reshape(-1, 1)

knr = KNeighborsRegressor()
x = np.arange(5, 45).reshape(-1, 1)

plt.scatter(perch_length, perch_weight)

for i in range(1, 57, 5):
    knr.n_neighbors = i
    knr.fit(perch_length, perch_weight)

    prediction = knr.predict(x)

    plt.plot(x, prediction, label=f'n_neighbors: {i}')
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.legend()

plt.show()
