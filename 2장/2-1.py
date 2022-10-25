import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from data.bream_smelt_list import *

fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]
fish_target = ['bream']*35 + ['smelt']*14

input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

index = np.arange(35+14)
np.random.shuffle(index)

train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]

test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

kn = KNeighborsClassifier()
kn.fit(train_input, train_target)

score = kn.score(test_input, test_target)

print(f'score: {score}')

plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(test_input[:,0], test_input[:,1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
