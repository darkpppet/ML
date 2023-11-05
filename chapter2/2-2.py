import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from data.bream_smelt_list import *

def data_normalize(data, mean, std):
    return (data - mean) / std

fish_data = np.column_stack((fish_length, fish_weight))
fish_target = np.concatenate((np.full(35, 'bream'), np.full(14, 'smelt')))

train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify=fish_target)

train_input_mean = np.mean(train_input, axis=0)
train_input_std = np.std(train_input, axis=0)

train_input_scaled = data_normalize(train_input, train_input_mean, train_input_std)
test_input_scaled = data_normalize(test_input, train_input_mean, train_input_std)

sample_data = np.array([[25, 150]])
sample_data_scaled = data_normalize(sample_data, train_input_mean, train_input_std)

kn = KNeighborsClassifier()
kn.fit(train_input_scaled, train_target)

score = kn.score(test_input_scaled, test_target)
predict = kn.predict(sample_data_scaled)
distances, indexes = kn.kneighbors(sample_data_scaled)

print(f'score: {score}')
print(f'prediction of {sample_data}: {predict}')

plt.scatter(train_input_scaled[:,0], train_input_scaled[:,1])
#plt.scatter(test_input_scaled[:,0], test_input_scaled[:,1])
plt.scatter(train_input_scaled[indexes, 0], train_input_scaled[indexes, 1], marker='D')
plt.scatter(sample_data_scaled[:,0], sample_data_scaled[:,1], marker='^')
#plt.xlim((0, 1000))
#plt.ylim((0, 1000))
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
