import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from data.perch_length_weight import *

train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight)
train_input, test_input = [x.reshape(-1, 1) for x in (train_input, test_input)]

knr = KNeighborsRegressor(n_neighbors=3)

knr.fit(train_input, train_target)

sample_input = [[50], [100]]
sample_predict = knr.predict(sample_input)

distances, indexes = knr.kneighbors(sample_input)

print(f'predict: {sample_predict}')

plt.scatter(train_input, train_target, label='train data')
#plt.scatter(test_input, test_target, label='test data')
plt.scatter(train_input[indexes], train_target[indexes], marker='D', label='neighbors')
plt.scatter(sample_input, sample_predict, marker='^', label='sample data')
plt.xlabel('length')
plt.ylabel('weight')
plt.legend()
plt.show()
