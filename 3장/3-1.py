import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from data.perch_length_weight import *

train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight)
train_input, test_input =[x.reshape(-1, 1) for x in (train_input, test_input)]

knr = KNeighborsRegressor(n_neighbors=3)

knr.fit(train_input, train_target)

test_prediction = knr.predict(test_input)

mae = mean_absolute_error(test_target, test_prediction)
print(f'MAE: {mae}')

print(f'test score: {knr.score(test_input, test_target)}')
print(f'train score: {knr.score(train_input, train_target)}')

plt.scatter(train_input, train_target)
plt.scatter(test_input, test_target)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
