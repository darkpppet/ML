import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from data.perch_length_weight import *

class Parabola:
    def __init__(self, coef, intercept):
        self.coef = coef
        self.intercept = intercept

    def calc_y(self, x):
        return x*x*self.coef[0] + x*self.coef[1] + self.intercept

def to_poly(n):
    n = np.array(n)
    return np.column_stack((n**2, n))

train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight)
train_poly, test_poly = [np.column_stack((x**2, x)) for x in (train_input, test_input)]

lr = LinearRegression()

lr.fit(train_poly, train_target)

print(f'train score: {lr.score(train_poly, train_target)}')
print(f'test score: {lr.score(test_poly, test_target)}')

sample_input = to_poly([50, 100])
sample_predict = lr.predict(sample_input)

regressed_parabola = Parabola(lr.coef_, lr.intercept_)
x_range = np.arange(0, 101, 0.1)
y_range = np.array([regressed_parabola.calc_y(x) for x in x_range])

plt.scatter(train_input, train_target, label='train data')
plt.scatter(test_input, test_target, label='test data')
plt.scatter(sample_input[:,1], sample_predict, marker='^', label='sample data')
plt.plot(x_range, y_range, label='regressed line')
plt.xlabel('length')
plt.ylabel('weight')
plt.legend()
plt.show()
