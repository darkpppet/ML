import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from data.perch_length_weight import *

class Line:
    def __init__(self, coef, intercept):
        self.coef = coef
        self.intercept = intercept

    def calc_y(self, x):
        return x*self.coef + self.intercept

train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight)
train_input, test_input = [x.reshape(-1, 1) for x in (train_input, test_input)]

lr = LinearRegression()

lr.fit(train_input, train_target)

print(f'train score: {lr.score(train_input, train_target)}')
print(f'test score: {lr.score(test_input, test_target)}')

sample_input = [[50], [100]]
sample_predict = lr.predict(sample_input)

regressed_line = Line(lr.coef_, lr.intercept_)
x_range = [5, 100]
y_range = [regressed_line.calc_y(x) for x in x_range]

plt.scatter(train_input, train_target, label='train data')
plt.scatter(test_input, test_target, label='test data')
plt.scatter(sample_input, sample_predict, marker='^', label='sample data')
plt.plot(x_range, y_range, label='regressed line')
plt.xlabel('length')
plt.ylabel('weight')
plt.legend()
plt.show()
