import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso
from data.perch_length_weight import perch_weight
from custom_functions.print_score import *

data_file = pd.read_csv(r'https://bit.ly/perch_csv_data')
perch_full = data_file.to_numpy()

train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight)

poly = PolynomialFeatures(degree=5, include_bias=False)
poly.fit(train_input)
train_poly, test_poly = [poly.transform(x) for x in (train_input, test_input)]

ss = StandardScaler()
ss.fit(train_poly)
train_scaled, test_scaled = [ss.transform(x) for x in (train_poly, test_poly)]

lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)

print_two_score(lasso.score(train_scaled, train_target), lasso.score(test_scaled, test_target))

#print(lasso.coef_)

train_score = []
test_score = []
alpha_degree_range = range(-3, 3)

for alpha_degree in alpha_degree_range:
    alpha = 10**alpha_degree
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(train_scaled, train_target)
    train_score.append(lasso.score(train_scaled, train_target))
    test_score.append(lasso.score(test_scaled, test_target))

plt.plot(alpha_degree_range, train_score, label='train score')
plt.plot(alpha_degree_range, test_score, label='test score')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()
plt.show()
