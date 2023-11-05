import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
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

lr = LinearRegression()
lr.fit(train_scaled, train_target)

print_two_score(lr.score(train_scaled, train_target), lr.score(test_scaled, test_target))
