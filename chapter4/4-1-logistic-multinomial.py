import pandas as pd
import numpy as np
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from custom_functions.print_score import *

fish = pd.read_csv('https://bit.ly/fish_csv_data')

#head = fish.head()
#fish_unique_species = pd.unique(fish['Species'])

fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target)

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)

train_score = lr.score(train_scaled, train_target)
test_score = lr.score(test_scaled, test_target)
predict = lr.predict(test_scaled[:5])
proba = lr.predict_proba(test_scaled[:5])

np.set_printoptions(precision=6, suppress=True)

print_two_score(train_score, test_score)
print(predict)
print(lr.classes_)
print(np.round(proba, decimals=5))

decision = lr.decision_function(test_scaled[:5])
proba = softmax(decision, axis=1)
print(np.round(decision, decimals=3))
print(np.round(proba, decimals=5))
