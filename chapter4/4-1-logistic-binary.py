import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
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

bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

classes = lr.classes_
predict = lr.predict(train_bream_smelt[:5])
proba = lr.predict_proba(train_bream_smelt[:5])
coefs = lr.coef_
intercept = lr.intercept_
decisions = lr.decision_function(train_bream_smelt[:5])
phis = expit(decisions)

np.set_printoptions(precision=6, suppress=True)

print(f'predict: {predict}')
print(classes)
print(proba)
print(coefs, intercept)
print(f'z: {decisions}')
print(f'phi: {phis}')

z = np.arange(-10, 10, 0.1)
phi = expit(z)

fig, ax = plt.subplots()

ax.spines['left'].set_position('center')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.plot(z, phi)
ax.axhline(y=0.5, color='gray', linestyle='--')
ax.scatter(decisions, phis, marker='o', color='green')

plt.show()
