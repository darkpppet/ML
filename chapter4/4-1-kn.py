import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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

kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)
train_score = kn.score(train_scaled, train_target)
test_score = kn.score(test_scaled, test_target)

print_two_score(train_score, test_score)

classes = kn.classes_
predict = kn.predict(test_scaled[:5])
proba = kn.predict_proba(test_scaled[:5])

print(f'predict: {predict}')
print(classes)
print(np.round(proba, decimals=5))

distances, indexes = kn.kneighbors(test_scaled[3:4])
print(train_target[indexes])
