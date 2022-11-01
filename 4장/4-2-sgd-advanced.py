import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from custom_functions.print_score import *

fish = pd.read_csv('https://bit.ly/fish_csv_data')

fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target)

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

sc = SGDClassifier(loss='log_loss')

train_score = []
test_score = []
classes = np.unique(train_target)

for _ in range(300):
    sc.partial_fit(train_scaled, train_target, classes=classes)
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))

plt.plot(train_score, label='train score')
plt.plot(test_score, label='test score')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()

sc_final = SGDClassifier(loss='log_loss')
sc_final.fit(train_scaled, train_target)

train_score_final = sc_final.score(train_scaled, train_target)
test_score_final = sc_final.score(test_scaled, test_target)

print('--- loss: log ---')
print_two_score(train_score_final, test_score_final)

sc_hinge = SGDClassifier(loss='hinge')
sc_hinge.fit(train_scaled, train_target)

train_score_hinge = sc_hinge.score(train_scaled, train_target)
test_score_hinge = sc_hinge.score(test_scaled, test_target)

print('--- loss: hinge ---')
print_two_score(train_score_hinge, test_score_hinge)
