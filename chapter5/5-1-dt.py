import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from custom_functions.print_score import *

wine = pd.read_csv(r'https://bit.ly/wine_csv_data')

#print(wine.head())
#print(wine.info())
#print(wine.describe())

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2)

dt = DecisionTreeClassifier(min_impurity_decrease=0.0005)
dt.fit(train_input, train_target)

train_score = dt.score(train_input, train_target)
test_score = dt.score(test_input, test_target)

print_two_score(train_score, test_score)

feature_importances = dt.feature_importances_
print(feature_importances)

plt.figure(figsize=(20, 15), dpi=300)
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()
