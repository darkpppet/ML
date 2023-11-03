import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2)

# sub_input, val_input, sub_target, val_target = train_test_split(
#     train_input, train_target, test_size=0.2)

dt = DecisionTreeClassifier()
# dt.fit(sub_input, sub_target)

splitter = StratifiedKFold(n_splits=10, shuffle=True)
scores = cross_validate(dt, train_input, train_target, cv=splitter)
print(np.mean(scores['test_score']))
