import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2)

hgb = HistGradientBoostingClassifier()

scores = cross_validate(hgb, train_input, train_target,
                        return_train_score=True)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))

hgb.fit(train_input, train_target)
result = permutation_importance(hgb, train_input, train_target,
                                n_repeats=10, n_jobs=-1)

print(result.importances_mean)

result = permutation_importance(hgb, test_input, test_target,
                                n_repeats=10, n_jobs=-1)

print(result.importances_mean)

print(hgb.score(test_input, test_target))
