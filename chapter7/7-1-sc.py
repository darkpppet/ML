import numpy as np
from tensorflow import keras
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier

(train_input, train_target), (test_input, test_target) =\
    keras.datasets.fashion_mnist.load_data()

train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)

sc = SGDClassifier(loss='log_loss', max_iter=20)

scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)

print(np.mean(scores['test_score']))
