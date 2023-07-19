import time
import numpy as np

from sklearn.datasets import make_classification

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

n_gpus = len(tf.config.list_physical_devices('GPU'))
print(f'Using this many GPUs: {n_gpus}')

X, y = make_classification(
    n_samples=250_000,
    n_features=200,
    n_informative=200,
    n_redundant=0,
    n_classes=10,
    random_state=42
)

model = Sequential()
model.add(Dense(128, activation="relu", input_shape=(200,)))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(
    optimizer='adam',
    loss="sparse_categorical_crossentropy",
    metrics=["acc"]
)

start = time.time()
hist = model.fit(
    X, y,
    batch_size=64,
    epochs=5,
    validation_split=0.2
)
end = time.time()

dur = round(end - start, 2)
print(f"The model took {dur} seconds to fit.")
