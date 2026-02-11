import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

inputs = Input((8,))

h1 = Dense(4, activation='relu')(inputs)
h2 = Dense(8, activation='relu')(h1)
h3 = Dense(4, activation='relu')(h2)

outputs = Dense(10, activation='softmax')(h3)

model = Model(inputs, outputs)

model.summary(show_trainable=True)