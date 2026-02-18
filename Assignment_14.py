#question 14
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# 1. Load Dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# 2. Model Builder
def build_model(activation='relu', loss_fn='categorical_crossentropy'):
    model = models.Sequential([
        layers.Input(shape=(32,32,3)),
        layers.Conv2D(32, (3,3), activation=activation),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation=activation),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation=activation),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss=loss_fn,
        metrics=['accuracy']
    )
    return model

# 3. Experiments
configs = [
    ('relu', 'categorical_crossentropy'),
    ('tanh', 'categorical_crossentropy'),
    ('relu', 'mean_squared_error'),
]

for act, loss in configs:
    print(f"\nTraining with Activation={act}, Loss={loss}")
    model = build_model(act, loss)

    if loss == 'categorical_crossentropy':
        model.fit(x_train, y_train_cat, epochs=5,
                  validation_data=(x_test, y_test_cat), batch_size=64)
    else:
        model.fit(x_train, y_train_cat, epochs=5,
                  validation_data=(x_test, y_test_cat), batch_size=64)

    test_loss, test_acc = model.evaluate(x_test, y_test_cat)
    print("Test Accuracy:", test_acc)