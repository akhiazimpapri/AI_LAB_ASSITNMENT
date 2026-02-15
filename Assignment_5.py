# =========================================================
# CNN Classifier for Fashion-MNIST, MNIST, CIFAR-10
# =========================================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(42)
tf.random.set_seed(42)

from tensorflow.keras.datasets import fashion_mnist, mnist, cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ---------------------------------------------------------
# 1️⃣ Load datasets
# ---------------------------------------------------------
(fx_train, fy_train), (fx_test, fy_test) = fashion_mnist.load_data()
(mx_train, my_train), (mx_test, my_test) = mnist.load_data()
(cx_train, cy_train), (cx_test, cy_test) = cifar10.load_data()

cy_train = cy_train.flatten()
cy_test = cy_test.flatten()

# ---------------------------------------------------------
# 2️⃣ Preprocessing function
# ---------------------------------------------------------
def preprocess_images(x_train, x_test, grayscale=True):
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    if grayscale:
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
    return x_train, x_test

fx_train, fx_test = preprocess_images(fx_train, fx_test, grayscale=True)
mx_train, mx_test = preprocess_images(mx_train, mx_test, grayscale=True)
cx_train, cx_test = preprocess_images(cx_train, cx_test, grayscale=False)  # RGB

# ---------------------------------------------------------
# 3️⃣ Build CNN Model
# ---------------------------------------------------------
def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", padding="same", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        Conv2D(64, (3,3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        Conv2D(128, (3,3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ---------------------------------------------------------
# 4️⃣ Callbacks
# ---------------------------------------------------------
early_stop = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
lr_schedule = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-5)

# ---------------------------------------------------------
# 5️⃣ Training Function
# ---------------------------------------------------------
def train_and_test(x_train, y_train, x_test, y_test, name):
    model = build_cnn(x_train.shape[1:], len(np.unique(y_train)))

    print(f"\nTraining on {name}")
    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=30,
        batch_size=128,
        callbacks=[early_stop, lr_schedule],
        verbose=1
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{name} Test Accuracy: {acc:.4f}")

    return history

# ---------------------------------------------------------
# 6️⃣ Train on Each Dataset
# ---------------------------------------------------------
hist_fashion = train_and_test(fx_train, fy_train, fx_test, fy_test, "Fashion-MNIST")
hist_mnist = train_and_test(mx_train, my_train, mx_test, my_test, "MNIST")
hist_cifar = train_and_test(cx_train, cy_train, cx_test, cy_test, "CIFAR-10")

# ---------------------------------------------------------
# 7️⃣ Plot Accuracy
# ---------------------------------------------------------
def plot_history(history, title, color1, color2):
    plt.figure(figsize=(7,4))
    plt.plot(history.history["accuracy"], label="Train Acc", color=color1)
    plt.plot(history.history["val_accuracy"], label="Val Acc", color=color2)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

plot_history(hist_fashion, "Fashion-MNIST Accuracy", "blue", "red")
plot_history(hist_mnist, "MNIST Accuracy", "green", "orange")
plot_history(hist_cifar, "CIFAR-10 Accuracy", "purple", "black")
