# =========================================================
# FCFNN Classifier for Fashion-MNIST, MNIST, CIFAR-10
# (Accuracy Improved — Structure Same)
# =========================================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------------------------------------
# 1️⃣ Load Datasets
# ---------------------------------------------------------
from tensorflow.keras.datasets import fashion_mnist, mnist, cifar10

(fx_train, fy_train), (fx_test, fy_test) = fashion_mnist.load_data()
(mx_train, my_train), (mx_test, my_test) = mnist.load_data()
(cx_train, cy_train), (cx_test, cy_test) = cifar10.load_data()

# ---------------------------------------------------------
# 2️⃣ Preprocessing Function (UNCHANGED)
# ---------------------------------------------------------
def preprocess_images(x_train, x_test):
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    
    return x_train, x_test

fx_train, fx_test = preprocess_images(fx_train, fx_test)
mx_train, mx_test = preprocess_images(mx_train, mx_test)
cx_train, cx_test = preprocess_images(cx_train, cx_test)

cy_train = cy_train.flatten()
cy_test = cy_test.flatten()

# ---------------------------------------------------------
# 3️⃣ Build FCFNN Classifier (Improved)
# ---------------------------------------------------------
def build_classifier(input_dim, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation="relu", input_shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ---------------------------------------------------------
# 🔹 Callbacks (NEW but no workflow change)
# ---------------------------------------------------------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)

lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    min_lr=1e-5
)

# ---------------------------------------------------------
# 4️⃣ Train Function (Same — only callbacks added)
# ---------------------------------------------------------
def train_and_evaluate(x_train, y_train, x_test, y_test, dataset_name):
    model = build_classifier(x_train.shape[1], len(np.unique(y_train)))
    
    print(f"\nTraining on {dataset_name}")
    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=35,  # slightly more training
        batch_size=128,
        callbacks=[early_stop, lr_schedule],
        verbose=1
    )
    
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{dataset_name} Test Accuracy: {acc:.4f}")
    
    return history

# ---------------------------------------------------------
# 5️⃣ Train on Each Dataset (UNCHANGED)
# ---------------------------------------------------------
hist_fashion = train_and_evaluate(fx_train, fy_train, fx_test, fy_test, "Fashion-MNIST")
hist_mnist = train_and_evaluate(mx_train, my_train, mx_test, my_test, "MNIST")
hist_cifar = train_and_evaluate(cx_train, cy_train, cx_test, cy_test, "CIFAR-10")

# ---------------------------------------------------------
# 6️⃣ Plot Training Curves (UNCHANGED)
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
