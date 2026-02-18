# =========================================================
# Binary Classification using CIFAR-10 (Cat vs Dog)
# Transfer Learning + Fine-Tuning Comparison with VGG16
# =========================================================

import tensorflow as tf
import matplotlib.pyplot as plt

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5   # increase to 10–20 for final training

# =========================================================
# 1️⃣ Load CIFAR-10
# =========================================================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# CIFAR-10 labels
# 3 = cat, 5 = dog
train_filter = (y_train.flatten() == 3) | (y_train.flatten() == 5)
test_filter  = (y_test.flatten() == 3) | (y_test.flatten() == 5)

x_train, y_train = x_train[train_filter], y_train[train_filter]
x_test, y_test   = x_test[test_filter],  y_test[test_filter]

# Convert labels to binary (cat=0, dog=1)
y_train = (y_train == 5).astype("float32")
y_test  = (y_test == 5).astype("float32")

# =========================================================
# 2️⃣ Preprocess (resize + normalize)
# =========================================================
def preprocess(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.map(preprocess).shuffle(1000).batch(BATCH_SIZE)

val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_ds = val_ds.map(preprocess).batch(BATCH_SIZE)

# =========================================================
# FUNCTION: Build Model
# =========================================================
def build_model():
    base_model = tf.keras.applications.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    x = tf.keras.layers.Flatten()(base_model.output)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    return model, base_model

# =========================================================
# 3️⃣ Experiment A — Full Fine-Tuning
# =========================================================
model_full, base_full = build_model()

for layer in base_full.layers:
    layer.trainable = True

model_full.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("Training FULL fine-tuning model...")
history_full = model_full.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# =========================================================
# 4️⃣ Experiment B — Partial Fine-Tuning
# =========================================================
model_partial, base_partial = build_model()

for layer in base_partial.layers[:15]:
    layer.trainable = False

for layer in base_partial.layers[15:]:
    layer.trainable = True

model_partial.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("Training PARTIAL fine-tuning model...")
history_partial = model_partial.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# =========================================================
# 5️⃣ Plot Comparison
# =========================================================
def plot_history(h1, h2, metric):
    plt.figure()
    plt.plot(h1.history[metric], label="Full Train")
    plt.plot(h1.history["val_" + metric], label="Full Val")
    plt.plot(h2.history[metric], label="Partial Train")
    plt.plot(h2.history["val_" + metric], label="Partial Val")
    plt.title(metric + " Comparison")
    plt.legend()
    plt.show()

plot_history(history_full, history_partial, "accuracy")
plot_history(history_full, history_partial, "loss")
