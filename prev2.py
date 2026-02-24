import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adagrad
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. Load MNIST
# -----------------------------

data = np.load('/Users/akhi/Desktop/Akhi/AI Assignment/my_dataset.npz')
x_train_full = data['x_train']
y_train_full = data['y_train']
x_test = data['x_test']
y_test = data['y_test']

x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train_full = np.expand_dims(x_train_full, -1)
x_test = np.expand_dims(x_test, -1)

# Odd = 1, Even = 0
y_train_full = (y_train_full % 2 == 1).astype(np.float32)
y_test = (y_test % 2 == 1).astype(np.float32)

# -----------------------------
# 2. Split data
# -----------------------------
x_train_85, _, y_train_85, _ = train_test_split(
    x_train_full, y_train_full, test_size=0.15, random_state=42
)
x_train, x_val, y_train, y_val = train_test_split(
    x_train_85, y_train_85, test_size=0.15, random_state=42
)

# -----------------------------
# 3. Build CNN
# -----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1), name="conv1"),
    MaxPooling2D(),

    Conv2D(64, (3,3), activation='relu', name="conv2"),
    MaxPooling2D(),

    Conv2D(128, (3,3), activation='relu', name="conv3"),
    MaxPooling2D(),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

from keras.optimizers import Adam
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------------
# 4. Model Checkpoint
# -----------------------------
checkpoint = ModelCheckpoint(
    "best_model.keras", monitor="val_loss", save_best_only=True, mode="min", verbose=1
)

# -----------------------------
# 5. Train first 10 epochs (all layers trainable)
# -----------------------------
history1 = model.fit(
    x_train, y_train, validation_data=(x_val, y_val),
    epochs=10, batch_size=32, callbacks=[checkpoint]
)

# -----------------------------
# 6. Freeze first 3 conv layers
# -----------------------------
model.get_layer("conv1").trainable = False
model.get_layer("conv2").trainable = False
model.get_layer("conv3").trainable = False

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# -----------------------------
# 7. Train next 20 epochs
# -----------------------------
history2 = model.fit(
    x_train, y_train, validation_data=(x_val, y_val),
    epochs=20, batch_size=32, callbacks=[checkpoint]
)

# -----------------------------
# 8. Save final model
# -----------------------------
model.save('my_model.keras')
print("Final model saved as 'my_model.keras'")

# -----------------------------
# 9. Load and plot predictions
# -----------------------------
loaded_model = load_model('my_model.keras')
y_pred_prob = loaded_model.predict(x_test)
y_pred = (y_pred_prob > 0.5).astype(int)

plt.figure(figsize=(15,4))
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    plt.axis('off')
    plt.title(f"P:{y_pred[i][0]}\nT:{int(y_test[i])}")
plt.show()

# -----------------------------
# 10. Plot Loss and Accuracy
# -----------------------------
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
epochs = range(1, 31)

plt.figure()
plt.plot(epochs, loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Val Loss')
plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Loss vs Epochs'); plt.legend(); plt.show()

plt.figure()
plt.plot(epochs, acc, label='Train Accuracy')
plt.plot(epochs, val_acc, label='Val Accuracy')
plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.title('Accuracy vs Epochs'); plt.legend(); plt.show()

#If instead your dataset is actually images in folders
#train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #'/Users/akhi/Desktop/Akhi/AI Assignment/my_dataset/train',
    #image_size=(28,28),
    #color_mode='grayscale',
    #batch_size=32,
    #label_mode='int'
#)

#test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #'/Users/akhi/Desktop/Akhi/AI Assignment/my_dataset/test',
    #image_size=(28,28),
    #color_mode='grayscale',
   # batch_size=32,
   # label_mode='int'
#)