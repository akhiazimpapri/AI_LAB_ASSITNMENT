import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# -----------------------------
# 1. Generate Dataset
# -----------------------------
np.random.seed(42)
x = np.linspace(-15, 15, 2000).reshape(-1, 1)
y = 7*x**2 - 4*x + 6  # simplified polynomial

# -----------------------------
# 2. Normalize [-1,1]
# -----------------------------
def normalize(data):
    min_val = data.min()
    max_val = data.max()
    return 2 * (data - min_val) / (max_val - min_val) - 1, min_val, max_val

def denormalize(norm, min_val, max_val):
    return (norm + 1) * (max_val - min_val) / 2 + min_val

x_norm, x_min, x_max = normalize(x)
y_norm, y_min, y_max = normalize(y)

# -----------------------------
# 3. Split Data (80/10/10)
# -----------------------------
x_train, x_temp, y_train, y_temp = train_test_split(x_norm, y_norm, test_size=0.2, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# -----------------------------
# 4. Build Model
# -----------------------------
model = Sequential([
    Dense(32, activation='relu', input_shape=(1,)),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
model.summary()

# -----------------------------
# 5. Train Model
# -----------------------------
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=100,
    batch_size=32,
    verbose=1
)

# -----------------------------
# 6. Evaluate on Test
# -----------------------------
y_pred_norm = model.predict(x_test)
y_test_true = denormalize(y_test, y_min, y_max)
y_pred_true = denormalize(y_pred_norm, y_min, y_max)

# R² Accuracy
r2 = r2_score(y_test_true, y_pred_true)
print("Test R² Accuracy:", r2)

# -----------------------------
# 7. Plot Training & Validation Loss
# -----------------------------
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.legend()
plt.show()

# -----------------------------
# 8. Plot Prediction vs True Values
# -----------------------------
plt.figure(figsize=(6,6))
plt.scatter(y_test_true, y_pred_true, color='blue', alpha=0.6, label='Predictions')
plt.plot([y_test_true.min(), y_test_true.max()],
         [y_test_true.min(), y_test_true.max()],
         color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Prediction Accuracy vs True Levels (Test Data)")
plt.legend()
plt.show()