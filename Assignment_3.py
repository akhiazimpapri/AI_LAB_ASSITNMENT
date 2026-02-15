import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------
# 1️⃣ Reproducibility
# ---------------------------------------------------------
np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------------------------------------
# 2️⃣ Dataset
# ---------------------------------------------------------
x = np.linspace(-10, 10, 500).reshape(-1, 1)

y_linear = 5*x + 10
y_quadratic = 3*x**2 + 5*x + 10
y_cubic = 4*x**3 + 3*x**2 + 5*x + 10

# ---------------------------------------------------------
# 3️⃣ Normalization
# ---------------------------------------------------------
scaler_x = MinMaxScaler()
x_scaled = scaler_x.fit_transform(x)

scaler_y1 = MinMaxScaler()
scaler_y2 = MinMaxScaler()
scaler_y3 = MinMaxScaler()

y_linear = scaler_y1.fit_transform(y_linear)
y_quadratic = scaler_y2.fit_transform(y_quadratic)
y_cubic = scaler_y3.fit_transform(y_cubic)

# ---------------------------------------------------------
# 4️⃣ Split
# ---------------------------------------------------------
def split_data(x, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.30, random_state=42)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

data1 = split_data(x_scaled, y_linear)
data2 = split_data(x_scaled, y_quadratic)
data3 = split_data(x_scaled, y_cubic)

# ---------------------------------------------------------
# 5️⃣ Model
# ---------------------------------------------------------
def build_model(neurons=32):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(neurons, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(neurons, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    return model

# ---------------------------------------------------------
# 6️⃣ Early Stopping
# ---------------------------------------------------------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

# ---------------------------------------------------------
# 7️⃣ Training (Epochs Visible)
# ---------------------------------------------------------
print("\nTraining Linear Model")
model1 = build_model(16)
hist1 = model1.fit(
    data1[0], data1[3],
    validation_data=(data1[1], data1[4]),
    epochs=300,
    callbacks=[early_stop],
    verbose=1
)

print("\nTraining Quadratic Model")
model2 = build_model(32)
hist2 = model2.fit(
    data2[0], data2[3],
    validation_data=(data2[1], data2[4]),
    epochs=400,
    callbacks=[early_stop],
    verbose=1
)

print("\nTraining Cubic Model")
model3 = build_model(64)
hist3 = model3.fit(
    data3[0], data3[3],
    validation_data=(data3[1], data3[4]),
    epochs=500,
    callbacks=[early_stop],
    verbose=1
)

# ---------------------------------------------------------
# 8️⃣ Evaluation
# ---------------------------------------------------------
print("\nTest Loss Linear:", model1.evaluate(data1[2], data1[5], verbose=0))
print("Test Loss Quadratic:", model2.evaluate(data2[2], data2[5], verbose=0))
print("Test Loss Cubic:", model3.evaluate(data3[2], data3[5], verbose=0))

# ---------------------------------------------------------
# 9️⃣ Plot Results (Colored)
# ---------------------------------------------------------
def plot_results(x_orig, y_scaler, y_true_scaled, model, title,
                 scatter_color, line_color):
    
    y_pred_scaled = model.predict(x_scaled, verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = y_scaler.inverse_transform(y_true_scaled)

    plt.figure(figsize=(7,4))
    plt.scatter(x_orig, y_true, label="Original", color=scatter_color, alpha=0.5)
    plt.plot(x_orig, y_pred, label="Predicted", color=line_color, linewidth=2)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

plot_results(x, scaler_y1, y_linear, model1, "Linear Function", "blue", "red")
plot_results(x, scaler_y2, y_quadratic, model2, "Quadratic Function", "green", "orange")
plot_results(x, scaler_y3, y_cubic, model3, "Cubic Function", "purple", "black")

# ---------------------------------------------------------
# 🔟 Plot Loss (Colored)
# ---------------------------------------------------------
def plot_loss(history, title, train_color, val_color):
    plt.figure(figsize=(7,4))
    plt.plot(history.history['loss'], label='Train Loss', color=train_color)
    plt.plot(history.history['val_loss'], label='Validation Loss', color=val_color)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

plot_loss(hist1, "Linear Training Loss", "blue", "red")
plot_loss(hist2, "Quadratic Training Loss", "green", "orange")
plot_loss(hist3, "Cubic Training Loss", "purple", "black")
