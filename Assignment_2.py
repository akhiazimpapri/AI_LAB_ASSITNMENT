import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create Sequential model		
model = Sequential([
    Dense(4, activation = 'relu', input_shape = (8,)),      
    Dense(6, activation = 'relu'),                                  
    Dense(8, activation = 'relu'),                                  
    Dense(6, activation = 'relu'),
    Dense(4, activation = 'relu'),
    Dense(2, activation='softmax')                          
])

model.summary(show_trainable=True)
