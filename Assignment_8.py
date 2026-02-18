import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model


def build_vgg_like(input_shape=(224, 224, 3), num_classes=10):

    inputs = Input(shape=input_shape)

    # ----- Block 1 -----
    x = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)

    # ----- Block 2 -----
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)

    # ----- Block 3 -----
    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)

    # ----- Block 4 -----
    x = Conv2D(512, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)

    # ----- Block 5 -----
    x = Conv2D(512, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)

    # ----- Classifier Head -----
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model


# -------- Example Compile --------
model = build_vgg_like(input_shape=(224,224,3), num_classes=10)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
     