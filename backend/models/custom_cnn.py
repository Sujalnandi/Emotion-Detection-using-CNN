from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def build_custom_cnn(input_shape=(48, 48, 1), num_classes=7, learning_rate=1e-3):
    """Build and compile a stronger custom CNN baseline."""
    model = Sequential(
        [
            Input(shape=input_shape),
            Conv2D(32, (3, 3), padding="same", use_bias=False, kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation("relu"),
            Conv2D(32, (3, 3), padding="same", use_bias=False, kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            Conv2D(64, (3, 3), padding="same", use_bias=False, kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation("relu"),
            Conv2D(64, (3, 3), padding="same", use_bias=False, kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D((2, 2)),
            Dropout(0.35),
            Conv2D(128, (3, 3), padding="same", use_bias=False, kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation("relu"),
            Conv2D(128, (3, 3), padding="same", use_bias=False, kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D((2, 2)),
            Dropout(0.4),
            Conv2D(256, (3, 3), padding="same", use_bias=False, kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation("relu"),
            GlobalAveragePooling2D(),
            Dense(256, activation="relu", kernel_regularizer=l2(1e-4)),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
