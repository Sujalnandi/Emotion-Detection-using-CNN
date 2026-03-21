from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam


def build_resnet50_transfer(input_shape=(224, 224, 3), num_classes=7, learning_rate=1e-4, freeze_base=True):
    """Build and compile a transfer-learning model using ResNet50."""
    inputs = Input(shape=input_shape)

    # Attempt to load ImageNet weights. If unavailable, fall back to random init.
    try:
        base_model = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)
    except Exception:
        base_model = ResNet50(weights=None, include_top=False, input_tensor=inputs)

    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
