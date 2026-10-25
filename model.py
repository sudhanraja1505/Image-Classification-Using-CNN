"""CNN architecture for CIFAR-10 (32x32 colour images, 10 classes)."""
from tensorflow import keras
from tensorflow.keras import layers

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def conv_block(x, filters, dropout):
    """Two Conv-BatchNorm-ReLU layers, then downsample and regularise."""
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    return layers.Dropout(dropout)(x)


def build_model(num_classes=len(CLASSES)):
    inputs = keras.Input(shape=(32, 32, 3))
    # Data augmentation: active only during training, ignored at prediction time
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomTranslation(0.1, 0.1)(x)
    x = layers.RandomRotation(0.05)(x)
    x = layers.Rescaling(1.0 / 255)(x)          # raw 0-255 pixels go in; scaling is part of the model
    for filters, drop in [(32, 0.2), (64, 0.3), (128, 0.4)]:
        x = conv_block(x, filters, drop)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="cifar10_cnn")
