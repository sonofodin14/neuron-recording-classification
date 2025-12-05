# Third-Party Imports
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1' # Set to stop keras warning

import numpy as np
from keras import layers
from keras import callbacks
from keras.models import Model
import matplotlib.pyplot as plt

# First-Party Imports
from DAE_funcs import noisy_inputs, expected_outputs, WINDOW_WIDTH

input = layers.Input(shape=(WINDOW_WIDTH, 1))

# --- ENCODER ---
# Block 1 (Downsample by 2)
# Stride 2 replaces Pooling
c1 = layers.Conv1D(32, 7, padding="same", strides=2)(input) 
c1 = layers.BatchNormalization()(c1)
c1 = layers.LeakyReLU(alpha=0.1)(c1)
# c1 shape: (125, 32)

# Block 2 (Downsample by 5)
c2 = layers.Conv1D(64, 5, padding="same", strides=2)(c1)
c2 = layers.BatchNormalization()(c2)
c2 = layers.LeakyReLU(alpha=0.1)(c2)
# c2 shape: (25, 64) - This is the Bottleneck

# --- DECODER ---
# Block 3 (Upsample by 5 to match c1)
u1 = layers.UpSampling1D(size=2)(c2)
u1 = layers.AveragePooling1D(3, strides=1, padding="same")(u1)
u1 = layers.Conv1D(64, 5, padding="same")(u1) # Smooth after upsampling
u1 = layers.BatchNormalization()(u1)
u1 = layers.LeakyReLU(alpha=0.1)(u1)

# SKIP CONNECTION: Concatenate u1 (decoder) with c1 (encoder)
# This feeds high-res details back in
merge1 = layers.Concatenate()([u1, c1]) 

# Block 4 (Upsample by 2 to match input)
u2 = layers.UpSampling1D(size=2)(merge1)
u2 = layers.AveragePooling1D(3, strides=1, padding="same")(u2)
u2 = layers.Conv1D(32, 7, padding="same")(u2)
u2 = layers.BatchNormalization()(u2)
x = layers.LeakyReLU(alpha=0.1)(u2)

# # Encoder
# x = layers.Conv1D(32, 3, activation="relu", padding="same")(input)
# x = layers.AveragePooling1D(2, padding="same")(x)
# x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
# x = layers.AveragePooling1D(2, padding="same")(x)

# # Decoder
# x = layers.UpSampling1D(size=2)(x)
# x = layers.AveragePooling1D(4, strides=1, padding="same")(x)
# x = layers.Conv1D(64, 7, activation="relu", padding="same")(x)
# x = layers.UpSampling1D(size=2)(x)
# x = layers.AveragePooling1D(4, strides=1, padding="same")(x)
# x = layers.Conv1D(32, 5, activation="relu", padding="same")(x)

# Output
x = layers.Conv1D(1, 3, activation="linear", padding="same")(x)

# Autoencoder
autoencoder = Model(input, x)
autoencoder.compile(optimizer="adam", loss="mean_absolute_error")
autoencoder.summary()

# Split data into training/testing sets
x_train = np.asarray(noisy_inputs[0:int(0.8*len(noisy_inputs))])
x_test = np.asarray(noisy_inputs[int(0.8*len(noisy_inputs)):-1])

y_train = np.asarray(expected_outputs[0:int(0.8*len(expected_outputs))])
y_test = np.asarray(expected_outputs[int(0.8*len(expected_outputs)):-1])

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# Shuffle the training set
idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]

# Set callback to save best model
model_callbacks = [
    callbacks.ModelCheckpoint(
        "best_denoiser.keras", save_best_only=True, monitor="val_loss"
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001
    ),
    callbacks.EarlyStopping(
        monitor="val_loss", patience=15, verbose=1
    ),
]


# Train the model
autoencoder.fit(
    x=x_train,
    y=y_train,
    epochs=20,
    batch_size=128,
    callbacks=model_callbacks,
    shuffle=True,
    validation_data=(x_test, y_test),
    verbose=1,
)

# Plot example for validation
predictions = autoencoder.predict(noisy_inputs)
plt.plot(noisy_inputs[11])
plt.plot(predictions[11])
plt.show()