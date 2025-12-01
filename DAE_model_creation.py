# Third-Party Imports
import numpy as np
from keras import layers
from keras import callbacks
from keras.models import Model
import matplotlib.pyplot as plt

# First-Party Imports
from DAE_funcs import noisy_inputs, expected_outputs, WINDOW_WIDTH

input = layers.Input(shape=(WINDOW_WIDTH, 1))

# Encoder
x = layers.Conv1D(32, 3, activation="relu", padding="same")(input)
x = layers.MaxPooling1D(5, padding="same")(x)
x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling1D(2, padding="same")(x) ########################

# Decoder
x = layers.Conv1DTranspose(32, 3, strides=2, activation="relu", padding="same")(x) # These layers stride size seems to alter resolution of denoising in some way
x = layers.Conv1DTranspose(32, 3, strides=5, activation="relu", padding="same")(x)
x = layers.Conv1D(1, 3, activation="linear", padding="same")(x)

# Autoencoder
autoencoder = Model(input, x)
autoencoder.compile(optimizer="adam", loss="mean_squared_error")
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
    )
]

# Train the model
autoencoder.fit(
    x=x_train,
    y=y_train,
    epochs=50,
    batch_size=32,
    callbacks=model_callbacks,
    shuffle=True,
    validation_data=(x_test, y_test),
)

# Plot example for validation
predictions = autoencoder.predict(noisy_inputs)
plt.plot(noisy_inputs[11])
plt.plot(predictions[11])
plt.show()