# Third-Party Imports
import matplotlib.pyplot as plt
import scipy.io as spio
import numpy as np
from skimage.restoration import denoise_wavelet
import scipy.signal as sig
from random import randint
import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler

TF_ENABLE_ONEDNN_OPTS=0
SPIKE_WIDTH = 75

# Functions
def load_data():
    files = ['D2.mat', 'D3.mat', 'D4.mat', 'D5.mat', 'D6.mat']
    data_entries = []
    for file in files:
        mat = spio.loadmat(file, spueeze_me=True)
        data_entries.append(mat['d'])
    return data_entries[0], data_entries[1], data_entries[2], data_entries[3], data_entries[4]

def load_file_data(filepath):
    mat = spio.loadmat(filepath, squeeze_me=True)
    return mat['d']

def load_training_data():
    mat = spio.loadmat('TRAINING DATA/D1.mat', squeeze_me=True)
    d = mat['d']
    Index = mat['Index']
    Class = mat['Class']
    return d, Index, Class

def create_hp_filter(numtaps, fc, fs):
    filter_coefficients = sig.firwin(numtaps, fc, pass_zero=False, window='hamming', fs=fs)
    return filter_coefficients

def filter_data(data, filter_coefficients, numtaps):
    filtered = sig.lfilter(filter_coefficients, 1.0, data)
    filtered = np.roll(filtered, -(int(numtaps/2)))
    return filtered

def wavelet_denoising(data):
    denoised = denoise_wavelet(
        data, 
        method='BayesShrink', 
        mode='soft', 
        wavelet_levels=5,
        wavelet='bior6.8',
        rescale_sigma='True'
        )
    return denoised

def minmax_scale(data):
    scaler = MinMaxScaler()
    data_shaped = np.asarray(data).reshape(-1, 1)
    data_scaled = scaler.fit_transform(data_shaped)
    return data_scaled

def noise_dependent_peak_detection(data):
    # Calculating the Dynamic Peak Threshold dependent on standard deviation of data - this changes with noise
    stand_devs = [abs(x)/0.6745 for x in data]
    threshold = 0.75*np.median(stand_devs)
    peaks, _ = sig.find_peaks(data, height=threshold, distance=10, prominence=0.125)
    return peaks

def peaks_to_spike_index(peaks):
    index = np.asarray([x - 15 for x in peaks])
    return index

def extract_spike_windows(data, index):
    spikes = []
    for i in index:
        spikes.append(data[i:i+SPIKE_WIDTH])
    # Pad any shorter items
    max_len = max(len(sublist) for sublist in spikes)
    padded_spikes = [list(sublist) + [0] * (max_len - len(sublist)) for sublist in spikes]
    return np.asarray(padded_spikes)

def load_classifier_model():
    return keras.models.load_model("best_model.keras")

def load_denoising_model():
    return keras.models.load_model("best_denoiser.keras")

def classify_spikes(spikes):
    # Load model and classify spikes
    loaded_model = load_classifier_model()

    # Reshape spikes to fit model input
    spikes = np.asarray(spikes)
    spikes = spikes.reshape((spikes.shape[0], spikes.shape[1], 1))
    Class = loaded_model.predict(spikes, verbose=2)

    # Convert Class into labelled prediction
    new_class = []
    for i in range(len(Class)):
        classification = np.argmax(Class[i]) + 1
        new_class.append(int(classification))
    
    return new_class

def save_mat_file(Index, Class, filename):
    filename = 'RESULT DATA/' + filename
    # Pack Index and Class into .mat file
    mat_data = {
        'Index': Index,
        'Class': Class
    }
    spio.savemat(file_name=filename, mdict=mat_data)

def make_model_OLD(input_shape, num_classes):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=32, kernel_size=5, padding="same", strides=1)(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same", strides=1)(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    # conv3 = keras.layers.Conv1D(filters=24, kernel_size=4, padding="same")(conv2)
    # conv3 = keras.layers.BatchNormalization()(conv3)
    # conv3 = keras.layers.ReLU()(conv3)

    # conv4 = keras.layers.Conv1D(filters=48, kernel_size=4, padding="same", strides=4)(conv3)
    # conv4 = keras.layers.BatchNormalization()(conv4)
    # conv4 = keras.layers.ReLU()(conv4)

    gap = keras.layers.MaxPooling1D(pool_size=2)(conv2)

    lstm1 = keras.layers.Bidirectional(keras.layers.LSTM(units=64, activation='relu', return_sequences=False))(gap)

    dense_layer1 = keras.layers.Dense(256, activation="relu")(lstm1)

    # dense_layer2 = keras.layers.Dense(256, activation="relu")(dense_layer1)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(dense_layer1)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.summary()

    return model

def make_model(input_shape, num_classes):
    input_layer = keras.layers.Input(input_shape)

    # Branch 1: Small details
    branch_a = layers.Conv1D(filters=32, kernel_size=4, padding="same")(input_layer)
    branch_a = layers.BatchNormalization()(branch_a)
    branch_a = layers.ReLU()(branch_a)

    # Branch 2: Medium patterns
    branch_b = layers.Conv1D(filters=48, kernel_size=6, padding="same")(input_layer)
    branch_b = layers.BatchNormalization()(branch_b)
    branch_b = layers.ReLU()(branch_b)

    # Branch 3: Long trends
    branch_c = layers.Conv1D(filters=64, kernel_size=12, padding="same")(input_layer)
    branch_c = layers.BatchNormalization()(branch_c)
    branch_c = layers.ReLU()(branch_c)

    x = layers.Concatenate()([branch_a, branch_b, branch_c])

    x = layers.Conv1D(filters=64, kernel_size=4, padding="same", strides=1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Bidirectional(layers.LSTM(units=72, return_sequences=False))(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    output_layer = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    model.summary()

    return model

def train_model(model, x_train, y_train, epochs=500, batch_size=32):
    callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_model.keras", save_best_only=True, monitor="val_categorical_accuracy"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=0.000001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    ]
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    model.compile(
        optimizer="adamw",
        loss=loss,
        metrics=["categorical_accuracy"],
    )
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split=0.2,
        verbose=1,
    )
    return history

if __name__ == "__main__":
    pass