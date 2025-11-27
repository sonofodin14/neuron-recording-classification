# Third-Party Imports
import matplotlib.pyplot as plt
import scipy.io as spio
import numpy as np
from skimage.restoration import denoise_wavelet
import scipy.signal as sig
from random import randint
import keras
from sklearn import preprocessing

TF_ENABLE_ONEDNN_OPTS=0
SPIKE_WIDTH = 100

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

def noise_dependent_peak_detection(data):
    # Calculating the Dynamic Peak Threshold dependent on standard deviation of data - this changes with noise
    stand_devs = [abs(x)/0.6745 for x in data]
    threshold = 4*np.median(stand_devs)
    peaks, _ = sig.find_peaks(data, height=threshold, distance=10, prominence=0.125)
    return peaks

def peaks_to_spike_index(peaks):
    index = np.asarray([x - 12 for x in peaks])
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

def make_model(input_shape, num_classes):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=24, kernel_size=4, padding="same", strides=4)(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=12, kernel_size=3, padding="same", strides=4)(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=16, kernel_size=4, padding="same", strides=2)(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    conv4 = keras.layers.Conv1D(filters=2, kernel_size=4, padding="same", strides=1)(conv3)
    conv4 = keras.layers.BatchNormalization()(conv4)
    conv4 = keras.layers.ReLU()(conv4)

    gap = keras.layers.GlobalAveragePooling1D()(conv4)

    dense_layer1 = keras.layers.Dense(256, activation="relu")(gap)

    dense_layer2 = keras.layers.Dense(64, activation="relu")(dense_layer1)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(dense_layer2)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)

def train_model(model, x_train, y_train, epochs=500, batch_size=32):
    callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_model.keras", save_best_only=True, monitor="val_sparse_categorical_accuracy"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=40, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=75, verbose=1),
    ]
    model.compile(
        optimizer="adamw",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
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