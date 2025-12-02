import utils
import DAE_funcs
from DAE_funcs import WINDOW_WIDTH, OVERLAP
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from sklearn.preprocessing import OneHotEncoder

if __name__ == "__main__":
    # Import training data from D1.mat
    d, Index, Class =  utils.load_training_data()

    # # Use wavelets to denoise the data
    # denoised_d = utils.wavelet_denoising(d)

    # High-pass filter the data
    numtaps = 1501
    fc = 50
    fs = 25000
    filter_coef = utils.create_hp_filter(numtaps, fc, fs)
    filtered_d = utils.filter_data(d, filter_coef, numtaps)

    # Load denoising model
    denoiser = utils.load_denoising_model()

    # Split data into windows and denoise
    noisy_windows = DAE_funcs.list_to_overlapping_windows(filtered_d, WINDOW_WIDTH, OVERLAP)
    predictions = denoiser.predict(noisy_windows, verbose=2)
    clean_windows = np.squeeze(predictions, axis=-1)
    # Merge windows back into single stream
    clean_data = DAE_funcs.overlapping_windows_to_list(clean_windows, OVERLAP)
    # Scale data to range [0,1]
    clean_data_scaled = utils.minmax_scale(clean_data)

    # Extract spikes from indexes
    spikes = utils.extract_spike_windows(clean_data_scaled, Index)

    # Create new spikes array that are shifted by random values to increase training robustness
    shifted_spikes_1 = []
    for spike in spikes:
        shift = randint(-5, 5)
        shifted_spikes_1.append(np.roll(spike, shift))

    shifted_spikes_2 = []
    for spike in spikes:
        shift = randint(-10, 10)
        shifted_spikes_2.append(np.roll(spike, shift))

    # Combine the original and added data
    spikes = np.asarray(list(spikes) + list(shifted_spikes_1) + list(shifted_spikes_2))
    Class = np.asarray(list(Class) + list(Class) + list(Class))

    # Shuffle the combined data
    idx = np.random.permutation(len(spikes))
    spikes = spikes[idx]
    Class = Class[idx]

    # Split data into training/testing sets
    x_train = np.asarray(spikes[0:int(0.8*len(spikes))])
    x_test = np.asarray(spikes[int(0.8*len(spikes)):-1])

    y_train = np.asarray(Class[0:int(0.8*len(Class))]) - 1
    y_test = np.asarray(Class[int(0.8*len(Class)):-1]) - 1

    # Convert classes to one hot encoded
    encoder = OneHotEncoder(sparse_output=False)
    y_test_reshape = y_test.reshape(-1, 1)
    y_test = encoder.fit_transform(y_test_reshape)
    y_train_reshape = y_train.reshape(-1, 1)
    y_train = encoder.fit_transform(y_train_reshape)

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # Shuffle the training set
    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]

    num_classes = 5
    model = utils.make_model(input_shape=x_train.shape[1:], num_classes=num_classes)

    # Training the model
    epochs = 500
    batch_size = 64
    history = utils.train_model(model, x_train, y_train, epochs=epochs, batch_size=batch_size)

    # Load and evaluate best model
    model = utils.load_classifier_model()
    test_loss, test_acc = model.evaluate(x_test, y_test)

    print("Test accuracy", test_acc)
    print("Test loss", test_loss)

    # Plot model's losses
    metric = "categorical_accuracy"
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.show()
    plt.close()