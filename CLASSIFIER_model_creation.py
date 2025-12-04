import utils
import DAE_funcs
from DAE_funcs import WINDOW_WIDTH, OVERLAP, add_brownian_white_noise
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from sklearn.preprocessing import OneHotEncoder

if __name__ == "__main__":
    # Import training data from D1.mat
    d, Index, Class =  utils.load_training_data()

    d_n1 = add_brownian_white_noise(d, 0.5)
    d_n2 = add_brownian_white_noise(d, 1)
    d_n3 = add_brownian_white_noise(d, 2)
    d_n4 = add_brownian_white_noise(d, 3)
    d_n5 = add_brownian_white_noise(d, 4)

    # High-pass filter the data
    numtaps = 1501
    fc = 50
    fs = 25000
    filter_coef = utils.create_hp_filter(numtaps, fc, fs)
    filtered_d = utils.filter_data(d, filter_coef, numtaps)
    # filtered_d = utils.minmax_scale(filtered_d)

    filtered_d_n1 = utils.filter_data(d_n1, filter_coef, numtaps)
    # filtered_d_n1 = utils.minmax_scale(filtered_d_n1)

    filtered_d_n2 = utils.filter_data(d_n2, filter_coef, numtaps)
    # filtered_d_n2 = utils.minmax_scale(filtered_d_n2)

    filtered_d_n3 = utils.filter_data(d_n3, filter_coef, numtaps)
    # filtered_d_n3 = utils.minmax_scale(filtered_d_n3)

    filtered_d_n4 = utils.filter_data(d_n4, filter_coef, numtaps)
    # filtered_d_n4 = utils.minmax_scale(filtered_d_n4)

    filtered_d_n5 = utils.filter_data(d_n5, filter_coef, numtaps)
    # filtered_d_n5 = utils.minmax_scale(filtered_d_n5)

    # Load denoising model
    denoiser = utils.load_denoising_model()

    # Split data into windows and denoise
    noisy_windows0 = DAE_funcs.list_to_overlapping_windows(filtered_d, WINDOW_WIDTH, OVERLAP)
    predictions0 = denoiser.predict(noisy_windows0, verbose=2)
    clean_windows0 = np.squeeze(predictions0, axis=-1)
    # Merge windows back into single stream
    clean_data0 = DAE_funcs.overlapping_windows_to_list(clean_windows0, OVERLAP)

    # Split data into windows and denoise
    noisy_windows1 = DAE_funcs.list_to_overlapping_windows(filtered_d_n1, WINDOW_WIDTH, OVERLAP)
    predictions1 = denoiser.predict(noisy_windows1, verbose=2)
    clean_windows1 = np.squeeze(predictions1, axis=-1)
    # Merge windows back into single stream
    clean_data1 = DAE_funcs.overlapping_windows_to_list(clean_windows1, OVERLAP)

    # Split data into windows and denoise
    noisy_windows2 = DAE_funcs.list_to_overlapping_windows(filtered_d_n2, WINDOW_WIDTH, OVERLAP)
    predictions2 = denoiser.predict(noisy_windows2, verbose=2)
    clean_windows2 = np.squeeze(predictions2, axis=-1)
    # Merge windows back into single stream
    clean_data2 = DAE_funcs.overlapping_windows_to_list(clean_windows2, OVERLAP)

    # Split data into windows and denoise
    noisy_windows3 = DAE_funcs.list_to_overlapping_windows(filtered_d_n3, WINDOW_WIDTH, OVERLAP)
    predictions3 = denoiser.predict(noisy_windows3, verbose=2)
    clean_windows3 = np.squeeze(predictions3, axis=-1)
    # Merge windows back into single stream
    clean_data3 = DAE_funcs.overlapping_windows_to_list(clean_windows3, OVERLAP)

    # Split data into windows and denoise
    noisy_windows4 = DAE_funcs.list_to_overlapping_windows(filtered_d_n4, WINDOW_WIDTH, OVERLAP)
    predictions4 = denoiser.predict(noisy_windows4, verbose=2)
    clean_windows4 = np.squeeze(predictions4, axis=-1)
    # Merge windows back into single stream
    clean_data4 = DAE_funcs.overlapping_windows_to_list(clean_windows4, OVERLAP)

    # Split data into windows and denoise
    noisy_windows5 = DAE_funcs.list_to_overlapping_windows(filtered_d_n5, WINDOW_WIDTH, OVERLAP)
    predictions5 = denoiser.predict(noisy_windows5, verbose=2)
    clean_windows5 = np.squeeze(predictions5, axis=-1)
    # Merge windows back into single stream
    clean_data5 = DAE_funcs.overlapping_windows_to_list(clean_windows5, OVERLAP)

    # Extract spikes from indexes
    spikes0 = utils.extract_spike_windows(clean_data0, Index)
    spikes1 = utils.extract_spike_windows(clean_data1, Index)
    spikes2 = utils.extract_spike_windows(clean_data2, Index)
    spikes3 = utils.extract_spike_windows(clean_data3, Index)
    spikes4 = utils.extract_spike_windows(clean_data4, Index)
    spikes5 = utils.extract_spike_windows(clean_data5, Index)

    # Create new spikes array that are shifted by random values to increase training robustness
    shifted_spikes_1 = []
    for spike in spikes0:
        shift = randint(-5, 5)
        shifted_spikes_1.append(np.roll(spike, shift))

    shifted_spikes_2 = []
    for spike in spikes0:
        shift = randint(-10, 10)
        shifted_spikes_2.append(np.roll(spike, shift))

    # Add slightly noisy spikes to training data
    shifted_spikes_3 = []
    for spike in spikes0:
        noisy_spike = DAE_funcs.add_white_noise(spike, 0.02)
        shifted_spikes_3.append(noisy_spike)

    plt.plot(shifted_spikes_3[0])
    plt.show()

    # Combine the original and added data
    spikes = np.asarray(list(spikes0) + list(shifted_spikes_1) + list(shifted_spikes_2) + list(shifted_spikes_3) + list(spikes1) + list(spikes2) + list(spikes3) + list(spikes4) + list(spikes5))
    Class = np.asarray(list(Class) + list(Class) + list(Class) + list(Class) + list(Class) + list(Class) + list(Class) + list(Class) + list(Class))

    print(spikes.shape)

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