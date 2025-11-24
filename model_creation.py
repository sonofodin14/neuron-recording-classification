import utils
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Import training data from D1.mat
    d, Index, Class =  utils.load_training_data()

    # Use wavelets to denoise the data
    denoised_d = utils.wavelet_denoising(d)

    # High-pass filter the data
    numtaps = 1501
    fc = 100
    fs = 25000
    filter_coef = utils.create_hp_filter(numtaps, fc, fs)
    filtered_d = utils.filter_data(denoised_d, filter_coef, numtaps)

    # Extract spikes from indexes
    spikes = utils.extract_spike_windows(filtered_d, Index)

    # Split data into training/testing sets
    x_train = np.asarray(spikes[0:int(0.8*len(spikes))])
    x_test = np.asarray(spikes[int(0.8*len(spikes)):-1])

    y_train = np.asarray(Class[0:int(0.8*len(Class))]) - 1
    y_test = np.asarray(Class[int(0.8*len(Class)):-1]) - 1

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # Shuffle the training set
    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]

    num_classes = len(np.unique(y_train))
    model = utils.make_model(input_shape=x_train.shape[1:], num_classes=num_classes)

    # Training the model
    epochs = 500
    batch_size = 6
    history = utils.train_model(model, x_train, y_train, epochs=epochs, batch_size=batch_size)

    # Load and evaluate best model
    model = utils.load_best_model()
    test_loss, test_acc = model.evaluate(x_test, y_test)

    print("Test accuracy", test_acc)
    print("Test loss", test_loss)

    # Plot model's losses
    metric = "sparse_categorical_accuracy"
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.show()
    plt.close()