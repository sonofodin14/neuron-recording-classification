# Third-party imports
import matplotlib.pyplot as plt
import scipy.io as spio
import numpy as np
import keras
from sklearn import preprocessing
from skimage.restoration import denoise_wavelet
import scipy.signal as sig

# Load in D1 training data
mat = spio.loadmat('D1.mat', squeeze_me=True)
d = mat['d']
Index = mat['Index']
sorted_Index = sorted(Index)
Class = mat['Class']

# Normalise data
# scaler = preprocessing.RobustScaler()
# d_shaped = d.reshape(-1,1)
# d_norm = scaler.fit_transform(d_shaped)
# d_to_use = d_norm.flatten()
d_to_use = d

# Setup Filter
numtaps = 101
fl, fu =  300, 3000
filter_coef = sig.firwin(numtaps, fl, pass_zero=False, window='hamming', fs=25000)

# Filter & Shift Signal back
d_filt = sig.lfilter(filter_coef, 1.0, d_to_use)
d_filt = np.roll(d_filt, -(int(numtaps/2)))

# Wavelet denoising
d_denoise = denoise_wavelet(
    d_filt, 
    method='BayesShrink', 
    mode='soft', 
    wavelet_levels=7,
    wavelet='sym8',
    rescale_sigma='True',
    )

# Filter & Shift Signal back
d_filtered = sig.lfilter(filter_coef, 1.0, d_denoise)
d_denoise = np.roll(d_filtered, -(int(numtaps/2)))
d_to_use = d_denoise

# Format: spike[x] = [d data], [index], [class]
spike_data = []

# Extract windows of spikes from index
for i in range(len(Index)):
    spike_data.append([d_to_use[Index[i]:Index[i]+50].tolist(), int(Index[i]), int(Class[i])])

spikes = [item[0] for item in spike_data]
classes = [item[2] for item in spike_data]

# Split data into train/test inputs and outputs
x_train = np.asarray(spikes[0:int(0.8*len(spikes))])
x_test = np.asarray(spikes[int(0.8*len(spikes)):-1])

y_train = np.asarray(classes[0:int(0.8*len(classes))]) - 1
y_test = np.asarray(classes[int(0.8*len(classes)):-1]) - 1

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

num_classes = len(np.unique(y_train))
print("Num Classes: ", num_classes)

# Shuffle training set
idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]

# Initial Model
def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=5, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=5, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    dense_layer = keras.layers.Dense(32, activation="softmax")(gap)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(dense_layer)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


model = make_model(input_shape=x_train.shape[1:])
# keras.utils.plot_model(model, show_shapes=True)

# Training the model
epochs = 500
batch_size = 32

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_model.keras", save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]
model.compile(
    optimizer="adam",
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

# Evaluate model on test data
model = keras.models.load_model("best_model.keras")

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