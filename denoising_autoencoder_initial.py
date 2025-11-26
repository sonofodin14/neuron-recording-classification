# Third-Party Imports
import matplotlib.pyplot as plt
import scipy.io as spio
import numpy as np
import scipy.signal as sig
from random import randint
import keras

# First-Party Imports
import utils

training_data = utils.load_training_data()

# METHOD:
# 1. Create "perfect", no noise training data (zero everywhere apart from spikes)
# 2. Window data in larger windows with overlap
# 3. Artificially inject noise into the clean data
# 4. Train denoising autoencoder (input is noisy, label is clean)



