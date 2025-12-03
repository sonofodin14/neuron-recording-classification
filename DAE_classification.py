# Third-Party Imports
import numpy as np

# First-Party Imports
import utils
import DAE_funcs
from DAE_funcs import WINDOW_WIDTH, OVERLAP

if __name__ == "__main__":
    file_names = ['D2.mat', 'D3.mat', 'D4.mat', 'D5.mat', 'D6.mat']

    # Filter Parameters
    numtaps = 1501
    fc = 50
    fs = 25000

    # Load denoising model
    denoiser = utils.load_denoising_model()

    for file in file_names:
        # Load test data from file
        file_path = 'TESTING DATA/' + file
        noisy_data = utils.load_file_data(file_path)

        # High-pass filter data
        filter_coef = utils.create_hp_filter(numtaps, fc, fs)
        filtered_data = utils.filter_data(noisy_data, filter_coef, numtaps)

        # Split data into windows and denoise
        noisy_windows = DAE_funcs.list_to_overlapping_windows(filtered_data, WINDOW_WIDTH, OVERLAP)
        predictions = denoiser.predict(noisy_windows, verbose=2)
        clean_windows = np.squeeze(predictions, axis=-1)
        # Merge windows back into single stream
        clean_data = DAE_funcs.overlapping_windows_to_list(clean_windows, OVERLAP)

        # Scale data to [0,1]
        clean_data_scaled = utils.minmax_scale(clean_data)

        # Find peaks in data using a dynamic, standard deviation based peak detection
        peaks = utils.noise_dependent_peak_detection(clean_data_scaled.flatten())

        # Convert peak indexes to spike start indexes
        Index = utils.peaks_to_spike_index(peaks)

        # Gather spike data windows
        spikes = utils.extract_spike_windows(clean_data_scaled, Index)
        print(spikes.shape)

        # Classify each spike
        Class = utils.classify_spikes(spikes)

        # Save results to .mat file
        utils.save_mat_file(Index, Class, file)