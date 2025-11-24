import utils

if __name__ == "__main__":
    file_names = ['D2.mat', 'D3.mat', 'D4.mat', 'D5.mat', 'D6.mat']

    for file in file_names:
        # Load test data from file
        file_path = 'TESTING DATA/' + file
        data = utils.load_file_data(file_path)

        # Denoise data using wavelets
        data_denoised = utils.wavelet_denoising(data)

        # High-pass filter data
        numtaps = 1501
        fc = 100
        fs = 25000
        filter_coef = utils.create_hp_filter(numtaps, fc, fs)
        data_filtered = utils.filter_data(data_denoised, filter_coef, numtaps)

        # Find peaks in data using a dynamic, standard deviation based peak detection
        peaks = utils.noise_dependent_peak_detection(data_filtered)

        # Convert peak indexes to spike start indexes
        Index = utils.peaks_to_spike_index(peaks)

        # Gather spike data windows
        spikes = utils.extract_spike_windows(data_filtered, Index)

        # Classify each spike
        Class = utils.classify_spikes(spikes)

        # Save results to .mat file
        utils.save_mat_file(Index, Class, file)