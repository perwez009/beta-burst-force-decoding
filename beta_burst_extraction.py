#This code is the modified version of the code used by Szul et al., 2023.

import numpy as np
from mne import read_epochs_eeglab
from specparam import SpectralModel as Specparam  
from superlet import superlet, scale_from_period
from burst_detection import extract_bursts

# Step 1: Load the epoched EEG data
file_path = r"file_path/file_name.set"
epochs = read_epochs_eeglab(file_path)
sfreq = epochs.info['sfreq']  
times = epochs.times  

# Select a channel for analysis
channel_name = 'C3'     # Or your IC extracted time series
channel_idx = epochs.ch_names.index(channel_name)
raw_trials = epochs.get_data()[:, channel_idx, :]  # Shape: (trials, time)

# Step 2: Time-Frequency Decomposition using Superlet
max_freq = 40  
foi = np.linspace(1, max_freq, 200)  # Frequencies of interest
scales = scale_from_period(1 / foi)

tf_trials = []
for trial in raw_trials:
    tf = superlet(trial, sfreq, scales, order_max=40, order_min=1, c_1=4, adaptive=True)
    tf_trials.append(np.abs(tf))  
tf_trials = np.array(tf_trials, dtype=np.float32)  # trials x freq x time

# Step 3: Define frequency range for beta burst detection
search_range = np.where((foi >= 10) & (foi <= 33))[0]
beta_lims = [13, 30] 
search_freqs = foi[search_range]

# Step 4: Compute aperiodic spectrum using SpectralParameterizer
# Average PSD across trials and time
average_psd = np.average(tf_trials, axis=(0, 2))  # Average over trials and time

# Fit Specparam (replaces FOOOF)
ff = Specparam()
ff.fit(foi, average_psd, [1, max_freq])  
aperiodic_spectrum = 10 ** ff._ap_fit  # Aperiodic fit
aperiodic_spectrum = aperiodic_spectrum[search_range]  # Limit to search range
aperiodic_spectrum = aperiodic_spectrum.reshape(-1, 1)  # Reshape as needed

# Step 5: Detect beta bursts using extract_bursts
bursts = extract_bursts(
    raw_trials,  # trial x time
    tf_trials[:, search_range, :],  # trial x freq x time (limited to search range)
    times,
    search_freqs,
    beta_lims,
    aperiodic_spectrum,
    sfreq,
    w_size=0.26  
)

# Step 6: Output results
print("Detected Bursts Dictionary Keys:", bursts.keys())
print("Number of bursts detected:", len(bursts['peak_time']))

# Step 7: Basic analysis of results
print(f"Average burst frequency: {np.mean(bursts['peak_freq']):.2f} Hz")
print(f"Average burst duration: {np.mean(bursts['fwhm_time']):.3f} seconds")
print(f"Frequency range: {np.min(bursts['peak_freq']):.1f} - {np.max(bursts['peak_freq']):.1f} Hz")

# Step 8: Save the results
np.save('sub_cond.npy', bursts)  
