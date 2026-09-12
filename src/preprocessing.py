import mne
import numpy as np
import warnings

warnings.filterwarnings("ignore")

FIXED_LENGTH = 3000

def preprocess(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    # Pick EEG channels
    raw.pick("eeg")

    # Bandpass filter
    raw.filter(0.5, 30, verbose=False)

    # Create 30-second epochs
    events = mne.make_fixed_length_events(raw, duration=30)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=30, baseline=None, verbose=False)

    data = epochs.get_data()

    # Use first channel
    data = data[:, 0, :]

    processed = []

    for epoch in data:
        if len(epoch) > FIXED_LENGTH:
            epoch = epoch[:FIXED_LENGTH]
        else:
            epoch = np.pad(epoch, (0, FIXED_LENGTH - len(epoch)))

        processed.append(epoch)

    data = np.array(processed)

    # Normalize
    data = (data - np.mean(data)) / np.std(data)

    return data