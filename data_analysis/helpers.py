import mne
import numpy as np
from pathlib import Path

# Configuration
base_path = Path('/home/megan/Downloads/cleaneddata/first_session')
subjects = ['A01T', 'A02T', 'A03T', 'A04T', 'A05T', 'A06T', 'A07T', 'A08T', 'A09T']

event_mapping = {'Left hand': '769', 'Right hand': '770', 'Feet': '771', 'Tongue': '772'}
freq_bands = {'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 40)}


def load_epochs( subject_file):
    """Load epochs for a subject."""
    data = mne.io.read_raw_fif(base_path / subject_file, preload=True, verbose=False)
    events, event_dict = mne.events_from_annotations(data)

    event_id = {label: event_dict[code] for label, code in event_mapping.items() if code in event_dict}

    epochs = mne.Epochs(data, events, event_id=event_id, tmin=-1, tmax=4,
                        baseline=(-1, 0), preload=True, picks='eeg', verbose=False)
    return epochs

def find_mi_window(trial_data, sfreq, baseline_end_idx):
    """
    Find the active MI period in a trial by detecting when power changes from baseline.
    Returns start and end indices of the MI window.
    """
    # Compute power (squared signal)
    power = trial_data ** 2

    # Get baseline power (before t=0)
    baseline_power = np.mean(power[:baseline_end_idx])
    baseline_std = np.std(power[:baseline_end_idx])

    # Find regions where power exceeds baseline + 2*std
    threshold = baseline_power + 2 * baseline_std
    active = power > threshold

    # Find the first sustained active period (at least 500ms)
    min_duration = int(0.5 * sfreq)
    active_start = None

    for i in range(baseline_end_idx, len(active) - min_duration):
        if np.sum(active[i:i+min_duration]) > 0.7 * min_duration:
            active_start = i
            break

    if active_start is None:
        # If no clear onset, use fixed window after cue
        active_start = baseline_end_idx + int(0.5 * sfreq)

    # MI window: 1.5 seconds from detected start
    active_end = min(active_start + int(1.5 * sfreq), len(trial_data))

    return active_start, active_end

def align_trials_to_mi_onset(data, sfreq, baseline_end_idx):
    """
    Align all trials to their detected MI onset.
    Returns aligned data.
    """
    n_trials, n_times = data.shape
    aligned_trials = []

    # Find MI windows for all trials
    windows = []
    for trial in data:
        start, end = find_mi_window(trial, sfreq, baseline_end_idx)
        windows.append((start, end))

    # Find common window length (use median)
    window_lengths = [end - start for start, end in windows]
    target_length = int(np.median(window_lengths))

    # Extract aligned windows
    for trial, (start, end) in zip(data, windows):
        if end - start >= target_length:
            aligned_trials.append(trial[start:start+target_length])
        else:
            # Pad if needed
            segment = trial[start:end]
            padded = np.pad(segment, (0, target_length - len(segment)), mode='edge')
            aligned_trials.append(padded)

    return np.array(aligned_trials)