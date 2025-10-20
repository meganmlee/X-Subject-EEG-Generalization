import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import helpers
from pathlib import Path

# Across Trials: Is the neural representation of a label consistent across trials?

# Output results
output_dir = Path('/home/megan/Downloads/cleaneddata/cross_trial')
output_dir.mkdir(exist_ok=True)

def compute_channel_consistency_temporal(epochs, condition):
    """
    Compute trial consistency for each channel separately (temporal domain).
    Uses auto-alignment + cross-correlation to handle timing drift.

    Returns: array of consistency scores, one per channel
    """
    data = epochs[condition].get_data()  # (n_trials, n_channels, n_times)
    n_channels = data.shape[1]
    sfreq = epochs.info['sfreq']

    # Baseline ends at t=0, which is 1 second into the epoch (tmin=-1)
    baseline_end_idx = int(1.0 * sfreq)

    # Allow 200ms fine-tuning after auto-alignment
    max_shift_samples = int(0.2 * sfreq)

    channel_consistencies = []

    for ch_idx in range(n_channels):
        # Get data for this channel across all trials: (n_trials, n_times)
        channel_data = data[:, ch_idx, :]

        # Auto-align trials to MI onset
        aligned_data = helpers.align_trials_to_mi_onset(channel_data, sfreq, baseline_end_idx)

        # Now use cross-correlation for fine-tuning
        n_trials = aligned_data.shape[0]
        correlations = []

        for i in range(n_trials):
            for j in range(i+1, n_trials):
                trial1 = aligned_data[i]
                trial2 = aligned_data[j]

                # Normalize
                trial1 = (trial1 - np.mean(trial1)) / (np.std(trial1) + 1e-10)
                trial2 = (trial2 - np.mean(trial2)) / (np.std(trial2) + 1e-10)

                # Cross-correlation
                xcorr = np.correlate(trial1, trial2, mode='same')
                xcorr = xcorr / len(trial1)

                # Find max correlation within allowed shift window
                center = len(xcorr) // 2
                start = max(0, center - max_shift_samples)
                end = min(len(xcorr), center + max_shift_samples)
                max_corr = np.max(xcorr[start:end])

                correlations.append(max_corr)

        # Average correlation for this channel
        channel_consistencies.append(np.mean(correlations))

    return np.array(channel_consistencies)

def compute_channel_consistency_frequency(epochs, condition, fmin, fmax):
    """
    Compute trial consistency for each channel separately (frequency domain).

    Returns: array of consistency scores, one per channel
    """
    spectrum = epochs[condition].compute_psd(method='welch', fmin=fmin, fmax=fmax,
                                             n_fft=512, verbose=False)
    psds = spectrum.get_data()  # (n_trials, n_channels, n_freqs)
    n_channels = psds.shape[1]

    channel_consistencies = []

    for ch_idx in range(n_channels):
        # Get PSD for this channel across all trials: (n_trials, n_freqs)
        channel_psd = psds[:, ch_idx, :]

        # Compute pairwise correlations between trials for this channel
        corr_matrix = np.corrcoef(channel_psd)

        # Extract upper triangle
        triu_idx = np.triu_indices_from(corr_matrix, k=1)
        pairwise_corrs = corr_matrix[triu_idx]

        channel_consistencies.append(np.mean(pairwise_corrs))

    return np.array(channel_consistencies)

def analyze_subject(subject_file):
    """Analyze per-channel trial consistency for one subject."""
    print(f"\nAnalyzing {subject_file}...")
    epochs = helpers.load_epochs(subject_file)
    subject_id = subject_file.replace('.fif', '')
    channel_names = epochs.ch_names

    results = {
        'subject': subject_id,
        'channel_names': channel_names,
        'conditions': {}
    }

    for condition in epochs.event_id.keys():
        n_trials = len(epochs[condition])
        if n_trials < 2:
            continue

        # Temporal consistency per channel (WITH AUTO-ALIGNMENT)
        temp_consistency = compute_channel_consistency_temporal(epochs, condition)

        # Frequency consistency per channel per band
        freq_consistency = {}
        for band_name, (fmin, fmax) in helpers.freq_bands.items():
            freq_consistency[band_name] = compute_channel_consistency_frequency(
                epochs, condition, fmin, fmax
            )

        results['conditions'][condition] = {
            'n_trials': n_trials,
            'temporal': temp_consistency,
            'frequency': freq_consistency
        }

    return results

def plot_subject_channel_consistency(results, save_path):
    """Create visualization showing per-channel consistency for one subject."""
    subject_id = results['subject']
    channel_names = results['channel_names']
    conditions = list(results['conditions'].keys())

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

    # Plot 1-4: Temporal consistency per channel for each condition (2x2 grid)
    subplot_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for idx, condition in enumerate(conditions[:4]):
        row, col = subplot_positions[idx]
        ax = fig.add_subplot(gs[row, col])

        temp_scores = results['conditions'][condition]['temporal']

        # Create bar plot
        x = np.arange(len(channel_names))
        colors = ['red' if score < 0.5 else 'orange' if score < 0.7 else 'green'
                  for score in temp_scores]
        bars = ax.bar(x, temp_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

        ax.axhline(0.7, color='green', linestyle='--', alpha=0.5, linewidth=2, label='High (>0.7)')
        ax.axhline(0.5, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='Moderate (>0.5)')
        ax.set_xticks(x[::2])
        ax.set_xticklabels([ch.replace('EEG-', '') for ch in channel_names[::2]],
                          rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Mean Trial Correlation', fontsize=10)
        ax.set_title(f'{condition} - Temporal Consistency per Channel', fontweight='bold', fontsize=11)
        ax.set_ylim([0, 1])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        # Highlight motor channels
        motor_indices = [i for i, ch in enumerate(channel_names)
                        if any(m in ch for m in ['C3', 'Cz', 'C4'])]
        for mi in motor_indices:
            ax.axvspan(mi-0.5, mi+0.5, alpha=0.2, color='blue')

    # Plot 5: Frequency band comparison for motor channels
    ax = fig.add_subplot(gs[2, :])
    motor_channels = ['EEG-C3', 'EEG-Cz', 'EEG-C4']
    motor_indices = [channel_names.index(ch) for ch in motor_channels if ch in channel_names]

    band_names = list(helpers.freq_bands.keys())
    x = np.arange(len(band_names))
    width = 0.2

    for cond_idx, condition in enumerate(conditions):
        band_scores = []
        for band_name in band_names:
            freq_scores = results['conditions'][condition]['frequency'][band_name]
            motor_scores = freq_scores[motor_indices]
            band_scores.append(np.mean(motor_scores))

        ax.bar(x + cond_idx*width, band_scores, width, label=condition, alpha=0.7)

    ax.set_xlabel('Frequency Band', fontsize=10)
    ax.set_ylabel('Mean Trial Correlation', fontsize=10)
    ax.set_title('Motor Channels Frequency Consistency\n(C3, Cz, C4 averaged)',
                 fontweight='bold', fontsize=11)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(band_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])

    # Plot 6: Heatmap of channel consistency across conditions
    ax = fig.add_subplot(gs[3, :])
    heatmap_data = []
    for condition in conditions:
        heatmap_data.append(results['conditions'][condition]['temporal'])
    heatmap_data = np.array(heatmap_data)

    im = ax.imshow(heatmap_data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(conditions)
    ax.set_xticks(range(len(channel_names)))
    ax.set_xticklabels([ch.replace('EEG-', '') for ch in channel_names],
                       rotation=90, fontsize=8)
    ax.set_xlabel('Channel', fontsize=10)
    ax.set_ylabel('Condition', fontsize=10)
    ax.set_title('Temporal Consistency Heatmap (All Channels)', fontweight='bold', fontsize=11)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Mean Trial Correlation', fontsize=10)

    for mi in motor_indices:
        ax.axvline(mi, color='blue', linestyle='--', alpha=0.5, linewidth=2)

    plt.suptitle(f'Per-Channel Consistency Analysis - {subject_id} (Auto-Aligned)',
                 fontsize=16, fontweight='bold')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {save_path}")

def create_summary_dataframe(all_results):
    """Create summary statistics across all subjects."""
    summary_data = []

    for result in all_results:
        subject = result['subject']
        channel_names = result['channel_names']

        for condition, cond_data in result['conditions'].items():
            # Overall metrics
            temp_consistency = cond_data['temporal']

            # Motor channel metrics
            motor_channels = ['EEG-C3', 'EEG-Cz', 'EEG-C4']
            motor_indices = [i for i, ch in enumerate(channel_names) if ch in motor_channels]
            motor_temp = temp_consistency[motor_indices] if motor_indices else temp_consistency

            row = {
                'Subject': subject,
                'Condition': condition,
                'N_Trials': cond_data['n_trials'],
                'Overall_Temporal_Mean': np.mean(temp_consistency),
                'Overall_Temporal_Std': np.std(temp_consistency),
                'Motor_Temporal_Mean': np.mean(motor_temp),
                'Motor_Temporal_Std': np.std(motor_temp),
            }

            # Add frequency band metrics for motor channels
            for band_name in helpers.freq_bands.keys():
                freq_scores = cond_data['frequency'][band_name]
                motor_freq = freq_scores[motor_indices] if motor_indices else freq_scores
                row[f'Motor_{band_name}_Mean'] = np.mean(motor_freq)

            summary_data.append(row)

    return pd.DataFrame(summary_data)

def plot_cross_subject_summary(df, save_path):
    """Create cross-subject summary plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    conditions = df['Condition'].unique()

    # Plot 1: Overall temporal consistency by subject
    ax = axes[0, 0]
    for condition in conditions:
        data = df[df['Condition'] == condition]
        ax.plot(data['Subject'], data['Overall_Temporal_Mean'],
                marker='o', label=condition, linewidth=2, markersize=8)
    ax.axhline(0.7, color='green', linestyle='--', alpha=0.3)
    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.3)
    ax.set_xlabel('Subject')
    ax.set_ylabel('Mean Consistency (All Channels)')
    ax.set_title('Overall Temporal Consistency', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 2: Motor channel temporal consistency by subject
    ax = axes[0, 1]
    for condition in conditions:
        data = df[df['Condition'] == condition]
        ax.plot(data['Subject'], data['Motor_Temporal_Mean'],
                marker='s', label=condition, linewidth=2, markersize=8)
    ax.axhline(0.7, color='green', linestyle='--', alpha=0.3)
    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.3)
    ax.set_xlabel('Subject')
    ax.set_ylabel('Mean Consistency (Motor Channels)')
    ax.set_title('Motor Channels (C3, Cz, C4) Consistency', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 3: Frequency band consistency (motor channels)
    ax = axes[0, 2]
    band_cols = [col for col in df.columns if 'Motor_' in col and '_Mean' in col
                 and col != 'Motor_Temporal_Mean']
    band_data = df.groupby('Condition')[band_cols].mean()
    band_data.columns = [col.replace('Motor_', '').replace('_Mean', '')
                         for col in band_data.columns]

    im = ax.imshow(band_data.values, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(band_data.columns)))
    ax.set_xticklabels(band_data.columns)
    ax.set_yticks(range(len(band_data.index)))
    ax.set_yticklabels(band_data.index)
    ax.set_title('Motor Channels Frequency Consistency', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Mean Correlation')

    for i in range(len(band_data.index)):
        for j in range(len(band_data.columns)):
            ax.text(j, i, f'{band_data.values[i, j]:.2f}',
                   ha="center", va="center", color="black", fontsize=9)

    # Plot 4: Distribution comparison
    ax = axes[1, 0]
    overall_data = [df[df['Condition'] == c]['Overall_Temporal_Mean'].values for c in conditions]
    bp = ax.boxplot(overall_data, labels=conditions, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.axhline(0.7, color='green', linestyle='--', alpha=0.3)
    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.3)
    ax.set_ylabel('Consistency')
    ax.set_title('Overall Distribution by Condition', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 5: Motor channel distribution
    ax = axes[1, 1]
    motor_data = [df[df['Condition'] == c]['Motor_Temporal_Mean'].values for c in conditions]
    bp = ax.boxplot(motor_data, labels=conditions, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightcoral')
    ax.axhline(0.7, color='green', linestyle='--', alpha=0.3)
    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.3)
    ax.set_ylabel('Consistency')
    ax.set_title('Motor Channels Distribution', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 6: Subject ranking
    ax = axes[1, 2]
    subject_avg = df.groupby('Subject').agg({
        'Overall_Temporal_Mean': 'mean',
        'Motor_Temporal_Mean': 'mean'
    }).reset_index()

    x = np.arange(len(subject_avg))
    width = 0.35
    ax.bar(x - width/2, subject_avg['Overall_Temporal_Mean'], width,
           label='All Channels', alpha=0.7)
    ax.bar(x + width/2, subject_avg['Motor_Temporal_Mean'], width,
           label='Motor Channels', alpha=0.7)
    ax.axhline(0.7, color='green', linestyle='--', alpha=0.3)
    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.3)
    ax.set_xlabel('Subject')
    ax.set_ylabel('Mean Consistency')
    ax.set_title('Average Consistency by Subject', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(subject_avg['Subject'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Cross-Subject Per-Channel Consistency Summary (Auto-Aligned)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved cross-subject summary plot")

def print_report(df):
    """Print detailed summary report."""
    print("\n" + "="*80)
    print("PER-CHANNEL CONSISTENCY ANALYSIS SUMMARY (WITH AUTO-ALIGNMENT)")
    print("="*80)
    print(f"\nTotal subjects: {df['Subject'].nunique()}")
    print(f"Total trials: {df['N_Trials'].sum()}")

    print("\n" + "-"*80)
    print("OVERALL CONSISTENCY (All Channels):")
    print(f"  Mean: {df['Overall_Temporal_Mean'].mean():.3f} ± {df['Overall_Temporal_Mean'].std():.3f}")
    print(f"  Range: [{df['Overall_Temporal_Mean'].min():.3f}, {df['Overall_Temporal_Mean'].max():.3f}]")

    print("\n" + "-"*80)
    print("PER-CONDITION BREAKDOWN:")
    for condition in df['Condition'].unique():
        cond_data = df[df['Condition'] == condition]
        print(f"\n  {condition}:")
        print(f"    Overall channels: {cond_data['Overall_Temporal_Mean'].mean():.3f} ± {cond_data['Overall_Temporal_Mean'].std():.3f}")

        # Best frequency band for motor channels
        band_means = {band: cond_data[f'Motor_{band}_Mean'].mean()
                     for band in helpers.freq_bands.keys()}
        best_band = max(band_means, key=band_means.get)
        print(f"    Best freq band (motor): {best_band} ({band_means[best_band]:.3f})")

# Main execution
if __name__ == "__main__":

    all_results = []

    for subject in helpers.subjects:
        try:
            result = analyze_subject(f'{subject}.fif')
            all_results.append(result)

            # Plot individual subject
            plot_path = output_dir / f'{subject}_channel_consistency.png'
            plot_subject_channel_consistency(result, plot_path)

        except Exception as e:
            print(f"Error with {subject}: {e}")

    # Create summary
    df = create_summary_dataframe(all_results)
    df.to_csv(output_dir / 'channel_consistency_summary.csv', index=False)
    print("Saved CSV summary")

    # Create summary plots
    summary_plot_path = output_dir / 'cross_subject_channel_summary.png'
    plot_cross_subject_summary(df, summary_plot_path)

    # Print report
    print_report(df)