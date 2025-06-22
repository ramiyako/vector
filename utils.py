"""
Signal processing utilities for packet extraction and vector generation
Utility functions for signal processing, spectrogram creation, and packet manipulation
"""

import os
import numpy as np
import scipy.io as sio
import scipy.signal
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def get_sample_rate_from_mat(file_path):
    """Extract sample rate from MAT file metadata"""
    try:
        # Try to extract from filename pattern
        filename = os.path.basename(file_path)
        if 'MHz' in filename:
            # Look for pattern like "56MHz" or "56_MHz"
            import re
            match = re.search(r'(\d+(?:\.\d+)?)[_\s]*MHz', filename, re.IGNORECASE)
            if match:
                return float(match.group(1)) * 1e6
        
        # Try to load additional metadata if available
        data = sio.loadmat(file_path)
        if 'sample_rate' in data:
            return float(data['sample_rate'])
        elif 'fs' in data:
            return float(data['fs'])
        elif 'sr' in data:
            return float(data['sr'])
            
        # Default fallback
        print(f"Warning: Could not determine sample rate from {file_path}, using default 56MHz")
        return 56e6
        
    except Exception as e:
        print(f"Error reading sample rate from {file_path}: {e}")
        return 56e6

def load_packet(file_path):
    """Load packet data from MAT file"""
    try:
        data = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
        
        if 'Y' in data:
            packet = data['Y']
        else:
            # Find the first non-metadata key
            candidates = [k for k in data.keys() if not k.startswith('__')]
            if len(candidates) == 1:
                packet = data[candidates[0]]
            else:
                raise ValueError(f"Ambiguous packet data in {file_path}. Available keys: {list(data.keys())}")
        
        # Ensure packet is 1D
        if packet.ndim > 1:
            packet = packet.flatten()
            
        return packet.astype(np.complex64)
        
    except Exception as e:
        print(f"Error loading packet from {file_path}: {e}")
        raise

def resample_signal(signal, orig_sr, target_sr):
    """Resample signal to target sample rate"""
    if orig_sr == target_sr:
        return signal
    
    # Calculate resampling ratio
    ratio = target_sr / orig_sr
    new_length = int(len(signal) * ratio)
    
    # Use scipy's resample for better quality
    resampled = scipy.signal.resample(signal, new_length)
    return resampled.astype(np.complex64)

def apply_frequency_shift(signal, freq_shift, sample_rate):
    """Apply frequency shift to signal using complex exponential multiplication"""
    if freq_shift == 0:
        return signal
        
    t = np.arange(len(signal)) / sample_rate
    shift_factor = np.exp(2j * np.pi * freq_shift * t)
    return (signal * shift_factor).astype(np.complex64)

def compute_freq_ranges(shifts, margin=1e6):
    """
    Compute frequency ranges for plotting based on frequency shifts
    
    Args:
        shifts: List of frequency shifts in Hz
        margin: Additional margin around the frequency range in Hz
    
    Returns:
        List of frequency ranges or None if no shifts
    """
    if not shifts or all(s == 0 for s in shifts):
        return None
        
    # Remove any non-numeric values
    valid_shifts = []
    for s in shifts:
        if isinstance(s, (int, float)) and not isinstance(s, bool):
            valid_shifts.append(s)
    
    if not valid_shifts:
        return None
    
    min_shift = min(valid_shifts)
    max_shift = max(valid_shifts)
    
    # Add margin
    freq_min = min_shift - margin
    freq_max = max_shift + margin
    
    return [(freq_min, freq_max)]

def create_spectrogram(sig, sr, center_freq=0, max_samples=1_000_000):
    """
    Create spectrogram with optimized parameters for performance
    
    Args:
        sig: Complex signal
        sr: Sample rate in Hz
        center_freq: Center frequency for display (Hz)
        max_samples: Maximum samples to process for performance
    
    Returns:
        f: Frequency array
        t: Time array  
        Sxx: Spectrogram matrix
    """
    if len(sig) == 0:
        raise ValueError("Signal is empty")

    # Limit signal length for performance
    if len(sig) > max_samples:
        step = len(sig) // max_samples
        sig = sig[::step]
        print(f"Downsampled signal from {len(sig)*step} to {len(sig)} samples for performance")
    
    # Optimized STFT parameters
    # Use a larger window for improved frequency resolution while still limiting
    # runtime on long inputs. For short signals use the full length.
    nperseg = min(4096, len(sig))
    noverlap = nperseg // 2
    nfft = max(4096, 2 ** int(np.ceil(np.log2(nperseg))))
    
    f, t, Sxx = scipy.signal.spectrogram(
        sig, 
        fs=sr, 
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        return_onesided=False,
        scaling='density'
    )
    
    # Shift frequencies relative to center frequency
    f = f + center_freq
    
    return f, t, Sxx

def normalize_spectrogram(Sxx, low_percentile=5, high_percentile=99, max_dynamic_range=60):
    """Normalize spectrogram for better visualization"""
    Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-12)
    
    vmin = np.percentile(Sxx_db, low_percentile)
    vmax = np.percentile(Sxx_db, high_percentile)
    
    # Limit dynamic range
    if vmax - vmin > max_dynamic_range:
        vmin = vmax - max_dynamic_range
    
    return Sxx_db, vmin, vmax

def plot_spectrogram(
    f,
    t,
    Sxx,
    center_freq=0,
    title='Spectrogram',
    packet_start=None,
    sample_rate=None,
    signal=None,
    packet_markers=None,
    freq_ranges=None,
    show_colorbar=True,
):
    """
    Plot spectrogram with optional packet markers and signal overlay
    """
    Sxx_db, vmin, vmax = normalize_spectrogram(Sxx)
    
    # Convert time to milliseconds for better readability
    t_ms = t * 1000
    
    if signal is not None and not freq_ranges:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])
        ax2 = ax2
    else:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        if freq_ranges:
            # Handle multiple frequency ranges if needed
            ax1.axs = [ax1]  # For colorbar compatibility
            ax2 = None
        else:
            ax2 = None
        im = ax1.pcolormesh(
            t_ms,
            f / 1e6,  # Convert to MHz
            Sxx_db,
            shading='nearest',
            cmap='viridis',
            norm=plt.Normalize(vmin=vmin, vmax=vmax),
        )
        ax1.set_ylim(f.min() / 1e6, f.max() / 1e6)
    
    if freq_ranges:
        # Plot multiple frequency ranges
        from matplotlib.colors import Normalize
        for i, (freq_min, freq_max) in enumerate(freq_ranges):
            mask = (f >= freq_min) & (f <= freq_max)
            f_range = f[mask]
            Sxx_range = Sxx_db[mask, :]
            
            if len(f_range) == 0:
                continue
                
            if i == 0:
                ax1 = plt.subplot(len(freq_ranges), 1, i + 1)
            else:
                ax1 = plt.subplot(len(freq_ranges), 1, i + 1, sharex=ax1)
            
            im = ax1.pcolormesh(
                t_ms,
                f_range / 1e6,
                Sxx_range,
                shading='nearest',
                cmap='viridis',
                norm=Normalize(vmin=vmin, vmax=vmax),
            )
            ax1.set_ylim(f_range.min() / 1e6, f_range.max() / 1e6)
            ax1.set_ylabel(f'Freq [MHz]\nRange {i+1}')
            
            if i == len(freq_ranges) - 1:
                ax1.set_xlabel('Time [ms]')
            else:
                ax1.set_xticklabels([])
            ax2 = None
        # After plotting the subranges no additional full-range plot is
        # required. Keep the last axes as the active one for colorbar usage.

    ax1.set_title(title)
    ax1.set_xlabel('Time [ms]')
    ax1.set_ylabel('Frequency [MHz]')
    ax1.grid(True)
    marker_styles = ['x', 'o', '^', 's', 'D', 'P', 'v', '1', '2', '3', '4']
    if packet_markers:
        seen_labels = {}
        for idx, marker in enumerate(packet_markers):
            if len(marker) >= 5:
                tm, freq, label, style, color = marker[:5]
            elif len(marker) == 4:
                tm, freq, label, style = marker
                color = f"C{idx % 10}"
            elif len(marker) == 3:
                tm, freq, label = marker
                style = marker_styles[idx % len(marker_styles)]
                color = f"C{idx % 10}"
            else:
                tm, freq = marker[:2]
                label = None
                style = marker_styles[idx % len(marker_styles)]
                color = f"C{idx % 10}"
            if label not in seen_labels:
                show_label = label
                seen_labels[label] = (style, color)
            else:
                show_label = "_nolegend_"
                style, color = seen_labels[label]
            ax1.plot(tm * 1000, freq / 1e6, linestyle='None', marker=style, color=color, label=show_label)
    if packet_start is not None and sample_rate is not None:
        packet_time = packet_start / sample_rate * 1000  # Convert to ms
        ax1.axvline(x=packet_time, color='r', linestyle='--', label='Packet Start')
    if show_colorbar:
        if freq_ranges:
            # When multiple frequency ranges are plotted, the figure contains
            # several axes. Attach a single colorbar covering all of them.
            plt.colorbar(im, ax=fig.axes, label='Power [dB]')
        else:
            plt.colorbar(im, ax=ax1, label='Power [dB]')

    if signal is not None and not freq_ranges:
        # Plot signal in time domain
        sample_times_ms = np.arange(len(signal)) / sample_rate * 1000
        ax2.plot(sample_times_ms, np.abs(signal))
        if packet_start is not None:
            packet_time_ms = packet_start / sample_rate * 1000
            ax2.axvline(x=packet_time_ms, color='r', linestyle='--', label='Packet Start')
        ax2.set_title('Signal with Packet Start Marker')
        ax2.set_xlabel('Time [ms]')
        ax2.set_ylabel('Amplitude')
        ax2.legend()
        ax2.grid(True)

    if packet_markers:
        ax1.legend()

    plt.tight_layout()
    plt.show()

def save_vector(vector, output_path):
    """Save vector as MAT file"""
    sio.savemat(output_path, {'Y': vector})

def save_vector_wv(vector, output_path, sample_rate, normalize=False):
    """Save vector as WV file using existing script"""
    from vector_analyzer.mat_to_wv_converter import mat2wv

    # mat2wv can accept numpy array directly
    mat2wv(vector, output_path, sample_rate, bNormalize=normalize)

def generate_sample_packet(duration, sr, frequency, amplitude=1.0):
    """Generate sample packet for testing"""
    # np.linspace includes endpoint by default, which causes one extra sample
    # beyond the expected number (sr * duration). Using endpoint=False ensures
    # the number of samples is exactly sr * duration and spacing is 1/sr.
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = amplitude * np.exp(2j * np.pi * frequency * t)
    return signal

def create_sample_packets():
    """Create 6 sample packets for testing"""
    sr = 44100  # Sample rate
    duration = 1.0  # Duration of each packet in seconds
    
    # Create 6 packets with different frequencies
    frequencies = [440, 880, 1320, 1760, 2200, 2640]  # Hz
    packets = []
    
    for i, freq in enumerate(frequencies):
        packet = generate_sample_packet(duration, sr, freq)
        packets.append(packet)
        
        # Save packet
        sio.savemat(f'data/packet_{i+1}.mat', {'Y': packet})
    
    return packets 

def find_packet_start(signal, template=None, threshold_ratio=0.2, window_size=None):
    """Find packet start in signal.

    If template is provided, correlation is performed and the best matching position is calculated.
    Otherwise, a smoothed energy envelope is built and the packet start position is determined.
    threshold_ratio defines the relative intensity (between noise level and maximum) above which
    the packet is considered to have started.
    """

    if template is not None:
        correlation = np.correlate(np.abs(signal), np.abs(template), mode="valid")
        return int(np.argmax(correlation))

    # Energy envelope with smoothing
    energy = np.abs(signal) ** 2
    if window_size is None:
        window_size = max(1, int(0.02 * len(signal)))
    window = np.ones(max(1, window_size)) / max(1, window_size)
    smoothed = np.convolve(energy, window, mode="same")

    noise_level = np.median(smoothed[: len(smoothed) // 10])
    max_energy = np.max(smoothed)
    threshold = noise_level + threshold_ratio * (max_energy - noise_level)

    indices = np.where(smoothed >= threshold)[0]
    return int(indices[0]) if len(indices) > 0 else 0

def measure_packet_timing(signal, template=None):
    """
    Measure intervals before and after packet
    
    Args:
        signal: Full signal
        template: Search template
    
    Returns:
        pre_samples: Number of samples before packet
        post_samples: Number of samples after packet
        packet_start: Packet start index
    """
    packet_start = find_packet_start(signal, template)
    
    # Calculate intervals
    pre_samples = packet_start
    post_samples = len(signal) - packet_start - len(template) if template is not None else 0
    
    return pre_samples, post_samples, packet_start

def plot_packet_with_markers(signal, packet_start, template=None, title='Packet Analysis'):
    """
    Display packet with real packet start marker
    """
    plt.figure(figsize=(12, 6))
    plt.plot(np.abs(signal), label='Signal')
    plt.axvline(x=packet_start, color='r', linestyle='--', label='Packet Start')
    if template is not None:
        plt.plot(packet_start + np.arange(len(template)), np.abs(template), 'g--', label='Template')
    plt.title(title)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

def adjust_packet_start_gui(signal, sample_rate, packet_start):
    f, t, Sxx = create_spectrogram(signal, sample_rate)
    Sxx_db, vmin, vmax = normalize_spectrogram(Sxx)

    # Reverse frequency axis and matrix to descending order (positive left, negative right)
    if f[0] < f[-1]:
        f = f[::-1]
        Sxx_db = Sxx_db[::-1, :]

    # Convert time axis to milliseconds
    t_ms = t * 1000
    sample_indices = np.arange(len(signal))
    sample_times_ms = sample_indices / sample_rate * 1000

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])

    pcm = ax1.pcolormesh(t_ms, f/1e6, Sxx_db, shading='nearest', cmap='viridis', vmin=vmin, vmax=vmax)
    ax1.set_ylim(ax1.get_ylim()[::-1])
    ax1.set_title('Spectrogram - drag the red line to adjust start')
    ax1.set_xlabel('Time [ms]')
    ax1.set_ylabel('Frequency [MHz]')
    ax1.grid(True)
    # Set ticks every 0.5ms
    min_t, max_t = t_ms[0], t_ms[-1]
    ax1.set_xticks(np.arange(np.floor(min_t*2)/2, np.ceil(max_t*2)/2 + 0.5, 0.5))

    ax2.plot(sample_times_ms, np.abs(signal))
    ax2.set_xlabel('Time [ms]')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)
    ax2.set_xticks(np.arange(np.floor(sample_times_ms[0]*2)/2, np.ceil(sample_times_ms[-1]*2)/2 + 0.5, 0.5))

    # Dashed line and label
    packet_time_ms = packet_start / sample_rate * 1000
    line1 = ax1.axvline(packet_time_ms, color='r', linestyle='--')
    line2 = ax2.axvline(packet_time_ms, color='r', linestyle='--')
    # Horizontal annotation
    label = ax1.annotate(f"{packet_time_ms*1000:.0f} μs", xy=(packet_time_ms, ax1.get_ylim()[1]),
                        xytext=(0, 5), textcoords='offset points', ha='center', va='bottom', color='red', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    state = {'drag': False, 'value': packet_start}

    def _press(event):
        if event.inaxes not in (ax1, ax2):
            return
        x_ms = event.xdata
        x_sample = int(x_ms * sample_rate / 1000) if event.inaxes is ax1 else int(event.xdata * sample_rate / 1000)
        if abs(x_sample - state['value']) < sample_rate * 0.01:
            state['drag'] = True

    def _move(event):
        if not state['drag'] or event.inaxes not in (ax1, ax2):
            return
        x_ms = event.xdata
        x_sample = int(x_ms * sample_rate / 1000) if event.inaxes is ax1 else int(event.xdata * sample_rate / 1000)
        x_sample = max(0, min(len(signal)-1, x_sample))
        state['value'] = x_sample
        packet_time_ms = x_sample / sample_rate * 1000
        line1.set_xdata([packet_time_ms, packet_time_ms])
        line2.set_xdata([packet_time_ms, packet_time_ms])
        # Update label
        label.set_position((packet_time_ms, ax1.get_ylim()[1]))
        label.set_text(f"{packet_time_ms*1000:.0f} μs")
        fig.canvas.draw_idle()

    def _release(event):
        state['drag'] = False

    cid_press = fig.canvas.mpl_connect('button_press_event', _press)
    cid_move = fig.canvas.mpl_connect('motion_notify_event', _move)
    cid_rel = fig.canvas.mpl_connect('button_release_event', _release)

    plt.tight_layout()
    plt.show()

    fig.canvas.mpl_disconnect(cid_press)
    fig.canvas.mpl_disconnect(cid_move)
    fig.canvas.mpl_disconnect(cid_rel)

    return state['value']

def adjust_packet_bounds_gui(signal, sample_rate, start_sample=0, end_sample=None):
    if end_sample is None:
        end_sample = len(signal)

    f, t, Sxx = create_spectrogram(signal, sample_rate)
    Sxx_db, vmin, vmax = normalize_spectrogram(Sxx)

    if f[0] < f[-1]:
        f = f[::-1]
        Sxx_db = Sxx_db[::-1, :]

    t_ms = t * 1000
    sample_indices = np.arange(len(signal))
    sample_times_ms = sample_indices / sample_rate * 1000

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])
    plt.subplots_adjust(bottom=0.18)  # Space for buttons

    pcm = ax1.pcolormesh(t_ms, f / 1e6, Sxx_db, shading='nearest', cmap='viridis', vmin=vmin, vmax=vmax)
    ax1.set_ylim(ax1.get_ylim()[::-1])
    ax1.set_title("Use 'g'/'r' to select a line, drag to move, Enter to finish")
    ax1.set_xlabel('Time [ms]')
    ax1.set_ylabel('Frequency [MHz]')
    ax1.grid(True)
    min_t, max_t = t_ms[0], t_ms[-1]
    ax1.set_xticks(np.arange(np.floor(min_t*2)/2, np.ceil(max_t*2)/2 + 0.5, 0.5))

    ax2.plot(sample_times_ms, np.abs(signal))
    ax2.set_xlabel('Time [ms]')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)
    ax2.set_xticks(np.arange(np.floor(sample_times_ms[0]*2)/2, np.ceil(sample_times_ms[-1]*2)/2 + 0.5, 0.5))

    start_time_ms = start_sample / sample_rate * 1000
    end_time_ms = end_sample / sample_rate * 1000
    start_line1 = ax1.axvline(start_time_ms, color='g', linestyle='--', linewidth=2)
    end_line1 = ax1.axvline(end_time_ms, color='r', linestyle='--', linewidth=1)
    start_line2 = ax2.axvline(start_time_ms, color='g', linestyle='--', linewidth=2)
    end_line2 = ax2.axvline(end_time_ms, color='r', linestyle='--', linewidth=1)
    # Horizontal labels
    label_start = ax1.annotate(f"{start_time_ms*1000:.0f} μs", xy=(start_time_ms, ax1.get_ylim()[0]),
                        xytext=(0, -18), textcoords='offset points', ha='center', va='top', color='green', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
    label_end = ax1.annotate(f"{end_time_ms*1000:.0f} μs", xy=(end_time_ms, ax1.get_ylim()[0]),
                        xytext=(0, -18), textcoords='offset points', ha='center', va='top', color='red', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
    
    state = {'drag': None, 'active': 'start', 'start': start_sample, 'end': end_sample}
    # Delta label (above right corner)
    def get_delta_us():
        return abs((state['end'] - state['start']) / sample_rate * 1e6)
    delta_us = get_delta_us()
    delta_label = ax1.annotate(f"Δ: {delta_us:.0f} μs", xy=(1, 1.08), xycoords='axes fraction', ha='right', va='bottom', fontsize=14, color='blue', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    def update_lines_and_labels():
        start_time_ms = state['start'] / sample_rate * 1000
        end_time_ms = state['end'] / sample_rate * 1000
        start_line1.set_xdata([start_time_ms, start_time_ms])
        end_line1.set_xdata([end_time_ms, end_time_ms])
        start_line2.set_xdata([start_time_ms, start_time_ms])
        end_line2.set_xdata([end_time_ms, end_time_ms])
        # Update time labels at line head (bottom y)
        y_bottom = ax1.get_ylim()[0]
        label_start.set_position((start_time_ms, y_bottom))
        label_start.set_text(f"{start_time_ms*1000:.0f} μs")
        label_end.set_position((end_time_ms, y_bottom))
        label_end.set_text(f"{end_time_ms*1000:.0f} μs")
        # Update delta label
        delta_us = get_delta_us()
        delta_label.set_text(f"Δ: {delta_us:.0f} μs")
        fig.canvas.draw_idle()

    def _select(which):
        state['active'] = which
        start_lw = 2 if which == 'start' else 1
        end_lw = 2 if which == 'end' else 1
        start_line1.set_linewidth(start_lw)
        start_line2.set_linewidth(start_lw)
        end_line1.set_linewidth(end_lw)
        end_line2.set_linewidth(end_lw)
        fig.canvas.draw_idle()

    _select('start')

    def _press(event):
        if event.inaxes not in (ax1, ax2):
            return
        x_ms = event.xdata
        x_sample = int(x_ms * sample_rate / 1000) if event.inaxes is ax1 else int(event.xdata * sample_rate / 1000)
        x_sample = max(0, min(len(signal) - 1, x_sample))
        state['drag'] = state['active']
        if state['drag'] == 'start':
            state['start'] = x_sample
        else:
            state['end'] = x_sample
        update_lines_and_labels()

    def _move(event):
        if state['drag'] is None or event.inaxes not in (ax1, ax2):
            return
        x_ms = event.xdata
        x_sample = int(x_ms * sample_rate / 1000) if event.inaxes is ax1 else int(event.xdata * sample_rate / 1000)
        x_sample = max(0, min(len(signal) - 1, x_sample))
        if state['drag'] == 'start':
            state['start'] = x_sample
        else:
            state['end'] = x_sample
        update_lines_and_labels()

    def _release(event):
        state['drag'] = None

    # Move buttons - new position centered below bottom graph
    axprev = plt.axes([0.42, 0.01, 0.07, 0.05])
    axnext = plt.axes([0.51, 0.01, 0.07, 0.05])
    bprev = Button(axprev, '<< 1μs')
    bnext = Button(axnext, '1μs >>')

    def move_line(delta_us):
        delta_samples = int(round(delta_us * sample_rate / 1e6))
        if state['active'] == 'start':
            state['start'] = max(0, min(len(signal) - 1, state['start'] + delta_samples))
        else:
            state['end'] = max(0, min(len(signal) - 1, state['end'] + delta_samples))
        update_lines_and_labels()

    bprev.on_clicked(lambda event: move_line(-1))
    bnext.on_clicked(lambda event: move_line(1))

    cid_press = fig.canvas.mpl_connect('button_press_event', _press)
    cid_move = fig.canvas.mpl_connect('motion_notify_event', _move)
    cid_rel = fig.canvas.mpl_connect('button_release_event', _release)
    cid_key = fig.canvas.mpl_connect(
        'key_press_event',
        lambda e: (
            _select('start') if e.key == 'g'
            else _select('end') if e.key == 'r'
            else plt.close(fig) if e.key == 'enter'
            else None
        ),
    )

    plt.tight_layout()
    plt.show()

    fig.canvas.mpl_disconnect(cid_press)
    fig.canvas.mpl_disconnect(cid_move)
    fig.canvas.mpl_disconnect(cid_rel)
    fig.canvas.mpl_disconnect(cid_key)

    return state['start'], state['end']

