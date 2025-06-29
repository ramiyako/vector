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
import warnings
from matplotlib.colors import Normalize
try:
    from brokenaxes import brokenaxes
except ImportError:  # pragma: no cover - optional dependency
    brokenaxes = None

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
    """Creates high-resolution spectrogram from signal."""
    if len(sig) == 0:
        raise ValueError("Signal is empty")
    if len(sig) > max_samples:
        factor = int(np.ceil(len(sig) / max_samples))
        sig = sig[::factor]
        fs = sr / factor
    else:
        factor = 1
        fs = sr

    window_size = min(8192, len(sig) // 4)      # Larger window for frequency precision
    overlap = int(window_size * 0.9)            # 90% overlap for time continuity
    nfft = 16384                                # More frequency bins (sharpness)

    freqs, times, Sxx = scipy.signal.spectrogram(
        sig,
        fs=fs,
        window='blackmanharris',                # Better sidelobe suppression
        nperseg=window_size,
        noverlap=overlap,
        nfft=nfft,
        return_onesided=False,
        detrend=False
    )

    freqs = np.fft.fftshift(freqs) * factor + center_freq
    Sxx = np.fft.fftshift(Sxx, axes=0)
    return freqs, times, Sxx


def normalize_spectrogram(Sxx, low_percentile=10, high_percentile=98, max_dynamic_range=60):
    """Normalize spectrogram with clipping and dB scaling."""
    Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-12)

    vmin = np.percentile(Sxx_db, low_percentile)
    vmax = np.percentile(Sxx_db, high_percentile)

    if vmax - vmin > max_dynamic_range:
        vmin = vmax - max_dynamic_range

    vmin = max(vmin, -100)  # Clip floor
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
    """Plot sharp and clear spectrogram with enhancements."""
    Sxx_db, vmin, vmax = normalize_spectrogram(Sxx)
    freq_axis = f / 1e6  # MHz

    fig, ax1 = plt.subplots(figsize=(12, 6)) if signal is None else plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])
    ax2 = None if signal is None else fig.axes[1]

    im = ax1.pcolormesh(
        t,
        freq_axis,
        Sxx_db,
        shading='gouraud',            # Smooth interpolation
        cmap='inferno',               # High-contrast colormap
        norm=Normalize(vmin=vmin, vmax=vmax),
    )

    ax1.set_title(title)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Frequency [MHz]')
    ax1.set_ylim(freq_axis.min(), freq_axis.max())
    ax1.grid(True)

    # Packet markers (optional)
    if packet_markers:
        marker_styles = ['x', 'o', '^', 's', 'D', 'P', 'v', '1', '2', '3', '4']
        seen_labels = {}
        for idx, marker in enumerate(packet_markers):
            tm, freq = marker[:2]
            label = marker[2] if len(marker) > 2 else None
            style = marker[3] if len(marker) > 3 else marker_styles[idx % len(marker_styles)]
            color = marker[4] if len(marker) > 4 else f"C{idx % 10}"
            show_label = label if label not in seen_labels else "_nolegend_"
            seen_labels[label] = (style, color)
            ax1.plot(tm, freq / 1e6, linestyle='None', marker=style, color=color, label=show_label)

    # Vertical packet start line
    if packet_start is not None and sample_rate is not None:
        packet_time = packet_start / sample_rate
        ax1.axvline(x=packet_time, color='r', linestyle='--', label='Packet Start')

    if show_colorbar:
        plt.colorbar(im, ax=ax1, label='Power [dB]')

    # Signal waveform below (optional)
    if signal is not None:
        ax2.plot(np.abs(signal))
        if packet_start is not None:
            ax2.axvline(x=packet_start, color='r', linestyle='--', label='Packet Start')
        ax2.set_title('Signal Amplitude')
        ax2.set_xlabel('Samples')
        ax2.set_ylabel('Amplitude')
        ax2.legend()

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

    # Create maximized figure and subplots
    plt.rcParams['figure.max_open_warning'] = 0  # Disable figure warning
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), height_ratios=[3, 1])
    plt.ioff(); plt.ion()  # Reset interactive mode
    
    manager = fig.canvas.manager
    try:
        # Try to maximize window on different backends
        if hasattr(manager, 'window'):
            if hasattr(manager.window, 'showMaximized'):
                manager.window.showMaximized()
            elif hasattr(manager.window, 'state'):
                manager.window.state('zoomed')
            elif hasattr(manager.window, 'wm_state'):
                manager.window.wm_state('zoomed')
        elif hasattr(manager, 'full_screen_toggle'):
            manager.full_screen_toggle()
        elif hasattr(manager, 'resize'):
            # Try to get screen dimensions and resize
            try:
                import tkinter as tk
                root = tk.Tk()
                screen_width = root.winfo_screenwidth()
                screen_height = root.winfo_screenheight()
                root.destroy()
                manager.resize(screen_width-100, screen_height-100)
            except:
                pass
    except:
        pass  # If maximization fails, continue with default size
    
    # Tight layout with minimal margins for maximum space usage
    plt.subplots_adjust(bottom=0.12, top=0.96, left=0.06, right=0.98, hspace=0.3)

    pcm = ax1.pcolormesh(t_ms, f / 1e6, Sxx_db, shading='nearest', cmap='viridis', vmin=vmin, vmax=vmax)
    ax1.set_ylim(ax1.get_ylim()[::-1])
    ax1.set_title("INSTRUCTIONS: Click and drag the orange line to adjust packet start position", fontsize=16, weight='bold')
    ax1.set_xlabel('Time [ms]', fontsize=12)
    ax1.set_ylabel('Frequency [MHz]', fontsize=12)
    
    # Adaptive resolution based on time range
    min_t, max_t = t_ms[0], t_ms[-1]
    time_range = max_t - min_t
    
    if time_range <= 10:  # Up to 10ms - 0.1ms resolution
        tick_step = 0.1
        multiplier = 10
    elif time_range <= 50:  # Up to 50ms - 0.5ms resolution
        tick_step = 0.5
        multiplier = 2
    elif time_range <= 200:  # Up to 200ms - 1ms resolution
        tick_step = 1.0
        multiplier = 1
    elif time_range <= 1000:  # Up to 1 second - 5ms resolution
        tick_step = 5.0
        multiplier = 0.2
    else:  # Over 1 second - 10ms resolution
        tick_step = 10.0
        multiplier = 0.1
    
    ax1.set_xticks(np.arange(np.floor(min_t*multiplier)/multiplier, np.ceil(max_t*multiplier)/multiplier + tick_step, tick_step))

    ax2.plot(sample_times_ms, np.abs(signal))
    ax2.set_xlabel('Time [ms]', fontsize=12)
    ax2.set_ylabel('Amplitude', fontsize=12)
    
    ax2.set_xticks(np.arange(np.floor(sample_times_ms[0]*multiplier)/multiplier, np.ceil(sample_times_ms[-1]*multiplier)/multiplier + tick_step, tick_step))

    # Clear and prominent line
    packet_time_ms = packet_start / sample_rate * 1000
    line1 = ax1.axvline(packet_time_ms, color='orange', linestyle='-', linewidth=4, alpha=0.9)
    line2 = ax2.axvline(packet_time_ms, color='orange', linestyle='-', linewidth=4, alpha=0.9)
    
    # Large and clear label
    label = ax1.annotate(f"PACKET START\n{packet_time_ms*1000:.0f} us", xy=(packet_time_ms, ax1.get_ylim()[1]),
                        xytext=(0, 10), textcoords='offset points', ha='center', va='bottom', 
                        color='orange', fontsize=12, weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.8, edgecolor='orange'))

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
        label.set_text(f"PACKET START\n{packet_time_ms*1000:.0f} us")
        fig.canvas.draw_idle()

    def _release(event):
        state['drag'] = False

    cid_press = fig.canvas.mpl_connect('button_press_event', _press)
    cid_move = fig.canvas.mpl_connect('motion_notify_event', _move)
    cid_rel = fig.canvas.mpl_connect('button_release_event', _release)

    plt.tight_layout()
    print("\n" + "="*50)
    print("INTERACTIVE WINDOW OPENED")
    print("Instructions:")
    print("1. Click and drag the orange line")
    print("2. Close window when done")
    print("="*50)
    
    # Show and wait for user interaction
    plt.show(block=True)
    
    # Wait for window to be closed
    try:
        while plt.get_fignums():
            plt.pause(0.1)
    except:
        pass

    fig.canvas.mpl_disconnect(cid_press)
    fig.canvas.mpl_disconnect(cid_move)
    fig.canvas.mpl_disconnect(cid_rel)

    return state['value']

def adjust_packet_bounds_gui(signal, sample_rate, start_sample=0, end_sample=None):
    if end_sample is None:
        end_sample = len(signal)

    f, t, Sxx = create_spectrogram(signal, sample_rate)
    Sxx_db, vmin, vmax = normalize_spectrogram(Sxx)

    # Reverse frequency axis and matrix to descending order (positive left, negative right)
    if f[0] < f[-1]:
        f = f[::-1]
        Sxx_db = Sxx_db[::-1, :]

    t_ms = t * 1000
    sample_indices = np.arange(len(signal))
    sample_times_ms = sample_indices / sample_rate * 1000

    # Create maximized figure and subplots
    plt.rcParams['figure.max_open_warning'] = 0  # Disable figure warning
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), height_ratios=[3, 1])
    plt.ioff(); plt.ion()  # Reset interactive mode
    
    manager = fig.canvas.manager
    try:
        # Try to maximize window on different backends
        if hasattr(manager, 'window'):
            if hasattr(manager.window, 'showMaximized'):
                manager.window.showMaximized()
            elif hasattr(manager.window, 'state'):
                manager.window.state('zoomed')
            elif hasattr(manager.window, 'wm_state'):
                manager.window.wm_state('zoomed')
        elif hasattr(manager, 'full_screen_toggle'):
            manager.full_screen_toggle()
        elif hasattr(manager, 'resize'):
            # Try to get screen dimensions and resize
            try:
                import tkinter as tk
                root = tk.Tk()
                screen_width = root.winfo_screenwidth()
                screen_height = root.winfo_screenheight()
                root.destroy()
                manager.resize(screen_width-100, screen_height-100)
            except:
                pass
    except:
        pass  # If maximization fails, continue with default size
    
    # Tight layout with minimal margins for maximum space usage
    plt.subplots_adjust(bottom=0.12, top=0.96, left=0.06, right=0.98, hspace=0.3)

    pcm = ax1.pcolormesh(t_ms, f / 1e6, Sxx_db, shading='nearest', cmap='viridis', vmin=vmin, vmax=vmax)
    ax1.set_ylim(ax1.get_ylim()[::-1])
    ax1.set_title("INSTRUCTIONS: Click 'g' for green line, 'r' for red line. Click and drag to move. Press Enter to finish.", fontsize=16, weight='bold')
    ax1.set_xlabel('Time [ms]', fontsize=12)
    ax1.set_ylabel('Frequency [MHz]', fontsize=12)
    
    # Adaptive resolution based on time range
    min_t, max_t = t_ms[0], t_ms[-1]
    time_range = max_t - min_t
    
    if time_range <= 10:  # Up to 10ms - 0.1ms resolution
        tick_step = 0.1
        multiplier = 10
    elif time_range <= 50:  # Up to 50ms - 0.5ms resolution
        tick_step = 0.5
        multiplier = 2
    elif time_range <= 200:  # Up to 200ms - 1ms resolution
        tick_step = 1.0
        multiplier = 1
    elif time_range <= 1000:  # Up to 1 second - 5ms resolution
        tick_step = 5.0
        multiplier = 0.2
    else:  # Over 1 second - 10ms resolution
        tick_step = 10.0
        multiplier = 0.1
    
    ax1.set_xticks(np.arange(np.floor(min_t*multiplier)/multiplier, np.ceil(max_t*multiplier)/multiplier + tick_step, tick_step))

    ax2.plot(sample_times_ms, np.abs(signal))
    ax2.set_xlabel('Time [ms]', fontsize=12)
    ax2.set_ylabel('Amplitude', fontsize=12)
    
    ax2.set_xticks(np.arange(np.floor(sample_times_ms[0]*multiplier)/multiplier, np.ceil(sample_times_ms[-1]*multiplier)/multiplier + tick_step, tick_step))

    start_time_ms = start_sample / sample_rate * 1000
    end_time_ms = end_sample / sample_rate * 1000
    
    # Thick and colorful lines for better visibility
    start_line1 = ax1.axvline(start_time_ms, color='lime', linestyle='-', linewidth=3, alpha=0.8)
    end_line1 = ax1.axvline(end_time_ms, color='red', linestyle='-', linewidth=3, alpha=0.8)
    start_line2 = ax2.axvline(start_time_ms, color='lime', linestyle='-', linewidth=3, alpha=0.8)
    end_line2 = ax2.axvline(end_time_ms, color='red', linestyle='-', linewidth=3, alpha=0.8)
    
    # Large and clear labels
    label_start = ax1.annotate(f"START\n{start_time_ms*1000:.0f} us", xy=(start_time_ms, ax1.get_ylim()[0]),
                        xytext=(0, -25), textcoords='offset points', ha='center', va='top', 
                        color='lime', fontsize=12, weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.8, edgecolor='lime'))
    label_end = ax1.annotate(f"END\n{end_time_ms*1000:.0f} us", xy=(end_time_ms, ax1.get_ylim()[0]),
                        xytext=(0, -25), textcoords='offset points', ha='center', va='top', 
                        color='red', fontsize=12, weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.8, edgecolor='red'))
    
    state = {'drag': None, 'active': 'start', 'start': start_sample, 'end': end_sample}
    
    # Enlarged and clear packet length label
    def get_delta_us():
        return abs((state['end'] - state['start']) / sample_rate * 1e6)
    delta_us = get_delta_us()
    delta_label = ax1.annotate(f"PACKET LENGTH\nΔ: {delta_us:.0f} us", xy=(1, 1.12), xycoords='axes fraction', 
                              ha='right', va='bottom', fontsize=14, weight='bold', color='white',
                              bbox=dict(boxstyle='round,pad=0.5', fc='blue', alpha=0.9, edgecolor='white'))

    def update_lines_and_labels():
        start_time_ms = state['start'] / sample_rate * 1000
        end_time_ms = state['end'] / sample_rate * 1000
        start_line1.set_xdata([start_time_ms, start_time_ms])
        end_line1.set_xdata([end_time_ms, end_time_ms])
        start_line2.set_xdata([start_time_ms, start_time_ms])
        end_line2.set_xdata([end_time_ms, end_time_ms])
        
        # Update time labels
        y_bottom = ax1.get_ylim()[0]
        label_start.set_position((start_time_ms, y_bottom))
        label_start.set_text(f"START\n{start_time_ms*1000:.0f} us")
        label_end.set_position((end_time_ms, y_bottom))
        label_end.set_text(f"END\n{end_time_ms*1000:.0f} us")
        
        # Update delta label
        delta_us = get_delta_us()
        delta_label.set_text(f"PACKET LENGTH\nΔ: {delta_us:.0f} us")
        
        # Highlight active line
        if state['active'] == 'start':
            start_line1.set_linewidth(4)
            start_line2.set_linewidth(4)
            end_line1.set_linewidth(2)
            end_line2.set_linewidth(2)
        else:
            start_line1.set_linewidth(2)
            start_line2.set_linewidth(2)
            end_line1.set_linewidth(4)
            end_line2.set_linewidth(4)
            
        fig.canvas.draw_idle()

    def _select(which):
        state['active'] = which
        update_lines_and_labels()

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

    # Fine adjustment buttons - now with 0.1us movement for high precision
    axprev = plt.axes([0.35, 0.01, 0.08, 0.05])
    axnext = plt.axes([0.44, 0.01, 0.08, 0.05])
    axprev_fine = plt.axes([0.53, 0.01, 0.08, 0.05])
    axnext_fine = plt.axes([0.62, 0.01, 0.08, 0.05])
    
    bprev = Button(axprev, '<< 1us')
    bnext = Button(axnext, '1us >>')
    bprev_fine = Button(axprev_fine, '<< 0.1us')
    bnext_fine = Button(axnext_fine, '0.1us >>')

    def move_line(delta_us):
        delta_samples = int(round(delta_us * sample_rate / 1e6))
        if state['active'] == 'start':
            state['start'] = max(0, min(len(signal) - 1, state['start'] + delta_samples))
        else:
            state['end'] = max(0, min(len(signal) - 1, state['end'] + delta_samples))
        update_lines_and_labels()

    bprev.on_clicked(lambda event: move_line(-1))
    bnext.on_clicked(lambda event: move_line(1))
    bprev_fine.on_clicked(lambda event: move_line(-0.1))
    bnext_fine.on_clicked(lambda event: move_line(0.1))

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
    print("\n" + "="*60)
    print("INTERACTIVE WINDOW OPENED")
    print("Instructions:")
    print("1. Press 'g' to select green START line")
    print("2. Press 'r' to select red END line") 
    print("3. Click and drag to move the selected line")
    print("4. Use buttons for fine adjustment")
    print("5. Press Enter to finish and close")
    print("="*60)
    
    # Show and wait for user interaction
    plt.show(block=True)
    
    # Wait for window to be closed
    try:
        while plt.get_fignums():
            plt.pause(0.1)
    except:
        pass

    fig.canvas.mpl_disconnect(cid_press)
    fig.canvas.mpl_disconnect(cid_move)
    fig.canvas.mpl_disconnect(cid_rel)
    fig.canvas.mpl_disconnect(cid_key)

    return state['start'], state['end']

