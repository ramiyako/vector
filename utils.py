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

def load_packet_info(file_path):
    """Load packet data and pre-buffer info from MAT file."""
    data = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    if 'Y' in data:
        packet = data['Y']
    else:
        candidates = [k for k in data.keys() if not k.startswith('__')]
        if len(candidates) == 1:
            packet = data[candidates[0]]
        else:
            raise ValueError(f"Ambiguous packet data in {file_path}. Available keys: {list(data.keys())}")

    if packet.ndim > 1:
        packet = packet.flatten()

    pre = int(data.get('pre_samples', 0))
    return packet.astype(np.complex64), pre

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

def create_spectrogram(sig, sr, center_freq=0, max_samples=2_000_000, time_resolution_us=1, adaptive_resolution=True):
    """Creates ultra-high-resolution spectrogram from signal for detailed packet analysis.

    Parameters
    ----------
    sig : ndarray
        Input complex signal.
    sr : float
        Sample rate in Hz.
    center_freq : float, optional
        Center frequency for shifting the axis.
    max_samples : int, optional
        Downsample if signal is longer than this. Increased to 2M for better detail.
    time_resolution_us : float, optional
        Desired time resolution in microseconds. Defaults to 1us for fine detail.
    adaptive_resolution : bool, optional
        Enable adaptive resolution based on signal characteristics. Default True.
    """
    if len(sig) == 0:
        raise ValueError("Signal is empty")
    
    # Preserve more data for better resolution
    if len(sig) > max_samples:
        factor = int(np.ceil(len(sig) / max_samples))
        sig = sig[::factor]
        fs = sr / factor
    else:
        factor = 1
        fs = sr

    # Calculate signal duration
    signal_duration_us = len(sig) / fs * 1e6
    
    # Clean spectrogram calculation - optimized for clear packet visualization
    if adaptive_resolution:
        # Use smaller windows for cleaner spectrograms
        if signal_duration_us <= 50:  # Very short signals (≤50μs)
            base_window = max(32, min(len(sig) // 12, 256))
            time_resolution_us = min(time_resolution_us, signal_duration_us / 15)
            freq_resolution_factor = 1.5
        elif signal_duration_us <= 500:  # Short signals (≤500μs)
            base_window = max(64, min(len(sig) // 10, 512))
            time_resolution_us = min(time_resolution_us, signal_duration_us / 25)
            freq_resolution_factor = 1.5
        elif signal_duration_us <= 5000:  # Medium signals (≤5ms)
            base_window = max(128, min(len(sig) // 8, 1024))
            time_resolution_us = min(time_resolution_us, 5)
            freq_resolution_factor = 2
        else:  # Long signals (>5ms)
            base_window = max(256, min(len(sig) // 6, 2048))
            time_resolution_us = min(time_resolution_us, 10)
            freq_resolution_factor = 2
    else:
        # Clean defaults for better visualization
        base_window = max(128, min(len(sig) // 8, 1024))
        freq_resolution_factor = 1.5

    # Calculate optimal window size and step
    if time_resolution_us is not None:
        step_samples = max(1, int(round(fs * time_resolution_us / 1e6)))
        # Ensure we get enough time bins
        min_steps = 10  # Minimum number of time steps
        max_step = len(sig) // min_steps
        step_samples = min(step_samples, max_step)
        step_samples = max(1, step_samples)
        
        # Window should be at least 2x step size for good overlap
        window_size = max(base_window, step_samples * 3)
        window_size = min(window_size, len(sig))
        
        # Calculate overlap
        overlap = max(0, window_size - step_samples)
    else:
        window_size = min(base_window, len(sig))
        overlap = int(window_size * 0.90)  # 90% overlap

    # Optimized NFFT for clean visualization
    nfft = max(256, int(2 ** np.ceil(np.log2(window_size * freq_resolution_factor))))
    nfft = max(nfft, 512)  # Minimum NFFT - smaller for cleaner display

    # Create spectrogram with error handling
    try:
        freqs, times, Sxx = scipy.signal.spectrogram(
            sig,
            fs=fs,
            window='blackmanharris',     # Excellent sidelobe suppression for clear spectral lines
            nperseg=window_size,
            noverlap=overlap,
            nfft=nfft,
            return_onesided=False,
            detrend='constant',          # Remove DC component for cleaner display
            scaling='density'            # Power spectral density for better dynamic range
        )
    except Exception as e:
        # Fallback to basic parameters if advanced settings fail
        window_size = min(256, len(sig))
        overlap = window_size // 2
        nfft = 512
        freqs, times, Sxx = scipy.signal.spectrogram(
            sig,
            fs=fs,
            window='hann',
            nperseg=window_size,
            noverlap=overlap,
            nfft=nfft,
            return_onesided=False,
            detrend='constant',
            scaling='density'
        )

    # Proper frequency shifting and centering
    freqs = np.fft.fftshift(freqs) * factor + center_freq
    Sxx = np.fft.fftshift(Sxx, axes=0)
    
    return freqs, times, Sxx


def normalize_spectrogram(Sxx, low_percentile=10.0, high_percentile=95.0, max_dynamic_range=30):
    """Normalize spectrogram for clean packet visualization with limited dynamic range."""
    # Handle edge cases
    if Sxx.size == 0:
        return np.array([]), 0, 0
    
    # Convert to dB with better handling of zeros
    Sxx_abs = np.abs(Sxx)
    # Use a floor value that's relative to the signal strength
    noise_floor = np.percentile(Sxx_abs[Sxx_abs > 0], 5) if np.any(Sxx_abs > 0) else 1e-12
    noise_floor = max(noise_floor, 1e-12)
    
    Sxx_db = 10 * np.log10(Sxx_abs + noise_floor)

    # Calculate percentiles with error handling - use tighter range for cleaner display
    try:
        vmin = np.percentile(Sxx_db, low_percentile)
        vmax = np.percentile(Sxx_db, high_percentile)  # 95% instead of 98%
    except:
        # Fallback if percentile calculation fails
        vmin = np.min(Sxx_db)
        vmax = np.max(Sxx_db)

    # Ensure reasonable dynamic range
    if np.isnan(vmin) or np.isnan(vmax) or vmax <= vmin:
        vmin = np.min(Sxx_db)
        vmax = np.max(Sxx_db)
        if vmax <= vmin:
            vmax = vmin + 30  # Smaller default range for cleaner display

    # Apply limited dynamic range for cleaner visualization
    if vmax - vmin > max_dynamic_range:
        vmin = vmax - max_dynamic_range  # 30 dB instead of 60 dB

    # Ensure reasonable floor
    vmin = max(vmin, -100)  # Not lower than -100 dB
    
    return Sxx_db, vmin, vmax


def plot_spectrogram(
    f,
    t,
    Sxx,
    center_freq=0,
    title='High-Resolution Spectrogram',
    packet_start=None,
    sample_rate=None,
    signal=None,
    packet_markers=None,
    freq_ranges=None,
    show_colorbar=True,
    enhance_contrast=True,
    high_detail_mode=True,
):
    """Plot ultra-sharp and detailed spectrogram with advanced enhancements for packet analysis."""
    # Handle edge cases
    if Sxx.size == 0:
        print("Empty spectrogram data!")
        return
    
    if enhance_contrast:
        # Enhanced normalization for better packet detail visibility
        Sxx_db, vmin, vmax = normalize_spectrogram(Sxx, low_percentile=5, high_percentile=95, max_dynamic_range=40)
    else:
        Sxx_db, vmin, vmax = normalize_spectrogram(Sxx)
    
    # Apply light median filtering for cleaner display
    try:
        from scipy import ndimage
        Sxx_db = ndimage.median_filter(Sxx_db, size=(2, 1))  # Light filtering in frequency only
    except ImportError:
        pass  # Skip filtering if scipy is not available
    
    # Handle single time bin case - extend the time axis slightly
    if len(t) == 1:
        print("Single time bin detected - extending time axis for visualization")
        dt = 1e-6  # 1 microsecond extension
        t = np.array([t[0] - dt/2, t[0] + dt/2])
        # Duplicate the spectrogram data
        Sxx_db = np.hstack([Sxx_db, Sxx_db])
    
    freq_axis = f / 1e6  # MHz

    # Create figure with higher DPI for crisp display
    if signal is None:
        fig, ax1 = plt.subplots(figsize=(16, 8), dpi=100)
        ax2 = None
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1], dpi=100)

    # Enhanced visualization parameters for high-resolution data
    if high_detail_mode:
        # Use 'nearest' for pixel-perfect display of high-resolution data
        shading_method = 'nearest'
        # Enhanced colormap with better contrast for packet analysis
        colormap = 'turbo'  # Better than inferno for distinguishing fine details
        interpolation = 'none'  # No interpolation for crisp edges
    else:
        shading_method = 'gouraud'
        colormap = 'inferno'
        interpolation = 'bilinear'

    # Create the pcolormesh plot with error handling
    try:
        im = ax1.pcolormesh(
            t,
            freq_axis,
            Sxx_db,
            shading=shading_method,
            cmap=colormap,
            norm=Normalize(vmin=vmin, vmax=vmax),
            rasterized=True,  # Better performance for high-resolution plots
        )
    except Exception as e:
        print(f"Error creating pcolormesh: {e}")
        # Fallback to basic plotting
        im = ax1.imshow(
            Sxx_db,
            aspect='auto',
            origin='lower',
            extent=[t.min(), t.max(), freq_axis.min(), freq_axis.max()],
            cmap=colormap,
            norm=Normalize(vmin=vmin, vmax=vmax)
        )

    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time [s]', fontsize=12)
    ax1.set_ylabel('Frequency [MHz]', fontsize=12)
    ax1.set_ylim(freq_axis.min(), freq_axis.max())
    
    # Enhanced grid for better readability
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.minorticks_on()
    ax1.grid(True, which='minor', alpha=0.1, linestyle=':', linewidth=0.3)

    # Enhanced packet markers with better visibility
    if packet_markers:
        marker_styles = ['x', 'o', '^', 's', 'D', 'P', 'v', '<', '>', '1', '2', '3', '4']
        marker_colors = ['red', 'cyan', 'yellow', 'lime', 'magenta', 'orange', 'white', 'pink']
        seen_labels = {}
        for idx, marker in enumerate(packet_markers):
            tm, freq = marker[:2]
            label = marker[2] if len(marker) > 2 else f'Marker {idx+1}'
            style = marker[3] if len(marker) > 3 else marker_styles[idx % len(marker_styles)]
            color = marker[4] if len(marker) > 4 else marker_colors[idx % len(marker_colors)]
            show_label = label if label not in seen_labels else "_nolegend_"
            seen_labels[label] = (style, color)
            ax1.plot(tm, freq / 1e6, linestyle='None', marker=style, color=color, 
                    label=show_label, markersize=8, markeredgewidth=2, markeredgecolor='black')

    # Enhanced vertical packet start line
    if packet_start is not None and sample_rate is not None:
        packet_time = packet_start / sample_rate
        ax1.axvline(x=packet_time, color='lime', linestyle='-', linewidth=3, 
                   alpha=0.8, label='Packet Start')

    # Enhanced colorbar with better formatting
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax1, label='Power Spectral Density [dB/Hz]', shrink=0.8)
        cbar.ax.tick_params(labelsize=10)

    # Enhanced signal waveform display
    if signal is not None and ax2 is not None:
        # Plot both magnitude and phase for complex signals
        time_axis = np.arange(len(signal)) / sample_rate if sample_rate else np.arange(len(signal))
        
        # Primary plot: magnitude
        ax2.plot(time_axis, np.abs(signal), 'b-', linewidth=1, label='Magnitude', alpha=0.8)
        
        # Secondary plot: phase (if signal is complex)
        if np.iscomplexobj(signal):
            ax2_phase = ax2.twinx()
            ax2_phase.plot(time_axis, np.angle(signal), 'r-', linewidth=0.8, alpha=0.6, label='Phase')
            ax2_phase.set_ylabel('Phase [rad]', fontsize=10, color='red')
            ax2_phase.tick_params(axis='y', labelcolor='red', labelsize=9)
            ax2_phase.set_ylim(-np.pi, np.pi)
        
        if packet_start is not None:
            packet_time = packet_start / sample_rate if sample_rate else packet_start
            ax2.axvline(x=packet_time, color='lime', linestyle='-', linewidth=3, 
                       alpha=0.8, label='Packet Start')
        
        ax2.set_title('Signal Time Domain', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time [s]' if sample_rate else 'Samples', fontsize=10)
        ax2.set_ylabel('Magnitude', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)

    # Enhanced frequency range indicators
    if freq_ranges:
        for i, (fmin, fmax) in enumerate(freq_ranges):
            ax1.axhspan(fmin/1e6, fmax/1e6, alpha=0.2, color=f'C{i%10}', 
                       label=f'Range {i+1}: {fmin/1e6:.1f}-{fmax/1e6:.1f} MHz')

    if packet_markers or freq_ranges:
        ax1.legend(fontsize=10, loc='upper right')

    # Tight layout with optimized spacing
    plt.tight_layout(pad=1.5)
    
    # Add text annotation with resolution info
    if hasattr(t, '__len__') and len(t) > 1:
        time_res_us = (t[1] - t[0]) * 1e6
        freq_res_khz = (f[1] - f[0]) / 1e3 if len(f) > 1 else 0
        resolution_text = f'Time res: {time_res_us:.2f}μs, Freq res: {freq_res_khz:.2f}kHz'
        ax1.text(0.02, 0.98, resolution_text, transform=ax1.transAxes, 
                fontsize=9, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.show()


def save_vector(vector, output_path):
    """Save vector as MAT file compatible with packet extractor"""
    # Ensure vector is 1D and complex64
    if vector.ndim > 1:
        vector = vector.flatten()
    vector = vector.astype(np.complex64)
    
    # Save with compatible format (add pre_samples=0 for vectors)
    sio.savemat(output_path, {
        'Y': vector,
        'pre_samples': 0  # Vectors don't have pre-buffer
    })

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

def detect_packet_bounds(signal, sample_rate, threshold_ratio=0.2):
    """Detect packet start and end with microsecond resolution."""
    energy = np.abs(signal) ** 2
    window = max(1, int(sample_rate // 1_000_000))  # 1 us smoothing window
    kernel = np.ones(window) / window
    smoothed = np.convolve(energy, kernel, mode="same")
    noise = np.median(smoothed[: max(1, len(smoothed) // 10)])
    max_en = smoothed.max()
    threshold = noise + threshold_ratio * (max_en - noise)
    indices = np.where(smoothed >= threshold)[0]
    if len(indices) == 0:
        return 0, len(signal)
    start = indices[0]
    end = indices[-1]
    return start, end

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
    f, t, Sxx = create_spectrogram(signal, sample_rate, time_resolution_us=1)
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

    f, t, Sxx = create_spectrogram(signal, sample_rate, time_resolution_us=1)
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

