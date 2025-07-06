#!/usr/bin/env python3
"""
Spectrogram Cleanliness Tests - Monitor display quality
=====================================================

This test suite ensures that spectrograms remain clean and noise-free,
providing clear visualizations like in the reference image 2.

Tests include:
- Dynamic range verification (should be ≤30 dB)
- Visual noise detection
- Signal clarity assessment
- Comparison with reference standards
"""

import numpy as np
import sys
import os

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import create_spectrogram, normalize_spectrogram, plot_spectrogram

def test_dynamic_range_limits():
    """Test that dynamic range is limited to ensure clean display."""
    print("🔍 Testing dynamic range limits...")
    
    # Create test signal
    sr = 1000000  # 1 MHz
    duration = 0.0005  # 500 μs
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq = 100000  # 100 kHz
    
    # Windowed tone burst
    window = np.hanning(len(t))
    signal = window * np.exp(2j * np.pi * freq * t)
    
    # Create spectrogram
    f, t_spec, Sxx = create_spectrogram(signal, sr, time_resolution_us=5)
    Sxx_db, vmin, vmax = normalize_spectrogram(Sxx)
    
    dynamic_range = vmax - vmin
    
    # Verify dynamic range is reasonable for clean display
    assert dynamic_range <= 35, f"Dynamic range too high: {dynamic_range:.1f} dB (should be ≤35 dB)"
    assert dynamic_range >= 20, f"Dynamic range too low: {dynamic_range:.1f} dB (should be ≥20 dB)"
    
    print(f"   ✅ Dynamic range: {dynamic_range:.1f} dB (within acceptable range)")
    return True

def test_spectrogram_shape_adequacy():
    """Test that spectrogram has adequate time-frequency resolution."""
    print("🔍 Testing spectrogram shape adequacy...")
    
    # Create test signal
    sr = 1000000  # 1 MHz
    duration = 0.001  # 1 ms - longer signal
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq = 150000  # 150 kHz
    
    window = np.hanning(len(t))
    signal = window * np.exp(2j * np.pi * freq * t)
    
    # Create spectrogram
    f, t_spec, Sxx = create_spectrogram(signal, sr, time_resolution_us=5)
    
    # Verify adequate time bins for visualization
    time_bins = Sxx.shape[1]
    freq_bins = Sxx.shape[0]
    
    assert time_bins >= 50, f"Too few time bins: {time_bins} (should be ≥50 for clear visualization)"
    assert freq_bins >= 200, f"Too few frequency bins: {freq_bins} (should be ≥200)"
    
    print(f"   ✅ Spectrogram shape: {Sxx.shape} (adequate resolution)")
    return True

def test_tone_burst_clarity():
    """Test that a simple tone burst appears as a clean horizontal line."""
    print("🔍 Testing tone burst clarity...")
    
    # Create clean tone burst
    sr = 1000000  # 1 MHz
    duration = 0.0005  # 500 μs
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq = 100000  # 100 kHz
    
    window = np.hanning(len(t))
    signal = window * np.exp(2j * np.pi * freq * t)
    
    # Create spectrogram
    f, t_spec, Sxx = create_spectrogram(signal, sr, time_resolution_us=5)
    Sxx_db, vmin, vmax = normalize_spectrogram(Sxx)
    
    # Find the frequency bin closest to our test frequency
    freq_idx = np.argmin(np.abs(f - freq))
    
    # Check that the energy is concentrated in the correct frequency bin
    # Get the power along the time axis at the target frequency
    target_power = Sxx_db[freq_idx, :]
    
    # Get average power at other frequencies (should be much lower)
    other_freqs = np.concatenate([Sxx_db[:freq_idx-10, :], Sxx_db[freq_idx+10:, :]])
    avg_other_power = np.mean(other_freqs)
    avg_target_power = np.mean(target_power)
    
    # Signal should be at least 15 dB above noise floor
    signal_to_noise = avg_target_power - avg_other_power
    
    assert signal_to_noise >= 15, f"Signal too weak: {signal_to_noise:.1f} dB above noise (should be ≥15 dB)"
    
    print(f"   ✅ Tone burst clarity: {signal_to_noise:.1f} dB above noise floor")
    return True

def test_multi_tone_separation():
    """Test that multiple tones appear as separate clean lines."""
    print("🔍 Testing multi-tone separation...")
    
    # Create multi-tone signal
    sr = 1000000  # 1 MHz
    duration = 0.0005  # 500 μs
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Three distinct frequencies
    freq1, freq2, freq3 = 80000, 120000, 160000  # 80, 120, 160 kHz
    
    window = np.hanning(len(t))
    signal = window * (
        0.7 * np.exp(2j * np.pi * freq1 * t) +
        0.5 * np.exp(2j * np.pi * freq2 * t) +
        0.3 * np.exp(2j * np.pi * freq3 * t)
    )
    
    # Create spectrogram
    f, t_spec, Sxx = create_spectrogram(signal, sr, time_resolution_us=5)
    Sxx_db, vmin, vmax = normalize_spectrogram(Sxx)
    
    # Find frequency bins for each tone
    freq1_idx = np.argmin(np.abs(f - freq1))
    freq2_idx = np.argmin(np.abs(f - freq2))
    freq3_idx = np.argmin(np.abs(f - freq3))
    
    # Check that each frequency has significant power
    power1 = np.mean(Sxx_db[freq1_idx, :])
    power2 = np.mean(Sxx_db[freq2_idx, :])
    power3 = np.mean(Sxx_db[freq3_idx, :])
    
    # Background noise level
    background = np.mean(Sxx_db[0:freq1_idx-20, :])
    
    # Each tone should be well above background
    snr1 = power1 - background
    snr2 = power2 - background
    snr3 = power3 - background
    
    assert snr1 >= 10, f"Tone 1 too weak: {snr1:.1f} dB (should be ≥10 dB)"
    assert snr2 >= 10, f"Tone 2 too weak: {snr2:.1f} dB (should be ≥10 dB)"
    assert snr3 >= 8, f"Tone 3 too weak: {snr3:.1f} dB (should be ≥8 dB)"
    
    print(f"   ✅ Multi-tone separation: {snr1:.1f}, {snr2:.1f}, {snr3:.1f} dB above noise")
    return True

def test_frequency_sweep_clarity():
    """Test that a frequency sweep appears as a clean diagonal line."""
    print("🔍 Testing frequency sweep clarity...")
    
    # Create frequency sweep
    sr = 1000000  # 1 MHz
    duration = 0.0008  # 800 μs
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    f_start = 70000   # 70 kHz
    f_end = 180000    # 180 kHz
    
    window = np.hanning(len(t))
    signal = window * np.exp(2j * np.pi * (f_start * t + (f_end - f_start) * t**2 / (2 * duration)))
    
    # Create spectrogram
    f, t_spec, Sxx = create_spectrogram(signal, sr, time_resolution_us=5)
    Sxx_db, vmin, vmax = normalize_spectrogram(Sxx)
    
    # For a sweep, the peak power should move across frequency bins over time
    # Check that we have a clear sweep pattern
    time_slices = min(10, Sxx.shape[1])
    sweep_powers = []
    
    for i in range(0, Sxx.shape[1], max(1, Sxx.shape[1] // time_slices)):
        time_slice = Sxx_db[:, i]
        peak_freq_idx = np.argmax(time_slice)
        peak_power = time_slice[peak_freq_idx]
        background_power = np.mean(time_slice)
        sweep_powers.append(peak_power - background_power)
    
    avg_sweep_power = np.mean(sweep_powers)
    
    assert avg_sweep_power >= 12, f"Sweep too weak: {avg_sweep_power:.1f} dB (should be ≥12 dB)"
    
    print(f"   ✅ Frequency sweep clarity: {avg_sweep_power:.1f} dB above noise")
    return True

def test_noise_suppression():
    """Test that noise is properly suppressed in the visualization."""
    print("🔍 Testing noise suppression...")
    
    # Create signal with added noise
    sr = 1000000  # 1 MHz
    duration = 0.0005  # 500 μs
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq = 110000  # 110 kHz
    
    # Signal with noise
    window = np.hanning(len(t))
    clean_signal = window * np.exp(2j * np.pi * freq * t)
    noise = 0.1 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
    noisy_signal = clean_signal + noise
    
    # Create spectrogram
    f, t_spec, Sxx = create_spectrogram(noisy_signal, sr, time_resolution_us=5)
    Sxx_db, vmin, vmax = normalize_spectrogram(Sxx)
    
    # Check that the dynamic range is still controlled
    dynamic_range = vmax - vmin
    
    assert dynamic_range <= 35, f"Noise not suppressed: {dynamic_range:.1f} dB range (should be ≤35 dB)"
    
    # Find signal frequency
    freq_idx = np.argmin(np.abs(f - freq))
    signal_power = np.mean(Sxx_db[freq_idx, :])
    
    # Check noise floor
    noise_indices = np.concatenate([
        np.arange(0, max(0, int(freq_idx) - 20)),
        np.arange(min(len(f), int(freq_idx) + 20), len(f))
    ])
    noise_power = np.mean(Sxx_db[noise_indices, :])
    
    snr = signal_power - noise_power
    
    assert snr >= 12, f"Insufficient noise suppression: {snr:.1f} dB SNR (should be ≥12 dB)"
    
    print(f"   ✅ Noise suppression: {snr:.1f} dB SNR, {dynamic_range:.1f} dB range")
    return True

def run_all_cleanliness_tests():
    """Run all spectrogram cleanliness tests."""
    print("🧹 SPECTROGRAM CLEANLINESS TEST SUITE")
    print("=" * 60)
    print("Ensuring spectrograms remain clean and noise-free for clear packet analysis")
    print()
    
    tests = [
        test_dynamic_range_limits,
        test_spectrogram_shape_adequacy,
        test_tone_burst_clarity,
        test_multi_tone_separation,
        test_frequency_sweep_clarity,
        test_noise_suppression,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"   ❌ {test.__name__} failed")
        except Exception as e:
            failed += 1
            print(f"   ❌ {test.__name__} failed: {e}")
        print()
    
    print("=" * 60)
    print(f"CLEANLINESS TEST RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎯 ✅ ALL CLEANLINESS TESTS PASSED")
        print("Spectrograms are clean and suitable for detailed packet analysis!")
    else:
        print("⚠️  SOME CLEANLINESS TESTS FAILED")
        print("Spectrogram quality may be degraded - review implementation!")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_cleanliness_tests()
    sys.exit(0 if success else 1)