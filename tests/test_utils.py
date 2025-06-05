import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import generate_sample_packet, find_packet_start, apply_frequency_shift, create_spectrogram


def test_generate_sample_packet_length():
    sr = 8000
    duration = 0.5
    packet = generate_sample_packet(duration, sr, frequency=1000)
    assert len(packet) == int(sr * duration)


def test_find_packet_start_energy():
    sig = np.concatenate([np.zeros(100), np.ones(50), np.zeros(20)])
    start = find_packet_start(sig)
    assert 98 <= start <= 102


def test_find_packet_start_template():
    template = np.array([1.0, 1.0, 1.0])
    sig = np.concatenate([np.zeros(10), template, np.zeros(5)])
    start = find_packet_start(sig, template=template)
    assert start == 10


def _peak_frequency(sig, sr):
    spectrum = np.fft.fftshift(np.fft.fft(sig))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(sig), d=1 / sr))
    return freqs[np.argmax(np.abs(spectrum))]


def test_apply_frequency_shift():
    sr = 8000
    duration = 0.1
    base_freq = 1000
    shift = 500
    sig = generate_sample_packet(duration, sr, base_freq)

    shifted_pos = apply_frequency_shift(sig, shift, sr)
    freq_pos = _peak_frequency(shifted_pos, sr)
    assert abs(freq_pos - (base_freq + shift)) < 1

    shifted_neg = apply_frequency_shift(sig, -shift, sr)
    freq_neg = _peak_frequency(shifted_neg, sr)
    assert abs(freq_neg - (base_freq - shift)) < 1

    # dtype should always be complex64
    assert shifted_pos.dtype == np.complex64
    assert shifted_neg.dtype == np.complex64


def _spectrogram_peak_frequency(sig, sr):
    f, _, Sxx = create_spectrogram(sig, sr)
    idx = np.unravel_index(np.argmax(np.abs(Sxx)), Sxx.shape)
    return f[idx[0]]


def test_spectrogram_reflects_shift():
    sr = 8000
    base_freq = 1000
    shift = 500
    sig = generate_sample_packet(0.1, sr, base_freq)
    shifted = apply_frequency_shift(sig, shift, sr)
    peak = _spectrogram_peak_frequency(shifted, sr)
    assert abs(peak - (base_freq + shift)) < 2  # within a bin


def test_create_spectrogram_preserves_rate():
    sr = 10_000_000
    sig = np.zeros(1_500_000, dtype=np.float32)
    f, _, _ = create_spectrogram(sig, sr)
    assert np.isclose(max(abs(f)), sr / 2, rtol=0.01)

