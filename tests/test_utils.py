import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import (
    generate_sample_packet,
    find_packet_start,
    apply_frequency_shift,
    create_spectrogram,
    plot_spectrogram,
    save_vector_wv,
    normalize_signal,
    check_vector_power_uniformity,
    enforce_vector_power_uniformity,
)


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


def test_plot_spectrogram_uses_full_range():
    import matplotlib
    matplotlib.use("Agg")
    sr = 8000
    sig = generate_sample_packet(0.1, sr, 1000)
    f, t, Sxx = create_spectrogram(sig, sr, center_freq=500)
    plot_spectrogram(f, t, Sxx, center_freq=500)
    ax = matplotlib.pyplot.gcf().axes[0]
    freq_axis = f / 1e6
    ylims = ax.get_ylim()
    assert np.isclose(ylims[0], freq_axis.min())
    assert np.isclose(ylims[1], freq_axis.max())
    matplotlib.pyplot.close("all")


def test_packet_markers_show_absolute_frequency():
    import matplotlib
    matplotlib.use("Agg")
    sr = 8000
    sig = generate_sample_packet(0.1, sr, 1000)
    f, t, Sxx = create_spectrogram(sig, sr, center_freq=500)
    plot_spectrogram(f, t, Sxx, center_freq=500, packet_markers=[(0.0, 700, "pkt")])
    ax = matplotlib.pyplot.gcf().axes[0]
    line = ax.lines[0]
    assert line.get_label() == "pkt"
    assert np.isclose(line.get_xdata()[0], 0.0)
    assert np.isclose(line.get_ydata()[0], 700 / 1e6)
    matplotlib.pyplot.close("all")


def test_marker_legend_deduplicated():
    import matplotlib
    matplotlib.use("Agg")
    sr = 8000
    sig = generate_sample_packet(0.1, sr, 1000)
    f, t, Sxx = create_spectrogram(sig, sr)
    markers = [
        (0.0, 900, "A", "o", "C0"),
        (0.1, 900, "A", "o", "C0"),
        (0.2, 950, "B", "x", "C1"),
    ]
    plot_spectrogram(f, t, Sxx, packet_markers=markers)
    ax = matplotlib.pyplot.gcf().axes[0]
    labels = [h.get_label() for h in ax.legend().legend_handles]
    assert labels.count("A") == 1
    assert labels.count("B") == 1
    lines = ax.lines
    assert lines[0].get_marker() == lines[1].get_marker()
    assert lines[0].get_color() == lines[1].get_color()
    matplotlib.pyplot.close("all")


def test_plot_spectrogram_without_signal_has_single_axis():
    import matplotlib
    matplotlib.use("Agg")
    sr = 8000
    sig = generate_sample_packet(0.1, sr, 1000)
    f, t, Sxx = create_spectrogram(sig, sr)
    plot_spectrogram(f, t, Sxx)
    fig = matplotlib.pyplot.gcf()
    axes_without_cb = [ax for ax in fig.axes if ax.get_ylabel() != 'Power [dB]']
    assert len(axes_without_cb) == 1
    matplotlib.pyplot.close("all")


def test_broken_axis_ranges():
    import matplotlib
    matplotlib.use("Agg")
    sr = 8000
    sig = generate_sample_packet(0.1, sr, 1000)
    f, t, Sxx = create_spectrogram(sig, sr)
    plot_spectrogram(f, t, Sxx, freq_ranges=[(900, 1100), (1500, 1700)])
    fig = matplotlib.pyplot.gcf()
    assert len(fig.axes) >= 2
    matplotlib.pyplot.close("all")


def test_save_vector_wv(tmp_path):
    data = np.array([1 + 1j, -1 - 1j], dtype=np.complex64)
    out_file = tmp_path / "out.wv"
    save_vector_wv(data, str(out_file), sample_rate=1e6)
    assert out_file.exists()
    with open(out_file, "rb") as f:
        header = f.read(20)
    assert b"TYPE: SMU-WV" in header


def test_normalize_signal_peak():
    data = np.array([0.5, 2.0, -1.0], dtype=np.float32)
    norm = normalize_signal(data)
    assert np.isclose(np.max(np.abs(norm)), 1.0)
    assert norm.dtype == np.float32


def test_check_vector_power_uniformity():
    vec = np.concatenate([np.ones(1000), 0.1 * np.ones(1000)]).astype(np.float32)
    with pytest.raises(ValueError):
        check_vector_power_uniformity(vec, window_size=100, max_db_delta=3)

    vec = np.ones(2000, dtype=np.complex64)
    check_vector_power_uniformity(vec, window_size=100, max_db_delta=3)


def test_enforce_vector_power_uniformity():
    vec = np.concatenate([np.ones(1000), 0.1 * np.ones(1000)]).astype(np.float32)
    fixed = enforce_vector_power_uniformity(vec, window_size=100, max_db_delta=3)
    check_vector_power_uniformity(fixed, window_size=100, max_db_delta=3)


