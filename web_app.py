import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (
    load_packet,
    resample_signal,
    apply_frequency_shift,
    create_spectrogram,
    compute_freq_ranges,
    save_vector,
    save_vector_wv,
)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "data")
MAX_PACKETS = 6
TARGET_SAMPLE_RATE = 56e6

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "vector-secret-key"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def list_mat_files():
    return [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith(".mat")]


def generate_vector(configs, vector_length, output_format="mat", normalize=True):
    total_samples = int(float(vector_length) * TARGET_SAMPLE_RATE)
    vector = np.zeros(total_samples, dtype=np.complex64)
    freq_shifts = []
    markers = []
    marker_styles = ['x', 'o', '^', 's', 'D', 'P', 'v', '1', '2', '3', '4']
    marker_colors = [f"C{i}" for i in range(10)]
    style_map = {}

    for cfg in configs:
        file_path = os.path.join(UPLOAD_FOLDER, cfg['file'])
        y = load_packet(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        if base_name not in style_map:
            idx_style = len(style_map) % len(marker_styles)
            idx_color = len(style_map) % len(marker_colors)
            style_map[base_name] = (marker_styles[idx_style], marker_colors[idx_color])
        marker_style, marker_color = style_map[base_name]

        if cfg['pre_samples'] > 0:
            y = y[cfg['pre_samples']:]
        if cfg['sample_rate'] != TARGET_SAMPLE_RATE:
            y = resample_signal(y, cfg['sample_rate'], TARGET_SAMPLE_RATE)
        if cfg['freq_shift'] != 0:
            y = apply_frequency_shift(y, cfg['freq_shift'], TARGET_SAMPLE_RATE)
            freq_shifts.append(cfg['freq_shift'])
        else:
            freq_shifts.append(0)

        period_samples = int(cfg['period'] * TARGET_SAMPLE_RATE)
        start_offset = int(round(cfg['start_time'] * TARGET_SAMPLE_RATE))
        for start in range(start_offset, total_samples, period_samples):
            if start >= total_samples:
                break
            end = start + len(y)
            y_to_add = y[: max(0, min(len(y), total_samples - start))]
            vector[start:start + len(y_to_add)] += y_to_add
            markers.append((start / TARGET_SAMPLE_RATE, cfg['freq_shift'], base_name, marker_style, marker_color))

    if normalize:
        max_abs = np.abs(vector).max()
        if max_abs > 0:
            vector = vector / max_abs

    if output_format == "wv":
        out_name = "output_vector.wv"
        save_vector_wv(vector, os.path.join(UPLOAD_FOLDER, out_name), TARGET_SAMPLE_RATE)
    else:
        out_name = "output_vector.mat"
        save_vector(vector, os.path.join(UPLOAD_FOLDER, out_name))

    if freq_shifts:
        center_freq = (min(freq_shifts) + max(freq_shifts)) / 2
    else:
        center_freq = 0
    f, t, Sxx = create_spectrogram(vector, TARGET_SAMPLE_RATE, center_freq=center_freq)
    ranges = compute_freq_ranges(freq_shifts)
    plt.figure()
    # reuse plot_spectrogram logic without showing window
    from utils import plot_spectrogram
    plot_spectrogram(
        f,
        t,
        Sxx,
        center_freq=center_freq,
        title='Final Vector Spectrogram',
        packet_markers=markers,
        freq_ranges=ranges,
        show_colorbar=False,
    )
    spec_file = os.path.join(UPLOAD_FOLDER, "spectrogram.png")
    plt.savefig(spec_file)
    plt.close("all")
    return out_name, os.path.basename(spec_file)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded = request.files.get("file")
        if uploaded and uploaded.filename.endswith('.mat'):
            dest = os.path.join(UPLOAD_FOLDER, uploaded.filename)
            uploaded.save(dest)
            flash("File uploaded", "success")
        else:
            flash("Invalid file", "error")
        return redirect(url_for("index"))
    files = list_mat_files()
    return render_template("index.html", files=files)


@app.route("/generate", methods=["GET", "POST"])
def generate():
    files = list_mat_files()
    if request.method == "POST":
        vector_length = request.form.get("vector_length", type=float)
        packet_count = min(int(request.form.get("packet_count", 1)), MAX_PACKETS)
        configs = []
        for i in range(packet_count):
            cfg = {
                'file': request.form.get(f"file_{i}"),
                'sample_rate': float(request.form.get(f"sr_{i}") or 56) * 1e6,
                'freq_shift': float(request.form.get(f"fs_{i}") or 0) * 1e6,
                'period': float(request.form.get(f"period_{i}") or 0.1) / 1000.0,
                'pre_samples': int(request.form.get(f"pre_{i}") or 0),
                'start_time': float(request.form.get(f"start_{i}") or 0) / 1000.0,
            }
            if cfg['file']:
                configs.append(cfg)
        output_format = request.form.get("format", "mat")
        out_name, spec_name = generate_vector(configs, vector_length, output_format)
        return render_template(
            "result.html",
            vector_file=out_name,
            spectrogram=spec_name,
        )
    return render_template("generate.html", files=files, max_packets=MAX_PACKETS)


@app.route("/data/<path:filename>")
def data_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
