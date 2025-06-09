import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.io as sio

from utils import (
    load_packet,
    resample_signal,
    apply_frequency_shift,
    create_spectrogram,
    compute_freq_ranges,
    save_vector,
    save_vector_wv,
    get_sample_rate_from_mat,
    measure_packet_timing,
    plot_packet_with_markers,
    plot_spectrogram,
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


def save_extract_preview(signal, sr, start, end, out_path):
    f, t, Sxx = create_spectrogram(signal, sr)
    plt.figure()
    plot_spectrogram(
        f,
        t,
        Sxx,
        title="Packet Preview",
        sample_rate=sr,
        signal=signal,
    )
    plt.axvline(start / sr, color="g", linestyle="--")
    plt.axvline(end / sr, color="r", linestyle="--")
    plt.savefig(out_path)
    plt.close("all")


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
        if uploaded:
            filename = secure_filename(uploaded.filename)
            if filename:
                try:
                    dest = os.path.join(UPLOAD_FOLDER, filename)
                    uploaded.save(dest)
                    flash("File uploaded", "success")
                except Exception as e:
                    flash(f"Error saving file: {str(e)}", "error")
            else:
                flash("Invalid file name", "error")
        else:
            flash("Invalid file", "error")
        return redirect(url_for("index"))
    files = list_mat_files()
    return render_template("index.html", files=files)


@app.route("/spectrogram/<path:filename>")
def spectrogram(filename):
    filename = secure_filename(filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.isfile(file_path):
        return redirect(url_for("index"))
    sr = get_sample_rate_from_mat(file_path) or TARGET_SAMPLE_RATE
    y = load_packet(file_path)
    f, t, Sxx = create_spectrogram(y, sr)
    plt.figure()
    plot_spectrogram(f, t, Sxx, title=f"Spectrogram of {filename}", sample_rate=sr, signal=y)
    image_name = f"{filename}_spec.png"
    plt.savefig(os.path.join(UPLOAD_FOLDER, image_name))
    plt.close("all")
    return render_template("spectrogram.html", image=image_name, filename=filename)


@app.route("/analyze/<path:filename>")
def analyze(filename):
    filename = secure_filename(filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.isfile(file_path):
        return redirect(url_for("index"))
    sr = get_sample_rate_from_mat(file_path) or TARGET_SAMPLE_RATE
    y = load_packet(file_path)
    pre, post, start = measure_packet_timing(y)
    plt.figure()
    plot_packet_with_markers(y, start, title=f"Packet Analysis - {filename}")
    image_name = f"{filename}_analysis.png"
    plt.savefig(os.path.join(UPLOAD_FOLDER, image_name))
    plt.close("all")
    return render_template(
        "analyze.html",
        filename=filename,
        pre_samples=pre,
        post_samples=post,
        packet_start=start,
        image=image_name,
    )


@app.route("/extract/<path:filename>", methods=["GET", "POST"])
def extract(filename):
    filename = secure_filename(filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.isfile(file_path):
        return redirect(url_for("index"))
    y = load_packet(file_path)
    sr = get_sample_rate_from_mat(file_path) or TARGET_SAMPLE_RATE
    total = len(y)
    start = 0
    end = total
    saved_name = None
    if request.method == "POST":
        try:
            start = int(request.form.get("start", "0"))
            end = int(request.form.get("end", str(total)))
        except (TypeError, ValueError):
            flash("Invalid range", "error")
        start = max(0, min(start, total - 1))
        end = max(start + 1, min(end, total))
        saved_name = f"{os.path.splitext(filename)[0]}_extract.mat"
        sio.savemat(os.path.join(UPLOAD_FOLDER, saved_name), {"Y": y[start:end]})
        flash(f"Packet saved as {saved_name}", "success")
    preview = f"{filename}_extract.png"
    save_extract_preview(y, sr, start, end, os.path.join(UPLOAD_FOLDER, preview))
    return render_template(
        "extract.html",
        filename=filename,
        start=start,
        end=end,
        image=preview,
        saved=saved_name,
    )


@app.route("/generate", methods=["GET", "POST"])
def generate():
    files = list_mat_files()
    if request.method == "POST":
        try:
            vector_length = float(request.form.get("vector_length", "1"))
        except (TypeError, ValueError):
            flash("Invalid vector length", "error")
            return render_template("generate.html", files=files, max_packets=MAX_PACKETS)

        try:
            packet_count = int(request.form.get("packet_count", "1"))
        except (TypeError, ValueError):
            flash("Invalid packet count", "error")
            return render_template("generate.html", files=files, max_packets=MAX_PACKETS)
        packet_count = max(1, min(packet_count, MAX_PACKETS))
        configs = []
        for i in range(packet_count):
            cfg = {
                "vector_length": vector_length,
                "packet_count": 1,
                "packet_index": i,
                "file": request.form.get("file")
            }
            configs.append(cfg)
        try:
            vector = create_vector(configs)
            if vector is None:
                flash("Error creating vector", "error")
                return render_template("generate.html", files=files, max_packets=MAX_PACKETS)
            return render_template("vector.html", vector=vector)
        except Exception as e:
            flash(f"Error creating vector: {str(e)}", "error")
            return render_template("generate.html", files=files, max_packets=MAX_PACKETS)
    return render_template("generate.html", files=files, max_packets=MAX_PACKETS)


@app.route("/data/<path:filename>")
def data_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
