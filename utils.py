import numpy as np
import scipy.io as sio
import librosa
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib as mpl
from matplotlib.colors import Normalize
import warnings
try:
    from brokenaxes import brokenaxes
except ImportError:  # pragma: no cover - optional dependency
    brokenaxes = None
from matplotlib.widgets import Button

# הגדרת כיוון RTL לטקסט בעברית
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

def get_sample_rate_from_mat(file_path):
    """מחלץ את קצב הדגימה מקובץ ``.mat``.

    הפונקציה מנסה למצוא שדה הקרוי ``xDelta`` (בכתיבים שונים) ומשתמשת בו כדי
    לחשב את קצב הדגימה. ישנם קבצים שבהם השדה כבר מכיל את קצב הדגימה עצמו
    (לדוגמה "xDelta" = 56000000), ואחרים שבהם השדה מייצג את מרווח הדגימה
    (למשל ``1/Fs``). לכן מתבצעת בדיקה פשוטה: אם הערך גדול מ-1 הוא נחשב כקצב
    הדגימה, אחרת משתמשים ב-
    ``1 / value``.
    """

    data = sio.loadmat(file_path)

    for key in data.keys():
        normalized = key.lower().replace('_', '').replace(' ', '')
        if normalized == "xdelta":
            try:
                value = float(np.squeeze(data[key]))
                if value <= 0:
                    continue
                return value if value > 1 else 1.0 / value
            except Exception:
                continue
    return None

def load_packet(file_path):
    """טוען פקטה מקובץ .mat ומחזיר תמיד np.complex64 או np.float32
    תומך גם בקבצים שבהם המשתנה אינו Y, אלא משתנה בודד אחר.
    """
    data = sio.loadmat(file_path)
    if 'Y' in data:
        y = data['Y'].flatten()
    else:
        # חפש משתנה בודד שאינו פרטי (לא __header__ וכו')
        candidates = [k for k in data.keys() if not k.startswith('__')]
        if len(candidates) == 1:
            y = data[candidates[0]].flatten()
        else:
            raise ValueError(f"לא נמצא משתנה מתאים בקובץ {file_path}. משתנים קיימים: {list(data.keys())}")
    # המרה לסוג מתאים
    if np.iscomplexobj(y):
        return y.astype(np.complex64)
    else:
        return y.astype(np.float32)

def resample_signal(signal, orig_sr, target_sr):
    """Resample גם לאותות קומפלקסיים וגם ריאליים"""
    if np.iscomplexobj(signal):
        real = librosa.resample(np.real(signal).astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)
        imag = librosa.resample(np.imag(signal).astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)
        return real + 1j * imag
    else:
        return librosa.resample(signal.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)


def apply_frequency_shift(signal, freq_shift, sample_rate):
    """Apply a positive or negative frequency shift to the signal.

    The returned array is always ``complex64`` to avoid dtype inflation when the
    input is real.
    """
    if freq_shift == 0:
        # Ensure a consistent complex dtype when no shift is requested
        return signal.astype(np.complex64) if np.isrealobj(signal) else signal

    t = np.arange(len(signal), dtype=np.float64) / sample_rate
    shifted = signal.astype(np.complex64) * np.exp(2j * np.pi * freq_shift * t)
    return shifted.astype(np.complex64)


def compute_freq_ranges(shifts, margin=1e6):
    """Return merged frequency ranges around each shift.

    Parameters
    ----------
    shifts : list of float
        The frequency shift values in Hz.
    margin : float, optional
        Extra bandwidth (Hz) to include around each shift.

    Returns
    -------
    list of (float, float)
        List of (start, end) ranges in Hz or ``None`` if no shifts.
    """
    if not shifts:
        return None
    values = sorted(set(shifts))
    ranges = []
    for freq in values:
        start = freq - margin
        end = freq + margin
        if ranges and start <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))
    return ranges


def normalize_signal(sig, target_peak=1.0):
    """Scale a signal so its peak amplitude equals ``target_peak``."""
    max_abs = np.max(np.abs(sig))
    if max_abs == 0:
        dtype = np.complex64 if np.iscomplexobj(sig) else np.float32
        return sig.astype(dtype)
    scaled = sig / max_abs * target_peak
    dtype = np.complex64 if np.iscomplexobj(scaled) else np.float32
    return scaled.astype(dtype)


def check_vector_power_uniformity(vector, window_size=1024, max_db_delta=3, rectify=False):
    """Validate or fix non-uniform power across ``vector``.

    The vector is divided into non-overlapping blocks of ``window_size`` and the
    RMS power of each block is computed. If the difference between the maximum
    and minimum block power exceeds ``max_db_delta`` in dB then either a
    ``ValueError`` is raised or, when ``rectify`` is ``True``, each block is
    scaled to match the median RMS level.
    """

    if window_size <= 0:
        return
    if window_size > len(vector):
        window_size = len(vector)

    n_blocks = len(vector) // window_size
    if n_blocks < 2:
        return

    trimmed = vector[: n_blocks * window_size]
    blocks = trimmed.reshape(n_blocks, window_size)
    rms = np.sqrt(np.mean(np.abs(blocks) ** 2, axis=1))

    valid = rms > 1e-12
    rms_valid = rms[valid]
    if len(rms_valid) == 0:
        return

    db = 20 * np.log10(rms_valid)
    delta = db.max() - db.min()

    if delta > max_db_delta:
        if not rectify:
            raise ValueError(
                f"Vector power variation {delta:.2f} dB exceeds {max_db_delta} dB"
            )

        target_rms = np.median(rms_valid)
        scale = np.ones_like(rms, dtype=np.float32)
        scale[valid] = target_rms / rms_valid
        blocks *= scale[:, None]
        vector[: n_blocks * window_size] = blocks.reshape(-1)
        # scale any remaining tail with last factor
        if n_blocks * window_size < len(vector):
            vector[n_blocks * window_size :] *= scale[-1]

        # verify again (without rectify to avoid infinite recursion)
        check_vector_power_uniformity(vector, window_size, max_db_delta, rectify=False)

def create_spectrogram(sig, sr, center_freq=0, max_samples=1_000_000):
    """יוצר ספקטוגרמה מהאות.

    אם האות ארוך במיוחד, מתבצע דילול מהיר כדי להאיץ את החישוב.
    """

    if len(sig) > max_samples:
        factor = int(np.ceil(len(sig) / max_samples))
        sig = sig[::factor]
        fs = sr / factor
    else:
        factor = 1
        fs = sr

    window_size = 1024
    overlap = window_size // 2
    nfft = 1024
    freqs, times, Sxx = signal.spectrogram(
        sig,
        fs=fs,
        window='hann',
        nperseg=window_size,
        noverlap=overlap,
        nfft=nfft,
        return_onesided=False,
        detrend=False
    )

    freqs = np.fft.fftshift(freqs) * factor + center_freq
    Sxx = np.fft.fftshift(Sxx, axes=0)
    return freqs, times, Sxx

def normalize_spectrogram(Sxx, low_percentile=5, high_percentile=99, max_dynamic_range=60):
    """Normalize spectrogram values and compute display range.

    Returns normalized Sxx in dB and suitable vmin/vmax for plotting."""
    Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-10)
    Sxx_db -= np.max(Sxx_db)
    vmin = np.percentile(Sxx_db, low_percentile)
    vmax = np.percentile(Sxx_db, high_percentile)
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
    """מציג ספקטוגרמה עם ציר תדר ב-MHz, עם אפשרות לסמן מיקומים של פקטות."""
    Sxx_db, vmin, vmax = normalize_spectrogram(Sxx)

    freq_axis = f / 1e6

    # הגנה: ודא ש-freq_ranges הוא list של tuples
    if freq_ranges:
        try:
            freq_ranges = list(freq_ranges)
            freq_ranges = [tuple(fr) for fr in freq_ranges]
        except Exception as e:
            warnings.warn(f"freq_ranges לא תקין: {e}. תוצג כל הספקטרוגרמה.")
            freq_ranges = None

    # אם יש רק טווח אחד, אל תשתמש ב-brokenaxes
    if freq_ranges and len(freq_ranges) == 1:
        freq_ranges = None

    if freq_ranges and len(freq_ranges) > 1:
        try:
            if brokenaxes is None:
                warnings.warn(
                    "brokenaxes is not installed; displaying full frequency range"
                )
                freq_ranges = None
            else:
                ylims = [(lo / 1e6, hi / 1e6) for lo, hi in freq_ranges]
                fig = plt.figure(figsize=(12, 6))
                ax1 = brokenaxes(ylims=ylims, hspace=0.05)
                im = ax1.pcolormesh(
                    t,
                    freq_axis,
                    Sxx_db,
                    shading='nearest',
                    cmap='viridis',
                    norm=Normalize(vmin=vmin, vmax=vmax),
                )
        except Exception as e:
            warnings.warn(f"שגיאה בשימוש ב-brokenaxes: {e}. תוצג כל הספקטרוגרמה.")
            freq_ranges = None

    if not freq_ranges:
        if signal is not None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])
        else:
            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax2 = None
        im = ax1.pcolormesh(
            t,
            freq_axis,
            Sxx_db,
            shading='nearest',
            cmap='viridis',
            norm=Normalize(vmin=vmin, vmax=vmax),
        )
        ax1.set_ylim(freq_axis.min(), freq_axis.max())

    ax1.set_title(title)
    ax1.set_xlabel('Time [s]')
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
            ax1.plot(tm, freq / 1e6, linestyle='None', marker=style, color=color, label=show_label)
    if packet_start is not None and sample_rate is not None:
        packet_time = packet_start / sample_rate
        ax1.axvline(x=packet_time, color='r', linestyle='--', label='Packet Start')
    if show_colorbar:
        if freq_ranges:
            plt.colorbar(im[0], ax=ax1.axs, label='Power [dB]')
        else:
            plt.colorbar(im, ax=ax1, label='Power [dB]')

    if signal is not None and not freq_ranges:
        ax2.plot(np.abs(signal))
        if packet_start is not None:
            ax2.axvline(x=packet_start, color='r', linestyle='--', label='Packet Start')
        ax2.set_title('Signal with Packet Start Marker')
        ax2.set_xlabel('Samples')
        ax2.set_ylabel('Amplitude')
        ax2.legend()
        ax2.grid(True)

    if packet_markers:
        ax1.legend()

    plt.tight_layout()
    plt.show()

def save_vector(vector, output_path):
    """שומר את הוקטור כקובץ .mat"""
    sio.savemat(output_path, {'Y': vector})

def save_vector_wv(vector, output_path, sample_rate, normalize=False):
    """שומר את הוקטור כקובץ .wv באמצעות הסקריפט הקיים."""
    from vector_analyzer.mat_to_wv_converter import mat2wv

    # mat2wv יודע לקבל ישירות numpy array
    mat2wv(vector, output_path, sample_rate, bNormalize=normalize)

def generate_sample_packet(duration, sr, frequency, amplitude=1.0):
    """יוצר פקטה לדוגמה"""
    # np.linspace כולל כברירת מחדל את נקודת הסיום, מה שגורם לדגימה אחת
    # נוספת מעבר למספר המצופה (sr * duration). שימוש ב-endpoint=False מבטיח
    # שמספר הדגימות יהיה בדיוק sr * duration והמרווח ביניהן יהיה 1/sr.
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = amplitude * np.exp(2j * np.pi * frequency * t)
    return signal

def create_sample_packets():
    """יוצר 6 פקטות לדוגמה"""
    sr = 44100  # Sample rate
    duration = 1.0  # משך כל פקטה בשניות
    
    # יצירת 6 פקטות בתדרים שונים
    frequencies = [440, 880, 1320, 1760, 2200, 2640]  # Hz
    packets = []
    
    for i, freq in enumerate(frequencies):
        packet = generate_sample_packet(duration, sr, freq)
        packets.append(packet)
        
        # שמירת הפקטה
        sio.savemat(f'data/packet_{i+1}.mat', {'Y': packet})
    
    return packets 

def find_packet_start(signal, template=None, threshold_ratio=0.2, window_size=None):
    """מוצא את תחילת הפקטה באות.

    אם ניתנת תבנית, מתבצעת קורלציה ומחושב המיקום המתאים ביותר. אחרת
    נבנה מעטפת אנרגיה מוחלקת ועל פיה נקבע מיקום תחילת הפקטה.``threshold_ratio``
    מגדיר את העוצמה היחסית (בין רמת הרעש למקסימום) שמעליה נחשב שהחלה הפקטה.
    """

    if template is not None:
        correlation = np.correlate(np.abs(signal), np.abs(template), mode="valid")
        return int(np.argmax(correlation))

    # מעטפת אנרגיה עם החלקה
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
    מודד את המרווחים לפני ואחרי הפקטה
    
    Args:
        signal: האות המלא
        template: תבנית החיפוש
    
    Returns:
        pre_samples: מספר הסמפלים לפני הפקטה
        post_samples: מספר הסמפלים אחרי הפקטה
        packet_start: אינדקס תחילת הפקטה
    """
    packet_start = find_packet_start(signal, template)
    
    # חישוב המרווחים
    pre_samples = packet_start
    post_samples = len(signal) - packet_start - len(template) if template is not None else 0
    
    return pre_samples, post_samples, packet_start

def plot_packet_with_markers(signal, packet_start, template=None, title='Packet Analysis'):
    """
    מציג את הפקטה עם סימון של תחילת הפקטה האמיתית
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

    # הפוך את ציר התדרים והמטריצה לסדר יורד (חיובי משמאל, שלילי מימין)
    if f[0] < f[-1]:
        f = f[::-1]
        Sxx_db = Sxx_db[::-1, :]

    # המרת ציר זמן למיליסקנד
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
    # קביעת שנתות כל 0.1ms
    min_t, max_t = t_ms[0], t_ms[-1]
    ax1.set_xticks(np.arange(np.floor(min_t*10)/10, np.ceil(max_t*10)/10 + 0.1, 0.1))

    ax2.plot(sample_times_ms, np.abs(signal))
    ax2.set_xlabel('Time [ms]')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)
    ax2.set_xticks(np.arange(np.floor(sample_times_ms[0]*10)/10, np.ceil(sample_times_ms[-1]*10)/10 + 0.1, 0.1))

    # קו מקווקו ומדבקה
    packet_time_ms = packet_start / sample_rate * 1000
    line1 = ax1.axvline(packet_time_ms, color='r', linestyle='--')
    line2 = ax2.axvline(packet_time_ms, color='r', linestyle='--')
    # תווית אופקית (annotation)
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
        # עדכון תווית
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
    plt.subplots_adjust(bottom=0.18)  # מקום לכפתורים

    pcm = ax1.pcolormesh(t_ms, f / 1e6, Sxx_db, shading='nearest', cmap='viridis', vmin=vmin, vmax=vmax)
    ax1.set_ylim(ax1.get_ylim()[::-1])
    ax1.set_title("Use 'g'/'r' to select a line, drag to move, Enter to finish")
    ax1.set_xlabel('Time [ms]')
    ax1.set_ylabel('Frequency [MHz]')
    ax1.grid(True)
    min_t, max_t = t_ms[0], t_ms[-1]
    ax1.set_xticks(np.arange(np.floor(min_t*10)/10, np.ceil(max_t*10)/10 + 0.1, 0.1))

    ax2.plot(sample_times_ms, np.abs(signal))
    ax2.set_xlabel('Time [ms]')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)
    ax2.set_xticks(np.arange(np.floor(sample_times_ms[0]*10)/10, np.ceil(sample_times_ms[-1]*10)/10 + 0.1, 0.1))

    start_time_ms = start_sample / sample_rate * 1000
    end_time_ms = end_sample / sample_rate * 1000
    start_line1 = ax1.axvline(start_time_ms, color='g', linestyle='--', linewidth=2)
    end_line1 = ax1.axvline(end_time_ms, color='r', linestyle='--', linewidth=1)
    start_line2 = ax2.axvline(start_time_ms, color='g', linestyle='--', linewidth=2)
    end_line2 = ax2.axvline(end_time_ms, color='r', linestyle='--', linewidth=1)
    # תוויות אופקיות
    label_start = ax1.annotate(f"{start_time_ms*1000:.0f} μs", xy=(start_time_ms, ax1.get_ylim()[0]),
                        xytext=(0, -18), textcoords='offset points', ha='center', va='top', color='green', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
    label_end = ax1.annotate(f"{end_time_ms*1000:.0f} μs", xy=(end_time_ms, ax1.get_ylim()[0]),
                        xytext=(0, -18), textcoords='offset points', ha='center', va='top', color='red', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
    
    state = {'drag': None, 'active': 'start', 'start': start_sample, 'end': end_sample}
    # תווית מרווח (מעל הפינה הימנית העליונה)
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
        # עדכון תוויות הזמן לראש הקו (y התחתון)
        y_bottom = ax1.get_ylim()[0]
        label_start.set_position((start_time_ms, y_bottom))
        label_start.set_text(f"{start_time_ms*1000:.0f} μs")
        label_end.set_position((end_time_ms, y_bottom))
        label_end.set_text(f"{end_time_ms*1000:.0f} μs")
        # עדכון תווית המרווח
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

    # כפתורי הזזה - מיקום חדש במרכז מתחת לגרף התחתון
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

