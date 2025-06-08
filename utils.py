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
    """טוען פקטה מקובץ .mat ומחזיר תמיד np.complex64 או np.float32"""
    data = sio.loadmat(file_path)
    y = data['Y'].flatten()
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
    """מציג ספקטוגרמה עם ציר תדר ב-MHz, עם אפשרות לסמן מיקומים של פקטות.

    Parameters
    ----------
    f, t, Sxx : arrays
        תוצרי ``create_spectrogram``.
    center_freq : float, optional
        תדר מרכזי ששימש בחישוב הספקטרוגרמה. הציר מוצג תמיד בתדר מוחלט
        (MHz), ולכן הפרמטר נדרש רק לחישובים פנימיים.
    title : str
        כותרת הגרף.
    packet_start : int or None
        דגימה המסמנת את תחילת הפקטה (לשרטוט בקו אנכי).
    sample_rate : float or None
        קצב הדגימה של האות לצורך המרת מיקום הפקטה לזמן.
    signal : array or None
        האות המקורי להצגה בחלון התחתון.
    packet_markers : list
        רשימת סמנים. כל סמן הוא ``(time, freq, label[, style])`` כאשר ``style``
        הוא סימון matplotlib לתצוגה (למשל ``'x'``). אם ``style`` לא סופק
        ייבחר סגנון אוטומטי.
    freq_ranges : list of (f_low, f_high), optional
        אם מצוין, מתווה הספקטרוגרמה בעזרת ציר שבור המאפשר דילוג על תחומי
        תדר ללא עניין.
    """
    Sxx_db, vmin, vmax = normalize_spectrogram(Sxx)

    freq_axis = f / 1e6

    if freq_ranges:
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
    else:
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
        seen_labels = set()
        for idx, marker in enumerate(packet_markers):
            if len(marker) >= 4:
                tm, freq, label, style = marker[:4]
            elif len(marker) == 3:
                tm, freq, label = marker
                style = marker_styles[idx % len(marker_styles)]
            else:
                tm, freq = marker[:2]
                label = None
                style = marker_styles[idx % len(marker_styles)]
            show_label = label if label not in seen_labels else "_nolegend_"
            seen_labels.add(label)
            ax1.plot(tm, freq / 1e6, style, label=show_label)
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
    """מציג ספקטוגרמה עם קו ניתן להזזה לתיקון מיקום תחילת הפקטה.

    המשתמש יכול לגרור את הקו האדום ובסיום (סגירת החלון) יוחזר המיקום החדש
    בדגימות.
    """

    f, t, Sxx = create_spectrogram(signal, sample_rate)
    Sxx_db, vmin, vmax = normalize_spectrogram(Sxx)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])

    ax1.pcolormesh(t, f/1e6, Sxx_db, shading='nearest', cmap='viridis', vmin=vmin, vmax=vmax)
    ax1.set_title('Spectrogram - drag the red line to adjust start')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Frequency [MHz]')
    ax1.grid(True)

    ax2.plot(np.abs(signal))
    ax2.set_xlabel('Samples')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)

    line1 = ax1.axvline(packet_start / sample_rate, color='r', linestyle='--')
    line2 = ax2.axvline(packet_start, color='r', linestyle='--')

    state = {'drag': False, 'value': packet_start}

    def _press(event):
        if event.inaxes not in (ax1, ax2):
            return
        x = event.xdata * sample_rate if event.inaxes is ax1 else event.xdata
        if abs(x - state['value']) < sample_rate * 0.01:
            state['drag'] = True

    def _move(event):
        if not state['drag'] or event.inaxes not in (ax1, ax2):
            return
        x = int(event.xdata * sample_rate) if event.inaxes is ax1 else int(event.xdata)
        x = max(0, min(len(signal)-1, x))
        state['value'] = x
        line1.set_xdata(x / sample_rate)
        line2.set_xdata(x)
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
    """Interactive GUI to adjust packet start (green) and end (red) positions.

    Parameters
    ----------
    signal : array-like
        The full signal.
    sample_rate : float
        Sampling rate in Hz.
    start_sample : int, optional
        Initial start sample position.
    end_sample : int or None, optional
        Initial end sample position. Defaults to ``len(signal)``.

    Returns
    -------
    tuple of int
        The selected (start_sample, end_sample).
    """
    if end_sample is None:
        end_sample = len(signal)

    f, t, Sxx = create_spectrogram(signal, sample_rate)
    Sxx_db, vmin, vmax = normalize_spectrogram(Sxx)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])

    ax1.pcolormesh(t, f / 1e6, Sxx_db, shading='nearest', cmap='viridis', vmin=vmin, vmax=vmax)
    ax1.set_title('Drag the lines to set packet start (green) and end (red)')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Frequency [MHz]')
    ax1.grid(True)

    ax2.plot(np.abs(signal))
    ax2.set_xlabel('Samples')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)

    start_line1 = ax1.axvline(start_sample / sample_rate, color='g', linestyle='--')
    end_line1 = ax1.axvline(end_sample / sample_rate, color='r', linestyle='--')
    start_line2 = ax2.axvline(start_sample, color='g', linestyle='--')
    end_line2 = ax2.axvline(end_sample, color='r', linestyle='--')

    state = {'drag': None, 'start': start_sample, 'end': end_sample}

    def _press(event):
        if event.inaxes not in (ax1, ax2):
            return
        x = event.xdata * sample_rate if event.inaxes is ax1 else event.xdata
        if abs(x - state['start']) < sample_rate * 0.01:
            state['drag'] = 'start'
        elif abs(x - state['end']) < sample_rate * 0.01:
            state['drag'] = 'end'

    def _move(event):
        if state['drag'] is None or event.inaxes not in (ax1, ax2):
            return
        x = int(event.xdata * sample_rate) if event.inaxes is ax1 else int(event.xdata)
        x = max(0, min(len(signal) - 1, x))
        if state['drag'] == 'start':
            state['start'] = x
            start_line1.set_xdata(x / sample_rate)
            start_line2.set_xdata(x)
        else:
            state['end'] = x
            end_line1.set_xdata(x / sample_rate)
            end_line2.set_xdata(x)
        fig.canvas.draw_idle()

    def _release(event):
        state['drag'] = None

    cid_press = fig.canvas.mpl_connect('button_press_event', _press)
    cid_move = fig.canvas.mpl_connect('motion_notify_event', _move)
    cid_rel = fig.canvas.mpl_connect('button_release_event', _release)

    plt.tight_layout()
    plt.show()

    fig.canvas.mpl_disconnect(cid_press)
    fig.canvas.mpl_disconnect(cid_move)
    fig.canvas.mpl_disconnect(cid_rel)

    return state['start'], state['end']

