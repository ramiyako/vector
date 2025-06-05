import os
import numpy as np
import scipy.io as sio
import librosa
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib as mpl
from matplotlib.colors import Normalize

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
    """טוען פקטה מקובץ .mat.

    כדי להאיץ את הטעינה, אם קיים קובץ ``.npy`` תואם נוצרת טעינה מהירה
    באמצעות ``numpy.load``. אם הקובץ לא קיים, הוא נוצר אוטומטית לאחר
    טעינת ה-``.mat`` כך שהטעינות הבאות יהיו זריזות יותר.
    """
    cache_path = file_path + ".npy"
    if os.path.exists(cache_path):
        y = np.load(cache_path, mmap_mode="r")
    else:
        data = sio.loadmat(file_path)
        y = data['Y'].flatten()
        if np.iscomplexobj(y):
            y = y.astype(np.complex64)
        else:
            y = y.astype(np.float32)
        try:
            np.save(cache_path, y)
        except Exception:
            pass
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

def create_spectrogram(sig, sr, center_freq=0, max_samples=1_000_000):
    """יוצר ספקטוגרמה מהאות.

    אם האות ארוך במיוחד, מתבצע דילול מהיר כדי להאיץ את החישוב.
    """

    if len(sig) > max_samples:
        factor = int(np.ceil(len(sig) / max_samples))
        sig = sig[::factor]
        sr = sr / factor

    window_size = 1024
    overlap = window_size // 2
    nfft = 1024
    freqs, times, Sxx = signal.spectrogram(
        sig,
        fs=sr,
        window='hann',
        nperseg=window_size,
        noverlap=overlap,
        nfft=nfft,
        return_onesided=False,
        detrend=False
    )

    freqs = np.fft.fftshift(freqs) + center_freq
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

def plot_spectrogram(f, t, Sxx, center_freq=0, title='Spectrogram', packet_start=None, sample_rate=None, signal=None):
    """מציג ספקטוגרמה עם ציר תדר ב-MHz, וגרף אות עם סימון תחילת פקטה"""
    Sxx_db, vmin, vmax = normalize_spectrogram(Sxx)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])

    im = ax1.pcolormesh(t, (f - center_freq)/1e6 if center_freq else f/1e6, Sxx_db, shading='nearest', cmap='viridis', norm=Normalize(vmin=vmin, vmax=vmax))
    ax1.set_title(title)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Frequency [MHz]')
    ax1.grid(True)
    if center_freq:
        ax1.set_ylim(-30, 30)
        ax1.axhline(y=-20, color='r', linestyle='--', alpha=0.5)
        ax1.axhline(y=20, color='r', linestyle='--', alpha=0.5)
    if packet_start is not None and sample_rate is not None:
        packet_time = packet_start / sample_rate
        ax1.axvline(x=packet_time, color='r', linestyle='--', label='Packet Start')
    plt.colorbar(im, ax=ax1, label='Power [dB]')

    # גרף האות
    if signal is not None:
        ax2.plot(np.abs(signal))
        if packet_start is not None:
            ax2.axvline(x=packet_start, color='r', linestyle='--', label='Packet Start')
        ax2.set_title('Signal with Packet Start Marker')
        ax2.set_xlabel('Samples')
        ax2.set_ylabel('Amplitude')
        ax2.legend()
        ax2.grid(True)

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
