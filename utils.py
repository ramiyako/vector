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
    """מחלץ את קצב הדגימה מקובץ .mat"""
    data = sio.loadmat(file_path)
    if 'X_delta' in data:
        return 1.0 / float(data['X_delta'])
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

def create_spectrogram(sig, sr, center_freq=0):
    """יוצר ספקטוגרמה מהאות, תומך גם ב-center_freq"""
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

def find_packet_start(signal, template=None, threshold=0.8):
    """
    מוצא את תחילת הפקטה האמיתית באות
    
    Args:
        signal: האות המלא
        template: תבנית החיפוש (אם None, משתמש בשינוי אנרגיה)
        threshold: סף הקורלציה או האנרגיה
    
    Returns:
        index: האינדקס של תחילת הפקטה
    """
    if template is not None:
        # חיפוש באמצעות קורלציה
        correlation = np.correlate(np.abs(signal), np.abs(template), mode='valid')
        return np.argmax(correlation)
    else:
        # חיפוש באמצעות שינוי אנרגיה
        energy = np.abs(signal) ** 2
        energy_diff = np.diff(energy)
        return np.argmax(energy_diff > threshold * np.max(energy_diff))

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