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

def load_packet(file_path):
    """טוען פקטה מקובץ .mat"""
    data = sio.loadmat(file_path)
    return data['Y'].flatten()

def resample_signal(signal, orig_sr, target_sr):
    """משנה את קצב הדגימה של האות"""
    return librosa.resample(signal, orig_sr=orig_sr, target_sr=target_sr)

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

def plot_spectrogram(f, t, Sxx, center_freq=0, title='Spectrogram'):
    """מציג ספקטוגרמה"""
    Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-10)
    vmin = np.percentile(Sxx_db, 10)
    vmax = np.percentile(Sxx_db, 99)
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(t, (f - center_freq)/1e6 if center_freq else f, Sxx_db, shading='nearest', cmap='viridis', norm=Normalize(vmin=vmin, vmax=vmax))
    plt.colorbar(label='Power [dB]')
    plt.title(title)
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency offset [MHz]' if center_freq else 'Frequency [Hz]')
    plt.grid(True)
    if center_freq:
        plt.ylim(-30, 30)
        plt.axhline(y=-20, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=20, color='r', linestyle='--', alpha=0.5)
    plt.show()

def save_vector(vector, output_path):
    """שומר את הוקטור כקובץ .mat"""
    sio.savemat(output_path, {'Y': vector})

def generate_sample_packet(duration, sr, frequency, amplitude=1.0):
    """יוצר פקטה לדוגמה"""
    t = np.linspace(0, duration, int(sr * duration))
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