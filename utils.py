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