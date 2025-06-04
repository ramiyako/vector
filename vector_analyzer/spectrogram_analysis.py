import numpy as np
from scipy.io import loadmat
from utils import get_sample_rate_from_mat
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def plot_spectrum(data, sample_rate, center_freq, title):
    """ציור ספקטרום של האות"""
    # חישוב FFT
    fft_size = len(data)
    freqs = np.fft.fftfreq(fft_size, 1/sample_rate) * sample_rate + center_freq
    fft_data = np.fft.fft(data)
    
    # חישוב ספקטרום הכוח
    spectrum = 20 * np.log10(np.abs(fft_data))
    
    # ציור
    plt.figure(figsize=(12, 6))
    plt.plot((freqs - center_freq)/1e6, spectrum)
    plt.title(f'Spectrum - {title}')
    plt.xlabel('Frequency Offset (MHz)')
    plt.ylabel('Power (dB)')
    plt.grid(True)
    plt.xlim(-30, 30)  # הצגת 60MHz סביב התדר המרכזי
    plt.axvline(x=-20, color='r', linestyle='--', alpha=0.5)  # סימון רוחב הסרט
    plt.axvline(x=20, color='r', linestyle='--', alpha=0.5)
    plt.savefig(f'spectrum_{title}.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_spectrogram(data, sample_rate, center_freq, title):
    """יצירת ספקטוגרמה של האות"""
    # הגדרות לספקטוגרמה - הקטנת הפרמטרים לחיסכון בזיכרון
    window_size = 1024
    overlap = window_size // 2
    nfft = 1024
    
    # חישוב הספקטוגרמה
    freqs, times, Sxx = signal.spectrogram(
        data,
        fs=sample_rate,
        window='hann',
        nperseg=window_size,
        noverlap=overlap,
        nfft=nfft,
        return_onesided=False,
        detrend=False
    )
    
    # הזזת התדרים לתדר המרכזי
    freqs = np.fft.fftshift(freqs) + center_freq
    Sxx = np.fft.fftshift(Sxx, axes=0)
    
    # חישוב הספקטרום בדציבלים
    Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-10)
    
    # ציור הספקטוגרמה
    plt.figure(figsize=(15, 8))
    plt.pcolormesh(times, (freqs - center_freq)/1e6, Sxx_db, 
                  shading='nearest', cmap='viridis',
                  norm=Normalize(vmin=np.percentile(Sxx_db, 10),
                               vmax=np.percentile(Sxx_db, 99)))
    
    plt.colorbar(label='Power (dB)')
    plt.title(f'Spectrogram - {title}')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency Offset (MHz)')
    plt.ylim(-30, 30)  # הצגת 60MHz סביב התדר המרכזי
    
    # הוספת קווים לסימון רוחב הסרט
    plt.axhline(y=-20, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=20, color='r', linestyle='--', alpha=0.5)
    
    plt.grid(True)
    plt.savefig(f'spectrogram_{title}.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_signal(signal, sample_rate, center_freq, title):
    """ניתוח אות - ספקטרום וספקטוגרמה"""
    # חישוב ספקטרום
    plot_spectrum(signal, sample_rate, center_freq, title)
    
    # חישוב ספקטוגרמה
    create_spectrogram(signal, sample_rate, center_freq, title)
    print(f'נשמרו גרפים לאות: {title}')

def main():
    # טעינת הקובץ המקורי
    original_file = 'vectors/OS40-5600-40-air3-5230.mat'
    original_data = loadmat(original_file)
    original_signal = original_data['Y'].flatten()
    original_sr = get_sample_rate_from_mat(original_file) or 56e6

    # ניתוח הקובץ המקורי
    analyze_signal(original_signal, original_sr, 5230e6, 'original')
    
    # ניתוח הערוצים המסוננים
    channels = [5220, 5240]
    for freq in channels:
        # טעינת הקובץ
        path = f'vectors/channel_{freq}MHz.mat'
        data = loadmat(path)
        signal = data['Y'].flatten()
        sr = get_sample_rate_from_mat(path) or original_sr

        # ניתוח הערוץ
        analyze_signal(signal, sr, 5230e6, f'{freq}MHz')

if __name__ == '__main__':
    main() 
