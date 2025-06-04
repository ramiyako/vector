import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from utils import get_sample_rate_from_mat

def plot_spectrum(data, sample_rate, center_freq, title):
    """ציור ספקטרום של אות"""
    # חישוב FFT
    fft_size = len(data)
    freqs = np.fft.fftfreq(fft_size, 1/sample_rate) + center_freq
    fft_data = np.fft.fft(data)
    
    # חישוב ספקטרום הכוח
    spectrum = 20 * np.log10(np.abs(fft_data))
    
    # ציור
    plt.figure(figsize=(12, 6))
    plt.plot((freqs - center_freq)/1e6, spectrum)
    plt.title(title)
    plt.xlabel('Frequency Offset (MHz)')
    plt.ylabel('Power (dB)')
    plt.grid(True)
    plt.xlim(-30, 30)  # הצגת 60MHz סביב התדר המרכזי
    plt.savefig(f'spectrum_{title}.png')
    plt.close()

def main():
    # טעינת הערוצים המסוננים
    channels = [5220, 5240]
    for freq in channels:
        # טעינת הקובץ
        path = f'vectors/channel_{freq}MHz.mat'
        data = loadmat(path)
        signal = data['Y'].flatten()
        sr = get_sample_rate_from_mat(path) or 56e6

        # ציור הספקטרום
        plot_spectrum(signal, sr, 5230e6, f'{freq}MHz')
        print(f'נשמר גרף ספקטרום: spectrum_{freq}MHz.png')

if __name__ == '__main__':
    main() 