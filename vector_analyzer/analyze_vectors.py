import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from utils import get_sample_rate_from_mat

# הגדרות
CENTER_FREQ = 5230e6  # 5230 MHz
BANDWIDTH = 40e6  # 40 MHz
TARGET_CHANNELS = [5220e6, 5240e6]  # הערוצים המעניינים אותנו
CHANNEL_BANDWIDTH = 20e6  # רוחב סרט של כל ערוץ

def load_vector_data(file_path):
    """טעינת נתונים מקובץ MATLAB"""
    data = loadmat(file_path)
    return data['Y'].flatten()  # המרה לוקטור חד-ממדי

def analyze_channel(data, center_freq, sample_rate, bandwidth):
    """ניתוח ערוץ ספציפי"""
    # חישוב תדרי ה-FFT
    fft_size = len(data)
    freqs = np.fft.fftfreq(fft_size, 1/sample_rate) + center_freq
    
    # חישוב ה-FFT
    fft_data = np.fft.fft(data)
    fft_magnitude = np.abs(fft_data)
    
    # חיתוך לתדרים הרלוונטיים
    mask = (freqs >= center_freq - bandwidth/2) & (freqs <= center_freq + bandwidth/2)
    return freqs[mask], fft_magnitude[mask]

def plot_channel(freqs, magnitude, title, filename):
    """ציור גרף של ערוץ ושמירה לקובץ"""
    plt.figure(figsize=(10, 6))
    plt.plot(freqs/1e6, 20*np.log10(magnitude))  # המרה ל-dB
    plt.title(title)
    plt.xlabel('תדר (MHz)')
    plt.ylabel('עוצמה (dB)')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()  # סגירת החלון כדי לחסוך בזיכרון

def main():
    # טעינת הנתונים
    file_path = 'vectors/OS40-5600-40-air3-5230.mat'
    data = load_vector_data(file_path)
    sample_rate = get_sample_rate_from_mat(file_path) or 56e6
    
    # ניתוח כל ערוץ
    for channel_freq in TARGET_CHANNELS:
        freqs, magnitude = analyze_channel(data, channel_freq, sample_rate, CHANNEL_BANDWIDTH)
        filename = f'channel_{int(channel_freq/1e6)}MHz.png'
        plot_channel(freqs, magnitude, f'ספקטרום ערוץ {channel_freq/1e6}MHz', filename)
        print(f'הגרף נשמר בקובץ: {filename}')

if __name__ == '__main__':
    main() 
