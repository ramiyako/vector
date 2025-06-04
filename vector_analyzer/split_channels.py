import numpy as np
from scipy.io import loadmat, savemat
from scipy import signal
from utils import get_sample_rate_from_mat

# הגדרות
CENTER_FREQ = 5230e6  # 5230 MHz
TARGET_CHANNELS = [5220e6, 5240e6]  # הערוצים המעניינים אותנו
CHANNEL_BANDWIDTH = 20e6  # רוחב סרט של כל ערוץ

def load_original_data(file_path):
    """טעינת כל הנתונים מהקובץ המקורי"""
    return loadmat(file_path)

def filter_channel(data, center_freq, sample_rate, bandwidth):
    """סינון הנתונים לערוץ ספציפי"""
    # חישוב FFT של האות
    fft_size = len(data)
    freqs = np.fft.fftfreq(fft_size, 1/sample_rate) * sample_rate + CENTER_FREQ
    
    # חישוב ה-FFT
    fft_data = np.fft.fft(data)
    
    # יצירת מסכה לתדרים הרצויים
    mask = (freqs >= center_freq - bandwidth/2) & (freqs <= center_freq + bandwidth/2)
    
    # אפס את כל התדרים מחוץ לתחום הרצוי
    filtered_fft = np.zeros_like(fft_data, dtype=complex)
    filtered_fft[mask] = fft_data[mask]
    
    # שמירה על סימטריה של ה-FFT
    # מציאת התדרים השליליים המתאימים
    negative_freqs = np.fft.fftshift(freqs) < CENTER_FREQ
    positive_freqs = np.fft.fftshift(freqs) >= CENTER_FREQ
    
    # העתקת התדרים החיוביים לתדרים השליליים
    filtered_fft_shifted = np.fft.fftshift(filtered_fft)
    filtered_fft_shifted[negative_freqs] = np.conj(np.flip(filtered_fft_shifted[positive_freqs]))
    filtered_fft = np.fft.ifftshift(filtered_fft_shifted)
    
    # המרה חזרה לדומיין הזמן
    filtered_data = np.real(np.fft.ifft(filtered_fft))
    
    return filtered_data

def save_channel_data(original_data, filtered_signal, output_file):
    """שמירת הנתונים המסוננים בפורמט זהה לקובץ המקורי"""
    # יצירת עותק של כל המטה-דאטה
    output_data = {}
    for key in original_data:
        if not key.startswith('__'):
            output_data[key] = original_data[key]
    
    # עדכון הנתונים המסוננים
    output_data['Y'] = filtered_signal.reshape(-1, 1)
    
    # שמירה לקובץ
    savemat(output_file, output_data)

def main():
    # טעינת הקובץ המקורי
    original_file = 'vectors/OS40-5600-40-air3-5230.mat'
    original_data = load_original_data(original_file)
    input_signal = original_data['Y'].flatten()
    sample_rate = get_sample_rate_from_mat(original_file) or 56e6
    
    # עיבוד כל ערוץ
    for channel_freq in TARGET_CHANNELS:
        # סינון הנתונים לערוץ הספציפי
        filtered_signal = filter_channel(input_signal, channel_freq, sample_rate, CHANNEL_BANDWIDTH)
        
        # שמירה לקובץ חדש
        output_file = f'vectors/channel_{int(channel_freq/1e6)}MHz.mat'
        save_channel_data(original_data, filtered_signal, output_file)
        print(f'נשמר קובץ חדש: {output_file}')

if __name__ == '__main__':
    main() 