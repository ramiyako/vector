import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
from utils import get_sample_rate_from_mat

def display_spectrogram(input_dir, target_file):
    """
    מציג את הספקטרום של קובץ ספציפי
    
    Args:
        input_dir (str): תיקיית המקור של קבצי ה-.mat
        target_file (str): שם הקובץ להצגה
    """
    # יצירת גרף
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # טעינת הנתונים
    mat_file_path = os.path.join(input_dir, target_file)
    print(f"טוען את הקובץ: {target_file}")
    
    mat_data = sio.loadmat(mat_file_path)
    signal = mat_data['Y'].squeeze()
    
    # חישוב ספקטרום
    N = len(signal)
    spectrum = np.fft.fftshift(np.fft.fft(signal))
    sr = get_sample_rate_from_mat(mat_file_path) or 56e6
    freq = np.fft.fftshift(np.fft.fftfreq(N, 1/sr))
    
    # הצגת הספקטרום
    ax.plot(freq/1e6, 20*np.log10(np.abs(spectrum)), 
           color='b', 
           label=target_file,
           linewidth=2)
    
    # הגדרת הגרף
    ax.set_title(f'ספקטרום של {target_file} סביב 5200MHz')
    ax.set_xlabel('תדר [MHz]')
    ax.set_ylabel('עוצמה [dB]')
    ax.grid(True)
    ax.legend()
    
    # הגבלת טווח התדרים ל-5200MHz ± 100MHz
    ax.set_xlim(5100, 5300)
    
    # הצגת הגרף
    plt.tight_layout()
    plt.show()

def main():
    # תיקיית המקור של קבצי ה-.mat
    input_dir = "vectors"
    
    # שם הקובץ להצגה
    target_file = "OS40-5600-40-air3-5200.mat"
    
    # הצגת הספקטרום
    display_spectrogram(input_dir, target_file)

if __name__ == "__main__":
    main() 