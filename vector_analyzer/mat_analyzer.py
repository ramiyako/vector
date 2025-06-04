import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk
import threading
from scipy import signal as sig
from utils import normalize_spectrogram

class MatAnalyzerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MAT File Analyzer")
        self.root.geometry("400x400")
        
        # יצירת ממשק משתמש
        self.create_widgets()
        
    def create_widgets(self):
        # כפתור לבחירת קובץ
        self.file_button = ttk.Button(self.root, text="בחר קובץ MAT", command=self.load_file)
        self.file_button.pack(pady=10)
        
        # שדות קלט
        ttk.Label(self.root, text="קצב דגימה (MHz):").pack()
        self.sample_rate = ttk.Entry(self.root)
        self.sample_rate.pack()
        self.sample_rate.insert(0, "40")  # ברירת מחדל
        
        ttk.Label(self.root, text="תדר מרכזי (MHz):").pack()
        self.center_freq = ttk.Entry(self.root)
        self.center_freq.pack()
        self.center_freq.insert(0, "5200")  # ברירת מחדל
        
        ttk.Label(self.root, text="רוחב חלון (דגימות):").pack()
        self.window_size = ttk.Entry(self.root)
        self.window_size.pack()
        self.window_size.insert(0, "1024")  # ברירת מחדל
        
        ttk.Label(self.root, text="חפיפה (%):").pack()
        self.overlap = ttk.Entry(self.root)
        self.overlap.pack()
        self.overlap.insert(0, "50")  # ברירת מחדל
        
        # כפתורים להצגת גרפים
        self.plot_spectrum_button = ttk.Button(self.root, text="הצג ספקטרום", command=self.plot_spectrum)
        self.plot_spectrum_button.pack(pady=5)
        
        self.plot_spectrogram_button = ttk.Button(self.root, text="הצג ספקטרוגרמה", command=self.plot_spectrogram)
        self.plot_spectrogram_button.pack(pady=5)
        
        # תווית סטטוס
        self.status_label = ttk.Label(self.root, text="")
        self.status_label.pack(pady=10)
        
    def load_file(self):
        self.file_path = filedialog.askopenfilename(
            filetypes=[("MAT files", "*.mat")],
            initialdir=r"C:\Users\rami\vector_analyzer\vectors"
        )
        if self.file_path:
            self.status_label.config(text=f"נבחר קובץ: {Path(self.file_path).name}")
            
    def plot_spectrogram(self):
        if not hasattr(self, 'file_path'):
            self.status_label.config(text="יש לבחור קובץ תחילה")
            return
            
        try:
            sample_rate = float(self.sample_rate.get()) * 1e6
            center_freq = float(self.center_freq.get()) * 1e6
            window_size = int(self.window_size.get())
            overlap_percent = float(self.overlap.get())
            
            # חישוב חפיפה בדגימות
            noverlap = int(window_size * overlap_percent / 100)
            
        except ValueError:
            self.status_label.config(text="יש להזין מספרים תקינים")
            return
            
        threading.Thread(target=self._analyze_and_plot_spectrogram,
                        args=(sample_rate, center_freq, window_size, noverlap),
                        daemon=True).start()
        
    def _analyze_and_plot_spectrogram(self, sample_rate, center_freq, window_size, noverlap):
        try:
            self.status_label.config(text="מחשב ספקטרוגרמה...")
            
            # טעינת הנתונים
            mat_data = sio.loadmat(self.file_path)
            signal = mat_data['Y'].squeeze()
            
            # בדיקות לאות ריק או מלא באפסים
            if np.all(signal == 0):
                self.status_label.config(text="האות מכיל רק אפסים")
                return
            if np.max(np.abs(signal)) == 0:
                self.status_label.config(text="האות ריק או קבוע")
                return
                
            # הסרת ממוצע (DC)
            signal = signal - np.mean(signal)
            
            # יצירת אות קומפלקס מ-I ו-Q
            complex_signal = signal[::2] + 1j * signal[1::2]
            
            # חישוב הספקטרוגרמה
            f, t, Sxx = sig.spectrogram(complex_signal, 
                                      fs=sample_rate,
                                      window='hann',
                                      nperseg=window_size,
                                      noverlap=noverlap,
                                      return_onesided=False)
            
            # נרמול הספקטרוגרמה
            Sxx_db, vmin, vmax = normalize_spectrogram(Sxx)
            
            # הזזת התדרים
            f = np.fft.fftshift(f)
            Sxx_db = np.fft.fftshift(Sxx_db, axes=0)
            
            # הוספת תדר מרכזי
            f = f + center_freq
            
            # יצירת מפת צבעים מותאמת
            default_colormap = plt.cm.viridis
            
            # הצגת הספקטרוגרמה
            plt.figure(figsize=(12, 6))
            plt.pcolormesh(f/1e6, t*1e3, Sxx_db.T,
                          shading='gouraud',
                          cmap=default_colormap,
                          vmin=vmin,
                          vmax=vmax)
            plt.colorbar(label='עוצמה [dB]')
            plt.title(f"ספקטרוגרמה - {Path(self.file_path).name}")
            plt.xlabel("תדר [MHz]")
            plt.ylabel("זמן [msec]")
            plt.grid(True)
            
            # הוספת רשת משנית
            plt.grid(True, which='minor', alpha=0.2)
            plt.grid(True, which='major', alpha=0.5)
            
            self.status_label.config(text="הספקטרוגרמה מוכנה")
            plt.show()
            
        except Exception as e:
            self.status_label.config(text=f"שגיאה: {str(e)}")
            
    def plot_spectrum(self):
        if not hasattr(self, 'file_path'):
            self.status_label.config(text="יש לבחור קובץ תחילה")
            return
            
        try:
            sample_rate = float(self.sample_rate.get()) * 1e6
            center_freq = float(self.center_freq.get()) * 1e6
        except ValueError:
            self.status_label.config(text="יש להזין מספרים תקינים")
            return
            
        threading.Thread(target=self._analyze_and_plot_spectrum,
                        args=(sample_rate, center_freq),
                        daemon=True).start()
        
    def _analyze_and_plot_spectrum(self, sample_rate, center_freq):
        try:
            self.status_label.config(text="טוען נתונים...")
            
            # טעינת הנתונים
            mat_data = sio.loadmat(self.file_path)
            signal = mat_data['Y'].squeeze()

            # בדיקות לאות ריק או מלא באפסים
            if np.all(signal == 0):
                self.status_label.config(text="האות מכיל רק אפסים")
                return
            if np.max(np.abs(signal)) == 0:
                self.status_label.config(text="האות ריק או קבוע")
                return

            # הסרת ממוצע (DC)
            signal = signal - np.mean(signal)
            
            # יצירת אות קומפלקס מ-I ו-Q
            complex_signal = signal[::2] + 1j * signal[1::2]
            
            # חישוב ספקטרום
            N = len(complex_signal)
            spectrum = np.fft.fftshift(np.fft.fft(complex_signal))
            freq = np.fft.fftshift(np.fft.fftfreq(N, 1/sample_rate))
            
            # טיפול בערכים אפסיים לפני log
            spectrum_db = 20 * np.log10(np.abs(spectrum) + 1e-12)
            spectrum_db -= np.max(spectrum_db)  # נרמול למקסימום 0 dB
            
            # הוספת תדר מרכזי
            freq = freq + center_freq
            
            # יצירת גרף ספקטרום
            plt.figure(figsize=(12, 6))
            plt.plot(freq/1e6, spectrum_db)
            plt.title(f"ספקטרום - {Path(self.file_path).name}")
            plt.xlabel("תדר [MHz]")
            plt.ylabel("עוצמה [dB]")
            plt.grid(True)
            
            # הוספת רשת משנית
            plt.grid(True, which='minor', alpha=0.2)
            plt.grid(True, which='major', alpha=0.5)
            
            self.status_label.config(text="הספקטרום מוכן")
            plt.show()
            
        except Exception as e:
            self.status_label.config(text=f"שגיאה: {str(e)}")
            
    def run(self):
        self.root.mainloop()

def main():
    app = MatAnalyzerGUI()
    app.run()

if __name__ == "__main__":
    main() 
