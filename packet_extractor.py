import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import tkinter as tk
from tkinter import filedialog, ttk
from utils import create_spectrogram, plot_spectrogram, get_sample_rate_from_mat, normalize_spectrogram

class PacketExtractor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Packet Extractor")
        self.root.geometry("400x200")
        
        self.signal = None
        self.sample_rate = None
        self.start_sample = None
        self.end_sample = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # כפתור לבחירת קובץ
        self.file_button = ttk.Button(self.root, text="בחר קובץ MAT", command=self.load_file)
        self.file_button.pack(pady=10)
        
        # שדות קלט
        ttk.Label(self.root, text="קצב דגימה (MHz):").pack()
        self.sample_rate_var = tk.StringVar()
        self.sample_rate_entry = ttk.Entry(self.root, textvariable=self.sample_rate_var, state='readonly')
        self.sample_rate_entry.pack()
        
        # כפתור להצגת ספקטוגרמה
        self.plot_button = ttk.Button(self.root, text="הצג ספקטוגרמה", command=self.show_spectrogram)
        self.plot_button.pack(pady=10)
        
        # כפתור לשמירת הפקטה
        self.save_button = ttk.Button(self.root, text="שמור פקטה", command=self.save_packet)
        self.save_button.pack(pady=10)
        
    def load_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("MAT files", "*.mat")],
            initialdir="data"
        )
        if file_path:
            try:
                # טעינת הקובץ
                data = sio.loadmat(file_path)
                self.signal = data['Y'].flatten()
                
                # קבלת קצב דגימה
                self.sample_rate = get_sample_rate_from_mat(file_path)
                if self.sample_rate:
                    self.sample_rate_var.set(str(self.sample_rate / 1e6))
                else:
                    self.sample_rate_var.set("56")  # ברירת מחדל
                    self.sample_rate = 56e6
                    
            except Exception as e:
                tk.messagebox.showerror("שגיאה", f"שגיאה בטעינת הקובץ: {e}")
                
    def show_spectrogram(self):
        if self.signal is None:
            tk.messagebox.showerror("שגיאה", "יש לבחור קובץ תחילה")
            return
            
        try:
            sample_rate = self.sample_rate
            f, t, Sxx = create_spectrogram(self.signal, sample_rate)
            Sxx_db, vmin, vmax = normalize_spectrogram(Sxx)
            
            # יצירת חלון חדש לספקטוגרמה
            fig, ax = plt.subplots(figsize=(12, 6))
            plt.subplots_adjust(bottom=0.2)
            
            # הצגת הספקטוגרמה
            im = ax.pcolormesh(t, f/1e6, Sxx_db, shading='nearest', cmap='viridis', vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, label='Power [dB]')
            ax.set_title("בחר את תחילת הפקטה")
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Frequency [MHz]')
            ax.grid(True)
            
            # הוספת RectangleSelector
            def line_select_callback(eclick, erelease):
                self.start_sample = int(eclick.xdata * sample_rate)
                self.end_sample = int(erelease.xdata * sample_rate)
                plt.close()
                
            rs = RectangleSelector(ax, line_select_callback,
                                 useblit=True,
                                 button=[1],
                                 minspanx=5, minspany=5,
                                 spancoords='pixels',
                                 interactive=True)
            
            plt.show()
            
        except Exception as e:
            tk.messagebox.showerror("שגיאה", f"שגיאה בהצגת הספקטוגרמה: {e}")
            
    def save_packet(self):
        if self.signal is None or self.start_sample is None or self.end_sample is None:
            tk.messagebox.showerror("שגיאה", "יש לבחור קובץ ולסמן את תחילת הפקטה")
            return
            
        try:
            # חיתוך הפקטה
            packet = self.signal[self.start_sample:self.end_sample]
            
            # שמירת הפקטה
            file_path = filedialog.asksaveasfilename(
                defaultextension=".mat",
                filetypes=[("MAT files", "*.mat")],
                initialdir="data"
            )
            
            if file_path:
                sio.savemat(file_path, {'Y': packet})
                tk.messagebox.showinfo("הצלחה", "הפקטה נשמרה בהצלחה")
                
        except Exception as e:
            tk.messagebox.showerror("שגיאה", f"שגיאה בשמירת הפקטה: {e}")
            
    def run(self):
        self.root.mainloop()

def main():
    app = PacketExtractor()
    app.run()

if __name__ == "__main__":
    main() 
