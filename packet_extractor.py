import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, ttk
from utils import get_sample_rate_from_mat, adjust_packet_bounds_gui
from utils import detect_packet_bounds

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

        # הצגת תחילת וסיום הפקטה שנבחרו
        ttk.Label(self.root, text="Start sample:").pack()
        self.start_var = tk.StringVar()
        self.start_entry = ttk.Entry(self.root, textvariable=self.start_var, state='readonly')
        self.start_entry.pack()

        ttk.Label(self.root, text="End sample:").pack()
        self.end_var = tk.StringVar()
        self.end_entry = ttk.Entry(self.root, textvariable=self.end_var, state='readonly')
        self.end_entry.pack()
        
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
                if 'Y' in data:
                    self.signal = data['Y'].flatten()
                else:
                    candidates = [k for k in data.keys() if not k.startswith('__')]
                    if len(candidates) == 1:
                        self.signal = data[candidates[0]].flatten()
                    else:
                        raise ValueError(f"לא נמצא משתנה מתאים בקובץ {file_path}. משתנים קיימים: {list(data.keys())}")
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
            if self.start_sample is None or self.end_sample is None:
                start_det, end_det = detect_packet_bounds(self.signal, sample_rate)
                buffer_samples = int(sample_rate // 1_000_000)
                start = max(0, start_det - buffer_samples)
                self.start_sample, self.end_sample = start, end_det
                self.detected_start = start_det
                self._pre_buffer = buffer_samples

            self.start_sample, self.end_sample = adjust_packet_bounds_gui(
                self.signal,
                sample_rate,
                self.start_sample,
                self.end_sample,
            )
            self.start_var.set(str(self.start_sample))
            self.end_var.set(str(self.end_sample))
        except Exception as e:
            tk.messagebox.showerror("שגיאה", f"שגיאה בהצגת הספקטוגרמה: {e}")
            
    def save_packet(self):
        if self.signal is None or self.start_sample is None or self.end_sample is None:
            tk.messagebox.showerror("שגיאה", "יש לבחור קובץ ולסמן את תחילת הפקטה")
            return
            
        try:
            # חיתוך הפקטה
            packet = self.signal[self.start_sample:self.end_sample]
            pre_samples = int(getattr(self, 'detected_start', self.start_sample) - self.start_sample)
            
            # שמירת הפקטה
            file_path = filedialog.asksaveasfilename(
                defaultextension=".mat",
                filetypes=[("MAT files", "*.mat")],
                initialdir="data"
            )
            
            if file_path:
                sio.savemat(file_path, {'Y': packet, 'pre_samples': pre_samples})
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
