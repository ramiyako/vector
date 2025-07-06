import os
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, BooleanVar
from utils import (
    load_packet,
    load_packet_info,
    resample_signal,
    create_spectrogram,
    plot_spectrogram,
    save_vector,
    get_sample_rate_from_mat,
    apply_frequency_shift,
    compute_freq_ranges,
)

MAX_PACKETS = 6
TARGET_SAMPLE_RATE = 56e6  # 56 MHz final sample rate

class PacketConfig:
    def __init__(self, parent, idx, file_choices):
        self.frame = ttk.LabelFrame(parent, text=f"Packet {idx+1}")
        self.frame.pack(fill=tk.X, padx=5, pady=5)

        self.sample_rate = None

        # File selection
        ttk.Label(self.frame, text="Select file:").grid(row=0, column=0, sticky=tk.W)
        self.file_var = tk.StringVar()
        self.file_menu = ttk.Combobox(self.frame, textvariable=self.file_var, values=file_choices, state="readonly")
        self.file_menu.grid(row=0, column=1, sticky=tk.W)
        if file_choices:
            self.file_var.set(file_choices[0])
            self.file_menu.bind('<<ComboboxSelected>>', self.on_file_selected)

        # Show spectrogram button
        btn = ttk.Button(self.frame, text="Show spectrogram", command=self.show_spectrogram)
        btn.grid(row=0, column=2, padx=5)

        # Analyze packet button
        btn = ttk.Button(self.frame, text="Analyze packet", command=self.analyze_packet)
        btn.grid(row=0, column=3, padx=5)

        # Sample rate (MHz)
        ttk.Label(self.frame, text="Sample Rate (MHz):").grid(row=1, column=0, sticky=tk.W)
        self.sr_var = tk.StringVar(value="")
        self.sr_entry = ttk.Entry(self.frame, textvariable=self.sr_var, width=10)
        self.sr_entry.grid(row=1, column=1, sticky=tk.W)

        # קרא קצב דגימה עבור הקובץ הראשון
        if file_choices:
            self.on_file_selected()

        # Frequency shift (MHz)
        ttk.Label(self.frame, text="Frequency shift (MHz):").grid(row=2, column=0, sticky=tk.W)
        self.freq_shift_var = tk.StringVar(value="0")
        ttk.Entry(self.frame, textvariable=self.freq_shift_var, width=10).grid(row=2, column=1, sticky=tk.W)

        # Period (ms)
        ttk.Label(self.frame, text="Period (ms):").grid(row=3, column=0, sticky=tk.W)
        self.period_var = tk.StringVar(value="100")
        ttk.Entry(self.frame, textvariable=self.period_var, width=10).grid(row=3, column=1, sticky=tk.W)

        # Pre-packet samples
        ttk.Label(self.frame, text="Pre-packet samples:").grid(row=4, column=0, sticky=tk.W)
        self.pre_samples_var = tk.StringVar(value="0")
        ttk.Entry(self.frame, textvariable=self.pre_samples_var, width=10).grid(row=4, column=1, sticky=tk.W)

        # Start time offset (ms) relative to first packet
        ttk.Label(self.frame, text="Start time offset (ms):").grid(row=5, column=0, sticky=tk.W)
        self.start_time_var = tk.StringVar(value="0")
        ttk.Entry(self.frame, textvariable=self.start_time_var, width=10).grid(row=5, column=1, sticky=tk.W)

    def on_file_selected(self, event=None):
        """מעדכן את קצב הדגימה אוטומטית כשנבחר קובץ"""
        file_path = self.file_var.get()
        if file_path:
            sample_rate = get_sample_rate_from_mat(file_path)
            if sample_rate:
                self.sample_rate = sample_rate
                self.sr_var.set(str(sample_rate / 1e6))
            else:
                self.sample_rate = 56e6
                self.sr_var.set("56")

    def get_config(self):
        try:
            sr_value = float(self.sr_var.get()) * 1e6
        except ValueError:
            sr_value = self.sample_rate or TARGET_SAMPLE_RATE

        return {
            'file': self.file_var.get(),
            'sample_rate': sr_value,
            'freq_shift': float(self.freq_shift_var.get()) * 1e6,  # MHz to Hz
            'period': float(self.period_var.get()) / 1000.0,  # ms to seconds
            'pre_samples': int(self.pre_samples_var.get()),
            'start_time': float(self.start_time_var.get()) / 1000.0,  # ms to seconds
        }

    def show_spectrogram(self):
        from utils import load_packet, create_spectrogram, plot_spectrogram
        import numpy as np
        file_path = self.file_var.get()
        try:
            y = load_packet(file_path)
            sample_rate = self.sample_rate
            if sample_rate is None:
                from tkinter import messagebox
                messagebox.showerror("Error", "Sample rate not found")
                return
            f, t, Sxx = create_spectrogram(y, sample_rate, time_resolution_us=1)
            plot_spectrogram(f, t, Sxx, title=f"Spectrogram of {file_path}", sample_rate=sample_rate, signal=y)
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Error", f"Error showing spectrogram: {e}")

    def analyze_packet(self):
        from utils import load_packet, find_packet_start, measure_packet_timing, adjust_packet_start_gui
        file_path = self.file_var.get()
        try:
            y = load_packet(file_path)
            packet_start = find_packet_start(y)
            pre_samples, _, _ = measure_packet_timing(y)

            sample_rate = self.sample_rate
            if sample_rate is None:
                from tkinter import messagebox
                messagebox.showerror("Error", "Sample rate not found")
                return

            packet_start = adjust_packet_start_gui(y, sample_rate, packet_start)
            pre_samples = packet_start
            self.pre_samples_var.set(str(pre_samples))

            f, t, Sxx = create_spectrogram(y, sample_rate, time_resolution_us=1)
            plot_spectrogram(f, t, Sxx, title=f"Packet Analysis - {file_path}", packet_start=packet_start, sample_rate=sample_rate, signal=y)
            
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Error", f"Error analyzing packet: {e}")

class VectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Packet Vector Generator")
        self.root.geometry("500x700")
        self.packet_frames = []
        self.packet_configs = []
        self.packet_count = tk.IntVar(value=1)
        self.normalize = BooleanVar(value=True)

        # Packet files
        self.packet_files = [f"data/{f}" for f in os.listdir('data') if f.endswith('.mat')]
        if not self.packet_files:
            messagebox.showerror("Error", "No .mat files found in data directory")
            self.root.destroy()
            return

        self.build_gui()

    def build_gui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Vector length (seconds)
        ttk.Label(main_frame, text="Vector length (seconds):").pack(anchor=tk.W)
        self.vector_length_var = tk.StringVar(value="1")  # שינוי ל-1 שנייה כברירת מחדל
        ttk.Entry(main_frame, textvariable=self.vector_length_var, width=10).pack(anchor=tk.W, pady=2)

        # Number of packets
        ttk.Label(main_frame, text="Number of packets (1-6):").pack(anchor=tk.W, pady=(10,0))
        count_spin = ttk.Spinbox(main_frame, from_=1, to=MAX_PACKETS, textvariable=self.packet_count, width=5, command=self.update_packets)
        count_spin.pack(anchor=tk.W)
        count_spin.bind('<FocusOut>', lambda e: self.update_packets())
        count_spin.bind('<Return>', lambda e: self.update_packets())

        # Packets container
        self.packets_container = ttk.Frame(main_frame)
        self.packets_container.pack(fill=tk.BOTH, expand=True, pady=10)
        self.update_packets()

        # Normalize
        ttk.Checkbutton(main_frame, text="Normalize final vector", variable=self.normalize).pack(anchor=tk.W, pady=5)

        # Create vector buttons
        ttk.Button(
            main_frame,
            text="Create MAT Vector",
            command=self.generate_mat_vector,
        ).pack(pady=5)
        ttk.Button(
            main_frame,
            text="Create WV Vector",
            command=self.generate_wv_vector,
        ).pack(pady=5)

    def update_packets(self):
        for pf in self.packet_frames:
            pf.frame.destroy()
        self.packet_frames = []
        self.packet_configs = []
        for i in range(self.packet_count.get()):
            pc = PacketConfig(self.packets_container, i, self.packet_files)
            self.packet_frames.append(pc)
            self.packet_configs.append(pc)

    def generate_mat_vector(self):
        self.generate_vector("mat")

    def generate_wv_vector(self):
        self.generate_vector("wv")

    def generate_vector(self, output_format="mat"):
        try:
            vector_length = float(self.vector_length_var.get())
            total_samples = int(vector_length * TARGET_SAMPLE_RATE)
            vector = np.zeros(total_samples, dtype=np.complex64)
            freq_shifts = []
            markers = []
            marker_styles = ['x', 'o', '^', 's', 'D', 'P', 'v', '1', '2', '3', '4']
            marker_colors = [f"C{i}" for i in range(10)]
            style_map = {}

            for idx, pc in enumerate(self.packet_configs):
                cfg = pc.get_config()
                y, pre_buf = load_packet_info(cfg['file'])
                base_name = os.path.splitext(os.path.basename(cfg['file']))[0]
                if base_name not in style_map:
                    idx_style = len(style_map) % len(marker_styles)
                    idx_color = len(style_map) % len(marker_colors)
                    style_map[base_name] = (marker_styles[idx_style], marker_colors[idx_color])
                marker_style, marker_color = style_map[base_name]
                
                # Resample ל-10MHz
                if cfg['sample_rate'] != TARGET_SAMPLE_RATE:
                    y = resample_signal(y, cfg['sample_rate'], TARGET_SAMPLE_RATE)
                
                # Frequency shift
                if cfg['freq_shift'] != 0:
                    y = apply_frequency_shift(y, cfg['freq_shift'], TARGET_SAMPLE_RATE)
                    freq_shifts.append(cfg['freq_shift'])
                else:
                    freq_shifts.append(0)
                
                # Period in samples
                period_samples = int(cfg['period'] * TARGET_SAMPLE_RATE)

                # Time offset of first insertion in samples relative to start of vector
                start_offset = max(0, int(round(cfg['start_time'] * TARGET_SAMPLE_RATE)) - pre_buf)

                # Insert packet at the specified offset and every period thereafter
                for start in range(start_offset, total_samples, period_samples):
                    end = start + len(y)
                    if end > total_samples:
                        break  # לא להכניס מופע חתוך, אבל כן להכניס את כל המופעים התקינים
                    vector[start:end] += y
                    markers.append(
                        (
                            (start + pre_buf) / TARGET_SAMPLE_RATE,
                            cfg['freq_shift'],
                            base_name,
                            marker_style,
                            marker_color,
                        )
                    )

            # Debug
            print("Vector abs sum:", np.abs(vector).sum())
            print("Vector max:", np.abs(vector).max())
            print("Vector min:", np.abs(vector).min())
            
            # Normalize
            if self.normalize.get():
                max_abs = np.abs(vector).max()
                if max_abs > 0:
                    vector = vector / max_abs
                    
            if output_format == "wv":
                from utils import save_vector_wv
                output_path = 'data/output_vector.wv'
                save_vector_wv(vector, output_path, TARGET_SAMPLE_RATE)
            else:
                save_vector(vector, 'data/output_vector.mat')
            center_freq = 0
            if freq_shifts:
                center_freq = (min(freq_shifts) + max(freq_shifts)) / 2
            f, t, Sxx = create_spectrogram(
                vector, TARGET_SAMPLE_RATE, center_freq=center_freq, time_resolution_us=1
            )
            # נקה את freq_shifts מערכים לא חוקיים
            clean_freq_shifts = []
            for val in freq_shifts:
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    clean_freq_shifts.append(val)
                else:
                    print(f"אזהרה: ערך freq_shift לא חוקי: {val} ({type(val)})")
            ranges = compute_freq_ranges(clean_freq_shifts)
            if ranges:
                try:
                    ranges = list(ranges)
                    ranges = [tuple(r) for r in ranges]
                except Exception as e:
                    import warnings
                    warnings.warn(f"ranges לא תקין: {e}. תוצג כל הספקטרוגרמה.")
                    ranges = None
            plot_spectrogram(
                f,
                t,
                Sxx,
                title='Final Vector Spectrogram',
                center_freq=center_freq,
                packet_markers=markers,
                freq_ranges=ranges,
                show_colorbar=False,
            )
            messagebox.showinfo(
                "Success", f"Vector created and saved successfully as {output_format.upper()}"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Error: {e}")

def main():
    root = tk.Tk()
    app = VectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 
