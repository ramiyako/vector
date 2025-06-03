import os
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, BooleanVar
from utils import load_packet, resample_signal, create_spectrogram, plot_spectrogram, save_vector

MAX_PACKETS = 6

class PacketConfig:
    def __init__(self, parent, idx, file_choices):
        self.frame = ttk.LabelFrame(parent, text=f"Packet {idx+1}")
        self.frame.pack(fill=tk.X, padx=5, pady=5)

        # File selection
        ttk.Label(self.frame, text="Select file:").grid(row=0, column=0, sticky=tk.W)
        self.file_var = tk.StringVar()
        self.file_menu = ttk.Combobox(self.frame, textvariable=self.file_var, values=file_choices, state="readonly")
        self.file_menu.grid(row=0, column=1, sticky=tk.W)
        if file_choices:
            self.file_var.set(file_choices[0])

        # Show spectrogram button
        btn = ttk.Button(self.frame, text="Show spectrogram", command=self.show_spectrogram)
        btn.grid(row=0, column=2, padx=5)

        # Sample rate (MHz)
        ttk.Label(self.frame, text="Sample Rate (MHz):").grid(row=1, column=0, sticky=tk.W)
        self.sr_var = tk.StringVar(value="56")
        ttk.Entry(self.frame, textvariable=self.sr_var, width=10).grid(row=1, column=1, sticky=tk.W)

        # Frequency shift (MHz)
        ttk.Label(self.frame, text="Frequency shift (MHz):").grid(row=2, column=0, sticky=tk.W)
        self.freq_shift_var = tk.StringVar(value="0")
        ttk.Entry(self.frame, textvariable=self.freq_shift_var, width=10).grid(row=2, column=1, sticky=tk.W)

        # Period (ms)
        ttk.Label(self.frame, text="Period (ms):").grid(row=3, column=0, sticky=tk.W)
        self.period_var = tk.StringVar(value="100")
        ttk.Entry(self.frame, textvariable=self.period_var, width=10).grid(row=3, column=1, sticky=tk.W)

    def get_config(self):
        return {
            'file': self.file_var.get(),
            'sample_rate': float(self.sr_var.get()) * 1e6,  # MHz to Hz
            'freq_shift': float(self.freq_shift_var.get()) * 1e6,  # MHz to Hz
            'period': float(self.period_var.get()) / 1000.0  # ms to seconds
        }

    def show_spectrogram(self):
        from utils import load_packet, create_spectrogram, plot_spectrogram
        import numpy as np
        file_path = self.file_var.get()
        try:
            y = load_packet(file_path)
            print(f"First values of {file_path}: {y[:10]}")
            f, t, Sxx = create_spectrogram(y, 56e6)  # default 56MHz
            plot_spectrogram(f, t, Sxx, title=f"Spectrogram of {file_path}")
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Error", f"Error showing spectrogram: {e}")

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
        self.vector_length_var = tk.StringVar(value="10")
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

        # Create vector button
        ttk.Button(main_frame, text="Create vector", command=self.generate_vector).pack(pady=10)

    def update_packets(self):
        for pf in self.packet_frames:
            pf.frame.destroy()
        self.packet_frames = []
        self.packet_configs = []
        for i in range(self.packet_count.get()):
            pc = PacketConfig(self.packets_container, i, self.packet_files)
            self.packet_frames.append(pc)
            self.packet_configs.append(pc)

    def generate_vector(self):
        try:
            vector_length = float(self.vector_length_var.get())
            # Use the highest sample rate for the output vector
            total_sr = max([pc.get_config()['sample_rate'] for pc in self.packet_configs])
            total_samples = int(vector_length * total_sr)
            vector = np.zeros(total_samples, dtype=np.complex64)

            for idx, pc in enumerate(self.packet_configs):
                cfg = pc.get_config()
                y = load_packet(cfg['file'])
                # Resample if needed
                if cfg['sample_rate'] != total_sr:
                    y = resample_signal(y, cfg['sample_rate'], total_sr)
                # Frequency shift
                if cfg['freq_shift'] != 0:
                    t = np.arange(len(y)) / total_sr
                    y = y * np.exp(2j * np.pi * cfg['freq_shift'] * t)
                # Period in samples
                period_samples = int(cfg['period'] * total_sr)
                # Insert packet from the beginning in every period
                for start in range(0, total_samples, period_samples):
                    end = start + len(y)
                    if start >= total_samples:
                        break
                    if end > total_samples:
                        y_to_add = y[:total_samples - start]
                    else:
                        y_to_add = y
                    vector[start:start + len(y_to_add)] += y_to_add
            # Debug
            print("Vector abs sum:", np.abs(vector).sum())
            print("Vector max:", np.abs(vector).max())
            print("Vector min:", np.abs(vector).min())
            # Normalize
            if self.normalize.get():
                max_abs = np.abs(vector).max()
                if max_abs > 0:
                    vector = vector / max_abs
            save_vector(vector, 'data/output_vector.mat')
            f, t, Sxx = create_spectrogram(vector, total_sr)
            plot_spectrogram(f, t, Sxx, title='Final Vector Spectrogram')
            messagebox.showinfo("Success", "Vector created and saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Error: {e}")

def main():
    root = tk.Tk()
    app = VectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 