# -*- coding: utf-8 -*-
"""
Unified GUI for packet extraction and vector generation
Combines packet extraction and vector building into a single modern interface
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import numpy as np
import scipy.io as sio
import os
import time
import traceback
from utils import (
    get_sample_rate_from_mat, 
    load_packet, 
    apply_frequency_shift, 
    adjust_packet_bounds_gui,
    plot_spectrogram,
    create_spectrogram,
    normalize_spectrogram,
    save_vector,
    save_vector_wv
)

# Modern theme configuration
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Constants
TARGET_SAMPLE_RATE = 56e6
MAX_PACKETS = 6

class ModernPacketExtractor:
    """Modern packet extraction interface"""
    
    def __init__(self, parent_frame):
        self.parent = parent_frame
        self.signal = None
        self.sample_rate = None
        self.start_sample = None
        self.end_sample = None
        self.extracted_packets = []
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ctk.CTkFrame(self.parent)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title
        title_label = ctk.CTkLabel(
            main_frame, 
            text="חילוץ פקטות", 
            font=ctk.CTkFont(size=24, weight="bold"),
            anchor="e"
        )
        title_label.pack(pady=10, fill="x")
        
        # File selection
        file_frame = ctk.CTkFrame(main_frame)
        file_frame.pack(fill="x", padx=20, pady=10)
        
        self.load_button = ctk.CTkButton(
            file_frame,
            text="\u200fבחר קובץ MAT",
            command=self.load_file,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="e",
        )
        self.load_button.pack(pady=10)
        
        # File info
        info_frame = ctk.CTkFrame(main_frame)
        info_frame.pack(fill="x", padx=20, pady=10)
        
        self.sample_rate_label = ctk.CTkLabel(
            info_frame, 
            text="קצב דגימה: --", 
            font=ctk.CTkFont(size=12),
            anchor="e"
        )
        self.sample_rate_label.pack(anchor="e", padx=10, pady=2)
        
        self.signal_length_label = ctk.CTkLabel(
            info_frame, 
            text="אורך האות: --", 
            font=ctk.CTkFont(size=12),
            anchor="e"
        )
        self.signal_length_label.pack(anchor="e", padx=10, pady=2)
        
        # Plot button
        self.plot_button = ctk.CTkButton(
            main_frame,
            text="\u200fפתח ספקטוגרמה וחתוך פקטה",
            command=self.show_spectrogram,
            state="disabled",
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="e",
        )
        self.plot_button.pack(pady=10)
        
        # Extracted packets list
        packets_frame = ctk.CTkFrame(main_frame)
        packets_frame.pack(fill="x", padx=20, pady=10)
        
        packets_label = ctk.CTkLabel(
            packets_frame, 
            text="פקטות שחולצו:", 
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="e"
        )
        packets_label.pack(anchor="e", padx=10, pady=5)
        
        # Listbox with Hebrew font support
        self.packets_listbox = tk.Listbox(
            packets_frame, 
            height=6,
            font=('Tahoma', 10)
        )
        self.packets_listbox.pack(fill="x", padx=10, pady=5)
        
    def load_file(self):
        """Load MAT file with performance optimization"""
        try:
            file_path = filedialog.askopenfilename(
                title="\u200fבחר קובץ MAT",
                filetypes=[("MAT files", "*.mat"), ("All files", "*.*")]
            )
            if not file_path:
                return
                
            # Check file size
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            if file_size > 100:  # If file is larger than 100MB
                result = messagebox.askyesno(
                    "קובץ גדול", 
                    f"הקובץ גדול ({file_size:.1f} MB). הטעינה עלולה לקחת זמן. להמשיך?"
                )
                if not result:
                    return
            
            # Load file with optimization
            print(f"Loading file {file_path} (size: {file_size:.1f} MB)...")
            start_time = time.time()
            data = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
            if 'Y' in data:
                self.signal = data['Y']
                if self.signal.ndim > 1:
                    self.signal = self.signal.flatten()
            else:
                candidates = [k for k in data.keys() if not k.startswith('__')]
                if len(candidates) == 1:
                    self.signal = data[candidates[0]]
                    if self.signal.ndim > 1:
                        self.signal = self.signal.flatten()
                else:
                    raise ValueError(f"Could not find suitable variable in {file_path}. Available variables: {list(data.keys())}")
            
            # Validate signal
            if len(self.signal) == 0:
                raise ValueError("Signal is empty")
                
            # Convert to more efficient data type
            if self.signal.dtype != np.complex64:
                self.signal = self.signal.astype(np.complex64)
                
            # Get sample rate
            self.sample_rate = get_sample_rate_from_mat(file_path)
            if not self.sample_rate:
                self.sample_rate = 56e6
            
            load_time = time.time() - start_time
            print(f"File loaded successfully in {load_time:.2f} seconds")
            
            # Update info
            self.sample_rate_label.configure(
                text=f"קצב דגימה: {self.sample_rate/1e6:.1f} MHz"
            )
            self.signal_length_label.configure(
                text=f"אורך האות: {len(self.signal):,} דגימות"
            )
            self.plot_button.configure(state="normal")
            
            messagebox.showinfo(
                "הצלחה", 
                f"הקובץ נטען בהצלחה!\nאורך: {len(self.signal):,} דגימות\nקצב דגימה: {self.sample_rate/1e6:.1f} MHz\nזמן טעינה: {load_time:.2f} שניות"
            )
            
        except Exception as e:
            messagebox.showerror("שגיאה", f"שגיאה בטעינת הקובץ: {e}")
            print(f"Error loading file: {traceback.format_exc()}")
                
    def show_spectrogram(self):
        """Show spectrogram and extract packet"""
        if self.signal is None:
            messagebox.showerror("שגיאה", "יש לבחור קובץ תחילה")
            return
            
        try:
            sample_rate = self.sample_rate
            self.start_sample, self.end_sample = adjust_packet_bounds_gui(
                self.signal,
                sample_rate,
                self.start_sample or 0,
                self.end_sample or len(self.signal),
            )
            
            # Validate bounds
            if self.start_sample >= self.end_sample:
                messagebox.showerror("שגיאה", "גבולות הפקטה לא תקינים")
                return
                
            # Extract packet
            packet = self.signal[self.start_sample:self.end_sample]
            
            if len(packet) == 0:
                messagebox.showerror("שגיאה", "הפקטה ריקה")
                return
            
            # Save packet
            packet_name = f"packet_{len(self.extracted_packets)+1}"
            packet_info = {
                'name': packet_name,
                'data': packet,
                'sample_rate': sample_rate,
                'start_sample': self.start_sample,
                'end_sample': self.end_sample,
                'file_path': f"data/{packet_name}.mat"
            }
            
            # Create data directory if it doesn't exist
            os.makedirs("data", exist_ok=True)
            
            # Save to file
            sio.savemat(packet_info['file_path'], {'Y': packet})
            print(f"Packet saved to {packet_info['file_path']}")
            
            # Add to list
            self.extracted_packets.append(packet_info)
            self.packets_listbox.insert(
                tk.END, 
                f"{packet_name} ({len(packet):,} דגימות)"
            )
            
            messagebox.showinfo(
                "הצלחה", 
                f"הפקטה נחלצה ונשמרה בהצלחה!\nשם: {packet_name}\nאורך: {len(packet):,} דגימות"
            )
            
        except Exception as e:
            messagebox.showerror("שגיאה", f"שגיאה בחילוץ הפקטה: {e}")
            print(f"Error extracting packet: {traceback.format_exc()}")

class ModernPacketConfig:
    """Class for packet configuration in vector building"""
    
    def __init__(self, parent, idx, packet_choices):
        self.frame = ctk.CTkFrame(parent)
        self.frame.pack(fill="x", padx=10, pady=5)
        
        # Title
        title_label = ctk.CTkLabel(
            self.frame, 
            text=f"פקטה {idx+1}", 
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="e"
        )
        title_label.pack(pady=5, fill="x")
        
        # Packet selection
        packet_label = ctk.CTkLabel(self.frame, text="בחר פקטה:", anchor="e")
        packet_label.pack(anchor="e", padx=10)
        
        self.packet_var = tk.StringVar()
        self.packet_menu = ctk.CTkOptionMenu(
            self.frame, 
            values=packet_choices if packet_choices else ["אין פקטות זמינות"], 
            variable=self.packet_var
        )
        self.packet_menu.pack(fill="x", padx=10, pady=5)
        if packet_choices:
            self.packet_var.set(packet_choices[0])
        
        # Parameters
        params_frame = ctk.CTkFrame(self.frame)
        params_frame.pack(fill="x", padx=10, pady=5)
        
        # Row 1
        row1 = ctk.CTkFrame(params_frame)
        row1.pack(fill="x", padx=5, pady=2)
        
        ctk.CTkLabel(row1, text="הזזת תדר (MHz):", width=120, anchor="e").pack(side="right", padx=5)
        self.freq_shift_var = tk.StringVar(value="0")
        ctk.CTkEntry(row1, textvariable=self.freq_shift_var, width=100).pack(side="right", padx=5)
        
        ctk.CTkLabel(row1, text="מחזור (ms):", width=80, anchor="e").pack(side="right", padx=5)
        self.period_var = tk.StringVar(value="100")
        ctk.CTkEntry(row1, textvariable=self.period_var, width=80).pack(side="right", padx=5)
        
        # Row 2
        row2 = ctk.CTkFrame(params_frame)
        row2.pack(fill="x", padx=5, pady=2)
        
        ctk.CTkLabel(row2, text="היסט זמן התחלה (ms):", width=140, anchor="e").pack(side="right", padx=5)
        self.start_time_var = tk.StringVar(value="0")
        ctk.CTkEntry(row2, textvariable=self.start_time_var, width=100).pack(side="right", padx=5)
        
        # Action buttons
        buttons_frame = ctk.CTkFrame(self.frame)
        buttons_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkButton(
            buttons_frame,
            text="\u200fהצג ספקטוגרמה",
            command=self.show_spectrogram,
            width=120,
            anchor="e",
        ).pack(side="right", padx=5)
        
        ctk.CTkButton(
            buttons_frame,
            text="\u200fניתוח פקטה",
            command=self.analyze_packet,
            width=120,
            anchor="e",
        ).pack(side="right", padx=5)
        
    def get_config(self):
        """Get packet configuration"""
        try:
            packet_path = self.packet_var.get()
            if not packet_path or packet_path == "אין פקטות זמינות":
                return None
                
            return {
                'file': packet_path,
                'freq_shift': float(self.freq_shift_var.get()) * 1e6,  # Convert MHz to Hz
                'period': float(self.period_var.get()) / 1000.0,  # Convert ms to seconds
                'start_time': float(self.start_time_var.get()) / 1000.0  # Convert ms to seconds
            }
        except ValueError as e:
            messagebox.showerror("שגיאה", f"ערכי פרמטרים לא תקינים: {e}")
            return None
            
    def show_spectrogram(self):
        """Show packet spectrogram"""
        try:
            config = self.get_config()
            if not config:
                return
                
            packet = load_packet(config['file'])
            f, t, Sxx = create_spectrogram(packet, TARGET_SAMPLE_RATE)
            plot_spectrogram(f, t, Sxx, title=f"Spectrogram - {os.path.basename(config['file'])}")
            
        except Exception as e:
            messagebox.showerror("שגיאה", f"שגיאה בהצגת ספקטוגרמה: {e}")
            print(f"Error showing spectrogram: {e}")
            
    def analyze_packet(self):
        """Analyze packet properties"""
        try:
            config = self.get_config()
            if not config:
                return
                
            packet = load_packet(config['file'])
            duration_ms = len(packet) / TARGET_SAMPLE_RATE * 1000
            energy = np.sum(np.abs(packet)**2)
            peak_amplitude = np.max(np.abs(packet))
            
            info = f"""פרטי הפקטה:
קובץ: {os.path.basename(config['file'])}
אורך: {len(packet):,} דגימות
משך: {duration_ms:.2f} מילישניות
אנרגיה: {energy:.2e}
משרעת שיא: {peak_amplitude:.2f}
הזזת תדר: {config['freq_shift']/1e6:.1f} MHz
מחזור: {config['period']*1000:.1f} ms
היסט זמן: {config['start_time']*1000:.1f} ms"""
            
            messagebox.showinfo("ניתוח פקטה", info)
            
        except Exception as e:
            messagebox.showerror("שגיאה", f"שגיאה בניתוח הפקטה: {e}")
            print(f"Error analyzing packet: {e}")

class UnifiedVectorApp:
    """Main unified application"""
    
    def __init__(self):
        self.packet_files = []
        self.packet_configs = []
        self.packet_count = None
        self.normalize = None
        self.vector_length_var = None
        self.packets_container = None
        self.packet_count_label = None
        self.update_packet_files()
        
    def update_packet_files(self):
        """Update list of available packet files"""
        if os.path.exists("data"):
            self.packet_files = [
                f"data/{f}" for f in os.listdir("data") 
                if f.endswith('.mat')
            ]
        else:
            self.packet_files = []
            
    def create_gui(self):
        """Create the main GUI"""
        self.root = ctk.CTk()
        self.root.title("Unified Vector Generator - מחולל וקטורים מאוחד")
        self.root.geometry("1400x900")
        
        # Initialize tkinter variables after root window is created
        self.packet_count = tk.IntVar(value=1)
        self.normalize = tk.BooleanVar(value=True)
        self.vector_length_var = tk.StringVar(value="1000")
        
        # Create notebook for tabs
        self.notebook = ctk.CTkTabview(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Packet extraction tab
        self.extraction_tab = self.notebook.add("חילוץ פקטות")
        self.packet_extractor = ModernPacketExtractor(self.extraction_tab)
        
        # Vector building tab
        self.vector_tab = self.notebook.add("בניית וקטור")
        self.create_vector_tab()
        
        # Refresh packets button
        refresh_button = ctk.CTkButton(
            self.root,
            text="\u200fרענן רשימת פקטות",
            command=self.refresh_packets,
            height=30,
            font=ctk.CTkFont(size=12),
            anchor="e",
        )
        refresh_button.pack(pady=5)
        
    def create_vector_tab(self):
        """Create vector building tab"""
        # Main frame
        main_frame = ctk.CTkFrame(self.vector_tab)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title
        title_label = ctk.CTkLabel(
            main_frame, 
            text="בניית וקטור", 
            font=ctk.CTkFont(size=24, weight="bold"),
            anchor="e"
        )
        title_label.pack(pady=10, fill="x")
        
        # General parameters
        general_frame = ctk.CTkFrame(main_frame)
        general_frame.pack(fill="x", padx=20, pady=10)
        
        general_label = ctk.CTkLabel(
            general_frame, 
            text="פרמטרים כלליים", 
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="e"
        )
        general_label.pack(pady=5, fill="x")
        
        # Vector length
        length_frame = ctk.CTkFrame(general_frame)
        length_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(length_frame, text="אורך וקטור (מילישניות):", width=150, anchor="e").pack(side="right", padx=5)
        ctk.CTkEntry(length_frame, textvariable=self.vector_length_var, width=100).pack(side="right", padx=5)
        
        # Number of packets
        packets_frame = ctk.CTkFrame(general_frame)
        packets_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(packets_frame, text="מספר פקטות (1-6):", width=150, anchor="e").pack(side="right", padx=5)
        ctk.CTkSlider(
            packets_frame, 
            from_=1, 
            to=MAX_PACKETS, 
            number_of_steps=MAX_PACKETS-1, 
            variable=self.packet_count, 
            command=self.update_packet_configs
        ).pack(side="right", padx=5, fill="x", expand=True)
        
        self.packet_count_label = ctk.CTkLabel(packets_frame, text="1", width=50)
        self.packet_count_label.pack(side="right", padx=5)
        
        # Normalization
        normalize_frame = ctk.CTkFrame(general_frame)
        normalize_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkCheckBox(
            normalize_frame, 
            text="נרמול וקטור סופי", 
            variable=self.normalize
        ).pack(side="right", padx=5)
        
        # Packets container
        self.packets_container = ctk.CTkScrollableFrame(main_frame)
        self.packets_container.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Generate buttons
        buttons_frame = ctk.CTkFrame(main_frame)
        buttons_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkButton(
            buttons_frame,
            text="\u200fצור וקטור MAT",
            command=self.generate_mat_vector,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="e",
        ).pack(side="right", padx=10, pady=5)
        
        ctk.CTkButton(
            buttons_frame,
            text="\u200fצור וקטור WV",
            command=self.generate_wv_vector,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="e",
        ).pack(side="right", padx=10, pady=5)
        
        # Create initial packet configurations (now that everything is ready)
        self.update_packet_configs()
        
    def update_packet_configs(self, *args):
        """Update packet configurations"""
        # Skip if GUI is not ready yet
        if not self.packet_count or not self.packet_count_label or not self.packets_container:
            return
            
        # Update label
        self.packet_count_label.configure(text=str(self.packet_count.get()))
        
        # Remove existing packets
        for pc in self.packet_configs:
            pc.frame.destroy()
        self.packet_configs = []
        
        # Create new packets
        for i in range(self.packet_count.get()):
            pc = ModernPacketConfig(self.packets_container, i, self.packet_files)
            self.packet_configs.append(pc)
            
    def refresh_packets(self):
        """Refresh packet list and update interface"""
        self.update_packet_files()
        
        # Update all menus
        for pc in self.packet_configs:
            current_values = self.packet_files if self.packet_files else ["אין פקטות זמינות"]
            pc.packet_menu.configure(values=current_values)
            if self.packet_files and not pc.packet_var.get():
                pc.packet_var.set(self.packet_files[0])
                
        messagebox.showinfo("רענון", f"רשימת הפקטות עודכנה! נמצאו {len(self.packet_files)} פקטות.")
        
    def generate_mat_vector(self):
        """Generate MAT vector"""
        self.generate_vector("mat")
        
    def generate_wv_vector(self):
        """Generate WV vector"""
        self.generate_vector("wv")
        
    def generate_vector(self, output_format="mat"):
        """Generate vector with performance optimization"""
        try:
            # Validate inputs
            if not self.packet_configs:
                messagebox.showerror("שגיאה", "אין פקטות מוגדרות")
                return
                
            vector_length_ms = float(self.vector_length_var.get())
            if vector_length_ms <= 0:
                messagebox.showerror("שגיאה", "אורך הווקטור חייב להיות חיובי")
                return
                
            # Convert from milliseconds to seconds
            vector_length = vector_length_ms / 1000.0
            
            # Show processing message
            progress_msg = messagebox.showinfo("מעבד...", "יוצר וקטור, אנא המתן...")
            
            total_samples = int(vector_length * TARGET_SAMPLE_RATE)
            vector = np.zeros(total_samples, dtype=np.complex64)
            freq_shifts = []
            markers = []
            marker_styles = ['x', 'o', '^', 's', 'D', 'P', 'v', '1', '2', '3', '4']
            marker_colors = [f"C{i}" for i in range(10)]
            style_map = {}
            
            valid_configs = 0
            print(f"Creating vector of length {vector_length_ms} ms ({total_samples:,} samples)")

            for idx, pc in enumerate(self.packet_configs):
                cfg = pc.get_config()
                if not cfg:
                    continue
                    
                valid_configs += 1
                print(f"Processing packet {idx+1}: {os.path.basename(cfg['file'])}")
                
                # Fast packet loading
                y = load_packet(cfg['file'])
                base_name = os.path.splitext(os.path.basename(cfg['file']))[0]
                if base_name not in style_map:
                    idx_style = len(style_map) % len(marker_styles)
                    idx_color = len(style_map) % len(marker_colors)
                    style_map[base_name] = (marker_styles[idx_style], marker_colors[idx_color])
                marker_style, marker_color = style_map[base_name]
                
                # Frequency shift
                if cfg['freq_shift'] != 0:
                    y = apply_frequency_shift(y, cfg['freq_shift'], TARGET_SAMPLE_RATE)
                    freq_shifts.append(cfg['freq_shift'])
                    print(f"  Applied frequency shift: {cfg['freq_shift']/1e6:.1f} MHz")
                else:
                    freq_shifts.append(0)
                
                # Period in samples
                period_samples = int(cfg['period'] * TARGET_SAMPLE_RATE)
                if period_samples <= 0:
                    messagebox.showerror("שגיאה", f"מחזור לא תקין בפקטה {idx+1}")
                    return

                # Time offset of first insertion in samples relative to start of vector
                start_offset = int(round(cfg['start_time'] * TARGET_SAMPLE_RATE))

                # Insert packet instances with correct timing
                current_pos = start_offset
                instance_count = 0
                
                while current_pos + len(y) <= total_samples:
                    end_pos = current_pos + len(y)
                    vector[current_pos:end_pos] += y
                    
                    # Add marker for this instance - convert sample position to time in seconds
                    marker_time = current_pos / TARGET_SAMPLE_RATE
                    markers.append((marker_time, cfg['freq_shift'], base_name, marker_style, marker_color))
                    
                    instance_count += 1
                    current_pos += period_samples
                
                print(f"  Inserted {instance_count} instances with period {cfg['period']*1000:.1f} ms")
            
            if valid_configs == 0:
                messagebox.showerror("שגיאה", "אין פקטות תקינות להכללה בוקטור")
                return
            
            # Normalization
            if self.normalize.get():
                max_val = np.max(np.abs(vector))
                if max_val > 0:
                    vector = vector / max_val
                    print("Vector normalized")
            
            # Save vector
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if output_format.lower() == "wv":
                output_path = f"vector_{timestamp}.wv"
                save_vector_wv(vector, output_path, TARGET_SAMPLE_RATE, normalize=False)
            else:
                output_path = f"vector_{timestamp}.mat"
                save_vector(vector, output_path)
            
            print(f"Vector saved to {output_path}")
            
            # Show final spectrogram
            try:
                f, t, Sxx = create_spectrogram(vector, TARGET_SAMPLE_RATE)
                plot_spectrogram(
                    f, t, Sxx, 
                    title=f"Final Vector Spectrogram - {output_format.upper()}",
                    packet_markers=markers,
                    sample_rate=TARGET_SAMPLE_RATE,
                    signal=vector
                )
                print("Final spectrogram displayed")
            except Exception as e:
                print(f"Warning: Could not display final spectrogram: {e}")
            
            messagebox.showinfo(
                "הצלחה", 
                f"הווקטור נוצר ונשמר בהצלחה!\nפורמט: {output_format.upper()}\nאורך: {vector_length_ms} מילישניות\nמספר דגימות: {len(vector):,}\nפקטות תקינות: {valid_configs}"
            )
            
        except ValueError as e:
            messagebox.showerror("שגיאה", f"ערך לא תקין: {e}")
            print(f"Value error: {e}")
        except Exception as e:
            messagebox.showerror("שגיאה", f"שגיאה ביצירת הווקטור: {e}")
            print(f"Error creating vector: {traceback.format_exc()}")
        
    def run(self):
        """Run the application"""
        self.root.mainloop()

def test_dependencies():
    """Test if all required dependencies are available"""
    try:
        import customtkinter
        import numpy
        import scipy
        import matplotlib
        from utils import load_packet
        print("All dependencies are available")
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without GUI"""
    try:
        # Test data directory creation
        os.makedirs("data", exist_ok=True)
        
        # Test numpy array creation
        test_array = np.zeros(1000, dtype=np.complex64)
        
        # Test packet loading (if packets exist)
        if os.path.exists("data") and any(f.endswith('.mat') for f in os.listdir("data")):
            packet_files = [f"data/{f}" for f in os.listdir("data") if f.endswith('.mat')]
            if packet_files:
                load_packet(packet_files[0])
                print(f"Successfully loaded packet: {packet_files[0]}")
        
        # Test vector saving
        test_vector = np.random.random(1000).astype(np.complex64)
        save_vector(test_vector, "test_vector.mat")
        os.remove("test_vector.mat")  # Clean up
        
        print("Basic functionality test passed")
        return True
    except Exception as e:
        print(f"Basic functionality test failed: {e}")
        return False

def main():
    """Main function"""
    print("Starting Unified Vector Generator...")
    
    # Test dependencies
    if not test_dependencies():
        print("Please install missing dependencies")
        return
    
    # Test basic functionality
    if not test_basic_functionality():
        print("Basic functionality test failed")
        return
    
    # Create and run application
    app = UnifiedVectorApp()
    app.create_gui()
    app.run()

if __name__ == "__main__":
    main()