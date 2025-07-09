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
    save_vector_wv,
    detect_packet_bounds,
    load_packet_info
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
        
        # Quality control settings - optimized for heavy packets
        self.quality_preset = tk.StringVar(value="Fast")
        self.max_samples = tk.IntVar(value=2_000_000)  # Increased for heavy packets
        self.time_resolution = tk.DoubleVar(value=10.0)  # Start with fast setting
        self.adaptive_mode = tk.BooleanVar(value=True)
        self.heavy_packet_mode = tk.BooleanVar(value=True)  # New: Heavy packet support
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ctk.CTkFrame(self.parent)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title
        title_label = ctk.CTkLabel(
            main_frame, 
            text="Packet Extraction with Quality Control", 
            font=ctk.CTkFont(size=24, weight="bold"),
            anchor="e"
        )
        title_label.pack(pady=10, fill="x")
        
        # Quality Control Section
        quality_frame = ctk.CTkFrame(main_frame)
        quality_frame.pack(fill="x", padx=20, pady=10)
        
        # Quality section title
        quality_title = ctk.CTkLabel(
            quality_frame,
            text="🎛️ Quality Control Settings",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        quality_title.pack(pady=(10, 5))
        
        # Quality presets
        preset_frame = ctk.CTkFrame(quality_frame)
        preset_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(preset_frame, text="Quality Preset:", font=ctk.CTkFont(size=14, weight="bold")).pack(side="left", padx=10, pady=5)
        
        self.quality_menu = ctk.CTkOptionMenu(
            preset_frame,
            values=["Fast", "Balanced", "High Quality"],
            variable=self.quality_preset,
            command=self.on_quality_preset_change,
            width=150,
            height=35,
            font=ctk.CTkFont(size=14)
        )
        self.quality_menu.pack(side="left", padx=10, pady=5)
        
        # Quality info label
        self.quality_info_label = ctk.CTkLabel(
            preset_frame, 
            text="⚡ Fast: Quick loading for large files",
            font=ctk.CTkFont(size=12)
        )
        self.quality_info_label.pack(side="left", padx=20, pady=5)
        
        # Advanced quality controls
        advanced_frame = ctk.CTkFrame(quality_frame)
        advanced_frame.pack(fill="x", padx=10, pady=5)
        
        # Max samples control
        max_samples_frame = ctk.CTkFrame(advanced_frame)
        max_samples_frame.pack(fill="x", padx=5, pady=2)
        
        ctk.CTkLabel(max_samples_frame, text="Max Samples:", width=120, anchor="w").pack(side="left", padx=5)
        self.max_samples_entry = ctk.CTkEntry(max_samples_frame, textvariable=self.max_samples, width=120)
        self.max_samples_entry.pack(side="left", padx=5)
        
        ctk.CTkLabel(max_samples_frame, text="Time Resolution (μs):", width=130, anchor="w").pack(side="left", padx=5)
        self.time_resolution_entry = ctk.CTkEntry(max_samples_frame, textvariable=self.time_resolution, width=100)
        self.time_resolution_entry.pack(side="left", padx=5)
        
        # Adaptive mode toggle
        adaptive_frame = ctk.CTkFrame(advanced_frame)
        adaptive_frame.pack(fill="x", padx=5, pady=2)
        
        self.adaptive_check = ctk.CTkCheckBox(
            adaptive_frame,
            text="Adaptive Resolution Mode",
            variable=self.adaptive_mode,
            font=ctk.CTkFont(size=12)
        )
        self.adaptive_check.pack(side="left", padx=5, pady=5)
        
        # Heavy packet mode toggle
        self.heavy_packet_check = ctk.CTkCheckBox(
            adaptive_frame,
            text="Heavy Packet Mode (>10M samples)",
            variable=self.heavy_packet_mode,
            font=ctk.CTkFont(size=12)
        )
        self.heavy_packet_check.pack(side="left", padx=20, pady=5)
        
        # Packet extraction section
        extraction_frame = ctk.CTkFrame(main_frame)
        extraction_frame.pack(fill="x", padx=20, pady=10)
        
        # Section title
        section_title = ctk.CTkLabel(
            extraction_frame,
            text="📁 Packet Extraction",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        section_title.pack(pady=(10, 5))
        
        # File selection
        file_frame = ctk.CTkFrame(extraction_frame)
        file_frame.pack(fill="x", padx=10, pady=5)
        
        self.file_button = ctk.CTkButton(
            file_frame,
            text="Select MAT File",
            command=self.load_file,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.file_button.pack(side="left", padx=10, pady=10)
        
        self.file_label = ctk.CTkLabel(file_frame, text="No file selected")
        self.file_label.pack(side="left", padx=10)
        
        # File info
        info_frame = ctk.CTkFrame(extraction_frame)
        info_frame.pack(fill="x", padx=10, pady=5)
        
        self.sample_rate_label = ctk.CTkLabel(info_frame, text="Sample Rate: --")
        self.sample_rate_label.pack(side="left", padx=10, pady=5)
        
        self.signal_length_label = ctk.CTkLabel(info_frame, text="Signal Length: --")
        self.signal_length_label.pack(side="left", padx=10, pady=5)
        
        # Quality performance info
        self.performance_label = ctk.CTkLabel(info_frame, text="")
        self.performance_label.pack(side="left", padx=10, pady=5)
        
        # Packet name input
        name_frame = ctk.CTkFrame(extraction_frame)
        name_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(name_frame, text="Packet Name:", font=ctk.CTkFont(size=14, weight="bold")).pack(side="left", padx=10, pady=5)
        
        self.packet_name_var = tk.StringVar(value=f"packet_{len(self.extracted_packets)+1}")
        self.packet_name_entry = ctk.CTkEntry(
            name_frame, 
            textvariable=self.packet_name_var,
            placeholder_text="Enter packet name...",
            width=200,
            height=35,
            font=ctk.CTkFont(size=14)
        )
        self.packet_name_entry.pack(side="left", padx=10, pady=5)
        
        # Auto-update button
        auto_name_button = ctk.CTkButton(
            name_frame,
            text="Auto Name",
            command=self.auto_generate_name,
            width=80,
            height=35
        )
        auto_name_button.pack(side="left", padx=5, pady=5)
        
        # Action buttons
        button_frame = ctk.CTkFrame(extraction_frame)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        self.extract_button = ctk.CTkButton(
            button_frame,
            text="Open Spectrogram and Cut Packet",
            command=self.show_spectrogram,
            state="disabled",
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.extract_button.pack(side="left", padx=10, pady=5)
        
        # Quality test button
        self.test_button = ctk.CTkButton(
            button_frame,
            text="Test Current Quality",
            command=self.test_current_quality,
            state="disabled",
            height=40,
            font=ctk.CTkFont(size=12),
            width=150
        )
        self.test_button.pack(side="left", padx=10, pady=5)
        
        # Extracted packets list
        packets_frame = ctk.CTkFrame(main_frame)
        packets_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Section title with refresh button
        packets_header = ctk.CTkFrame(packets_frame)
        packets_header.pack(fill="x", pady=(10, 5))
        
        packets_title = ctk.CTkLabel(
            packets_header,
            text="Extracted Packets:",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        packets_title.pack(side="left", pady=5)
        
        # Auto-refresh toggle
        self.auto_refresh_var = tk.BooleanVar(value=True)
        auto_refresh_check = ctk.CTkCheckBox(
            packets_header,
            text="Auto Refresh",
            variable=self.auto_refresh_var,
            font=ctk.CTkFont(size=12)
        )
        auto_refresh_check.pack(side="right", padx=10, pady=5)
        
        refresh_button = ctk.CTkButton(
            packets_header,
            text="Refresh Now",
            command=self.refresh_packet_list,
            width=100,
            height=30
        )
        refresh_button.pack(side="right", padx=5, pady=5)
        
        # Create scrollable frame for packets list
        self.packets_scroll = ctk.CTkScrollableFrame(packets_frame, height=150)
        self.packets_scroll.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Headers
        headers_frame = ctk.CTkFrame(self.packets_scroll)
        headers_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(headers_frame, text="Packet Name", width=200, font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        ctk.CTkLabel(headers_frame, text="Samples", width=100, font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        ctk.CTkLabel(headers_frame, text="Duration (ms)", width=120, font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        ctk.CTkLabel(headers_frame, text="Actions", width=100, font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        
        # Initialize packets list
        self.packet_rows = []
        
        # Load existing packets on startup
        self.refresh_packet_list()
        
    def auto_generate_name(self):
        """Auto-generate packet name"""
        self.packet_name_var.set(f"packet_{len(self.extracted_packets)+1}")
        
    def on_quality_preset_change(self, choice):
        """Update quality controls based on preset selection - optimized for heavy packets"""
        if choice == "Fast":
            self.max_samples.set(1_000_000)  # Much lower for speed
            self.time_resolution.set(50.0)   # Much faster for heavy packets
            self.adaptive_mode.set(True)
            self.heavy_packet_mode.set(True)
            self.quality_info_label.configure(text="⚡ Fast: Maximum speed for heavy packets")
        elif choice == "Balanced":
            self.max_samples.set(2_000_000)  # Balanced for heavy packets
            self.time_resolution.set(25.0)
            self.adaptive_mode.set(True)
            self.heavy_packet_mode.set(True)
            self.quality_info_label.configure(text="⚖️ Balanced: Good balance for heavy packets")
        elif choice == "High Quality":
            self.max_samples.set(5_000_000)  # Reduced maximum for speed
            self.time_resolution.set(10.0)
            self.adaptive_mode.set(True)
            self.heavy_packet_mode.set(True)
            self.quality_info_label.configure(text="🔬 High Quality: Best quality for heavy packets")
        self.test_button.configure(state="disabled") # Disable test button until file is loaded
        
    def auto_adjust_quality_settings(self, signal_length, file_size_mb):
        """Auto-adjust quality settings based on file characteristics"""
        print(f"📊 Auto-adjusting quality settings for {signal_length:,} samples ({file_size_mb:.1f}MB)")
        
        # Determine appropriate settings based on signal characteristics
        duration_sec = signal_length / 56e6  # Assume 56MHz sample rate
        
        if signal_length <= 1_000_000:  # Small files (< 1M samples)
            recommended_preset = "High Quality"
            max_samples = 2_000_000
            time_resolution = 5.0
            info_text = "🔬 High Quality: Small file detected"
        elif signal_length <= 5_000_000:  # Medium files (1-5M samples)
            recommended_preset = "Balanced"  
            max_samples = 2_000_000
            time_resolution = 15.0
            info_text = "⚖️ Balanced: Medium file detected"
        elif signal_length <= 20_000_000:  # Large files (5-20M samples)
            recommended_preset = "Fast"
            max_samples = 1_000_000
            time_resolution = 30.0
            info_text = "⚡ Fast: Large file detected"
        else:  # Very large files (>20M samples)
            recommended_preset = "Fast"
            max_samples = 500_000
            time_resolution = 50.0
            info_text = "⚡ Fast: Very large file - maximum optimization"
        
        # Update the settings
        self.quality_preset.set(recommended_preset)
        self.max_samples.set(max_samples)
        self.time_resolution.set(time_resolution)
        self.quality_info_label.configure(text=info_text)
        
        # Enable heavy packet mode for large files
        if signal_length > 5_000_000:
            self.heavy_packet_mode.set(True)
            
        print(f"✅ Settings updated: {recommended_preset} preset, {max_samples:,} max samples, {time_resolution}μs resolution")
        
    def test_current_quality(self):
        """Test the current quality settings"""
        if self.signal is None:
            messagebox.showerror("Error", "Please select a file first")
            return
            
        try:
            max_samples = self.max_samples.get()
            time_resolution_us = self.time_resolution.get()
            adaptive = self.adaptive_mode.get()
            
            print(f"Testing quality with:")
            print(f"  Max Samples: {max_samples:,}")
            print(f"  Time Resolution: {time_resolution_us:.1f} μs")
            print(f"  Adaptive Mode: {adaptive}")
            
            # Simulate loading a large file
            large_signal = np.random.random(max_samples).astype(np.complex64)
            
            start_time = time.time()
            f, t, Sxx = create_spectrogram(large_signal, TARGET_SAMPLE_RATE, time_resolution_us=int(time_resolution_us), adaptive_resolution=adaptive)
            load_time = time.time() - start_time
            
            print(f"Spectrogram created in {load_time:.2f} seconds")
            
            messagebox.showinfo(
                "Quality Test Result",
                f"Quality Test Result:\n"
                f"  Max Samples: {max_samples:,} (used)\n"
                f"  Time Resolution: {time_resolution_us:.1f} μs (used)\n"
                f"  Adaptive Mode: {adaptive}\n"
                f"  Loading Time: {load_time:.2f} seconds"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Error testing quality: {e}")
            print(f"Error testing quality: {traceback.format_exc()}")
            
    def refresh_packet_list(self):
        """Refresh the displayed packet list"""
        try:
            # Clear existing rows
            for row in self.packet_rows:
                row.destroy()
            self.packet_rows = []
            
            # Scan data directory for existing packets
            if os.path.exists("data"):
                packet_files = [f for f in os.listdir("data") if f.endswith('.mat')]
                packet_files.sort()  # Sort alphabetically
                
                for packet_file in packet_files:
                    try:
                        # Load packet to get info
                        packet_path = os.path.join("data", packet_file)
                        data = sio.loadmat(packet_path, squeeze_me=True)
                        
                        if 'Y' in data:
                            packet_data = data['Y']
                            if packet_data.ndim > 1:
                                packet_data = packet_data.flatten()
                        else:
                            # Try to find the main data variable
                            candidates = [k for k in data.keys() if not k.startswith('__')]
                            if candidates:
                                packet_data = data[candidates[0]]
                                if packet_data.ndim > 1:
                                    packet_data = packet_data.flatten()
                            else:
                                continue
                        
                        # Create display row
                        packet_row = ctk.CTkFrame(self.packets_scroll)
                        packet_row.pack(fill="x", pady=2)
                        
                        packet_name = os.path.splitext(packet_file)[0]
                        duration_ms = len(packet_data) / TARGET_SAMPLE_RATE * 1000
                        
                        ctk.CTkLabel(packet_row, text=packet_name, width=200).pack(side="left", padx=5)
                        ctk.CTkLabel(packet_row, text=f"{len(packet_data):,}", width=100).pack(side="left", padx=5)
                        ctk.CTkLabel(packet_row, text=f"{duration_ms:.1f}", width=120).pack(side="left", padx=5)
                        
                        # Delete button
                        delete_btn = ctk.CTkButton(
                            packet_row,
                            text="Delete",
                            command=lambda p=packet_path: self.delete_packet(p),
                            width=80,
                            height=25,
                            fg_color="red",
                            hover_color="darkred"
                        )
                        delete_btn.pack(side="left", padx=5)
                        
                        self.packet_rows.append(packet_row)
                        
                    except Exception as e:
                        print(f"Error loading packet {packet_file}: {e}")
                        continue
                        
            print(f"Packet list refreshed: found {len(self.packet_rows)} packets")
            
        except Exception as e:
            print(f"Error refreshing packet list: {e}")
            
    def delete_packet(self, packet_path):
        """Delete a packet file"""
        try:
            result = messagebox.askyesno(
                "Confirm Delete", 
                f"Are you sure you want to delete {os.path.basename(packet_path)}?"
            )
            if result:
                os.remove(packet_path)
                self.refresh_packet_list()
                messagebox.showinfo("Success", "Packet deleted successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Error deleting packet: {e}")
        
    def load_file(self):
        """Load MAT file with performance optimization"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select MAT File",
                filetypes=[("MAT files", "*.mat"), ("All files", "*.*")]
            )
            if not file_path:
                return
                
            # Check file size
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            if file_size > 100:  # If file is larger than 100MB
                result = messagebox.askyesno(
                    "Large File", 
                    f"The file is large ({file_size:.1f} MB). Loading may take time. Continue?"
                )
                if not result:
                    return
            
            # Load file with optimization
            print(f"Loading file {file_path} (size: {file_size:.1f} MB)...")
            start_time = time.time()
            
            max_samples = self.max_samples.get()
            time_resolution_us = self.time_resolution.get()
            adaptive = self.adaptive_mode.get()
            
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
                text=f"Sample Rate: {self.sample_rate/1e6:.1f} MHz"
            )
            self.signal_length_label.configure(
                text=f"Signal Length: {len(self.signal):,} samples"
            )
            
            # Auto-adjust quality settings based on file characteristics
            self.auto_adjust_quality_settings(len(self.signal), file_size)
            
            self.extract_button.configure(state="normal")
            self.test_button.configure(state="normal") # Enable test button after file is loaded
            
            messagebox.showinfo(
                "Success", 
                f"File loaded successfully!\nLength: {len(self.signal):,} samples\nSample Rate: {self.sample_rate/1e6:.1f} MHz\nLoading Time: {load_time:.2f} seconds"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading file: {e}")
            print(f"Error loading file: {traceback.format_exc()}")
                
    def show_spectrogram(self):
        """Show spectrogram and extract packet - optimized for heavy packets"""
        if self.signal is None:
            messagebox.showerror("Error", "Please select a file first")
            return
            
        try:
            sample_rate = self.sample_rate
            
            # Get quality settings
            max_samples = self.max_samples.get()
            time_resolution_us = int(self.time_resolution.get())
            adaptive_resolution = self.adaptive_mode.get()
            heavy_mode = self.heavy_packet_mode.get()
            
            # Heavy packet detection and optimization
            is_heavy = len(self.signal) > 5_000_000
            if is_heavy and heavy_mode:
                print(f"🔍 Heavy packet mode activated for {len(self.signal):,} samples")
                # Optimize parameters for heavy packets - MUCH more aggressive
                max_samples = min(max_samples, 1_000_000)  # Limit to 1M for speed
                time_resolution_us = max(time_resolution_us, 50)  # Minimum 50μs
                print(f"📉 Using optimized parameters: max_samples={max_samples:,}, time_res={time_resolution_us}μs")
            
            if self.start_sample is None or self.end_sample is None:
                start_det, end_det = detect_packet_bounds(self.signal, sample_rate)
                buffer_samples = int(sample_rate // 1_000_000)
                start = max(0, start_det - buffer_samples)
                self.start_sample, self.end_sample = start, end_det
                self.detected_start = start_det
                self._pre_buffer = buffer_samples

            start_time = time.time()
            
            # Always show GUI for packet bounds adjustment, but with optimized parameters for heavy packets
            print(f"🔍 Opening spectrogram window with optimized settings")
            self.start_sample, self.end_sample = adjust_packet_bounds_gui(
                self.signal,
                sample_rate,
                self.start_sample,
                self.end_sample,
                max_samples=max_samples,
                time_resolution_us=time_resolution_us,
                adaptive_resolution=adaptive_resolution
            )
            process_time = time.time() - start_time
            
            # Update performance info with heavy packet indicator
            perf_text = f"Processing time: {process_time:.2f}s"
            if is_heavy:
                perf_text += " (Heavy Packet)"
            self.performance_label.configure(text=perf_text)
            
            # Validate bounds
            if self.start_sample >= self.end_sample:
                messagebox.showerror("Error", "Invalid packet bounds")
                return
                
            # Extract packet
            packet = self.signal[self.start_sample:self.end_sample]
            
            if len(packet) == 0:
                messagebox.showerror("Error", "Empty packet")
                return
            
            # Validate and clean packet name
            packet_name = self.packet_name_var.get().strip()
            if not packet_name:
                packet_name = f"packet_{len(self.extracted_packets)+1}"
                self.packet_name_var.set(packet_name)
            
            # Remove invalid characters from filename
            import re
            packet_name = re.sub(r'[<>:"/\\|?*]', '_', packet_name)
            
            # Check if file already exists and ask for confirmation
            file_path = f"data/{packet_name}.mat"
            if os.path.exists(file_path):
                result = messagebox.askyesno(
                    "File Exists", 
                    f"A packet named '{packet_name}' already exists. Overwrite?"
                )
                if not result:
                    return
            
            packet_info = {
                'name': packet_name,
                'data': packet,
                'sample_rate': sample_rate,
                'start_sample': self.start_sample,
                'end_sample': self.end_sample,
                'file_path': file_path,
                'pre_samples': int(getattr(self, 'detected_start', self.start_sample) - self.start_sample)
            }
            
            # Create data directory if it doesn't exist
            os.makedirs("data", exist_ok=True)
            
            # Save to file
            sio.savemat(packet_info['file_path'], {'Y': packet, 'pre_samples': packet_info['pre_samples']})
            print(f"Packet saved to {packet_info['file_path']}")
            
            # Add to list
            self.extracted_packets.append(packet_info)
            
            # Auto-refresh the packet list if enabled
            if self.auto_refresh_var.get():
                self.refresh_packet_list()
                
                # Also refresh the vector building tab if connected
                if hasattr(self, 'parent_app') and self.parent_app:
                    self.parent_app.auto_refresh_packets()
                
            # Update name for next packet
            next_num = len(self.extracted_packets) + 1
            self.packet_name_var.set(f"packet_{next_num}")
            
            messagebox.showinfo(
                "Success", 
                f"Packet extracted and saved successfully!\nName: {packet_name}\nLength: {len(packet):,} samples\nDuration: {len(packet) / sample_rate * 1000:.1f} ms"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Error extracting packet: {e}")
            print(f"Error extracting packet: {traceback.format_exc()}")

class ModernPacketConfig:
    """Class for packet configuration in vector building"""
    
    def __init__(self, parent, idx, packet_choices):
        self.frame = ctk.CTkFrame(parent)
        self.frame.pack(fill="x", padx=10, pady=5)
        
        # Title
        title_label = ctk.CTkLabel(
            self.frame,
            text=f"Packet {idx+1}",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w"
        )
        title_label.pack(pady=5, fill="x")
        
        # Packet selection
        packet_label = ctk.CTkLabel(self.frame, text="Select Packet:", anchor="w")
        packet_label.pack(anchor="w", padx=10)
        
        self.packet_var = tk.StringVar()
        self.packet_menu = ctk.CTkOptionMenu(
            self.frame, 
            values=packet_choices if packet_choices else ["No packets available"], 
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
        
        ctk.CTkLabel(row1, text="Freq Shift (MHz):", width=120, anchor="w").pack(side="left", padx=5)
        self.freq_shift_var = tk.StringVar(value="0")
        ctk.CTkEntry(row1, textvariable=self.freq_shift_var, width=100).pack(side="left", padx=5)

        ctk.CTkLabel(row1, text="Period (ms):", width=80, anchor="w").pack(side="left", padx=5)
        self.period_var = tk.StringVar(value="100")
        ctk.CTkEntry(row1, textvariable=self.period_var, width=80).pack(side="left", padx=5)
        
        # Row 2
        row2 = ctk.CTkFrame(params_frame)
        row2.pack(fill="x", padx=5, pady=2)
        
        ctk.CTkLabel(row2, text="Start Time Offset (ms):", width=140, anchor="w").pack(side="left", padx=5)
        self.start_time_var = tk.StringVar(value="0")
        ctk.CTkEntry(row2, textvariable=self.start_time_var, width=100).pack(side="left", padx=5)
        
        # Action buttons
        buttons_frame = ctk.CTkFrame(self.frame)
        buttons_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkButton(
            buttons_frame, 
            text="Show Spectrogram", 
            command=self.show_spectrogram, 
            width=120
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            buttons_frame, 
            text="Analyze Packet", 
            command=self.analyze_packet, 
            width=120
        ).pack(side="left", padx=5)
        
    def get_config(self):
        """Get packet configuration"""
        try:
            packet_path = self.packet_var.get()
            if not packet_path or packet_path == "No packets available":
                return None
                
            return {
                'file': packet_path,
                'freq_shift': float(self.freq_shift_var.get()) * 1e6,  # Convert MHz to Hz
                'period': float(self.period_var.get()) / 1000.0,  # Convert ms to seconds
                'start_time': float(self.start_time_var.get()) / 1000.0  # Convert ms to seconds
            }
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter values: {e}")
            return None
            
    def show_spectrogram(self):
        """Show packet spectrogram"""
        try:
            config = self.get_config()
            if not config:
                return
                
            packet = load_packet(config['file'])
            f, t, Sxx = create_spectrogram(packet, TARGET_SAMPLE_RATE, time_resolution_us=1)
            plot_spectrogram(f, t, Sxx, title=f"Spectrogram - {os.path.basename(config['file'])}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error showing spectrogram: {e}")
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
            
            info = f"""Packet Details:
File: {os.path.basename(config['file'])}
Length: {len(packet):,} samples
Duration: {duration_ms:.2f} milliseconds
Energy: {energy:.2e}
Peak Amplitude: {peak_amplitude:.2f}
Freq Shift: {config['freq_shift']/1e6:.1f} MHz
Period: {config['period']*1000:.1f} ms
Start Time Offset: {config['start_time']*1000:.1f} ms"""
            
            messagebox.showinfo("Packet Analysis", info)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error analyzing packet: {e}")
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
        self.root.title("Unified Vector Generator - Unified Vector Generator")
        self.root.geometry("1400x900")
        
        # Initialize tkinter variables after root window is created
        self.packet_count = tk.IntVar(value=1)
        self.normalize = tk.BooleanVar(value=True)
        self.vector_length_var = tk.StringVar(value="1000")
        
        # Create notebook for tabs
        self.notebook = ctk.CTkTabview(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Packet extraction tab
        self.extraction_tab = self.notebook.add("Packet Extraction")
        self.packet_extractor = ModernPacketExtractor(self.extraction_tab)
        
        # Set up connection for auto-refresh
        self.packet_extractor.parent_app = self
        
        # Vector building tab
        self.vector_tab = self.notebook.add("Vector Building")
        self.create_vector_tab()
        
        # Refresh packets button
        refresh_button = ctk.CTkButton(
            self.root, 
            text="Refresh Packet List", 
            command=self.refresh_packets,
            height=30,
            font=ctk.CTkFont(size=12)
        )
        refresh_button.pack(pady=5)
        
        # Start auto-refresh timer (every 2 seconds)
        self.start_auto_refresh_timer()
        
    def create_vector_tab(self):
        """Create vector building tab"""
        # Main frame
        main_frame = ctk.CTkFrame(self.vector_tab)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title
        title_label = ctk.CTkLabel(
            main_frame,
            text="Vector Building",
            font=ctk.CTkFont(size=24, weight="bold"),
            anchor="w"
        )
        title_label.pack(pady=10, fill="x")
        
        # General parameters
        general_frame = ctk.CTkFrame(main_frame)
        general_frame.pack(fill="x", padx=20, pady=10)
        
        general_label = ctk.CTkLabel(
            general_frame,
            text="General Parameters",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w"
        )
        general_label.pack(pady=5, fill="x")
        
        # Vector length
        length_frame = ctk.CTkFrame(general_frame)
        length_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(length_frame, text="Vector Length (milliseconds):", width=150, anchor="w").pack(side="left", padx=5)
        ctk.CTkEntry(length_frame, textvariable=self.vector_length_var, width=100).pack(side="left", padx=5)
        
        # Number of packets
        packets_frame = ctk.CTkFrame(general_frame)
        packets_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(packets_frame, text="Number of Packets (1-6):", width=150, anchor="w").pack(side="left", padx=5)
        ctk.CTkSlider(
            packets_frame, 
            from_=1, 
            to=MAX_PACKETS, 
            number_of_steps=MAX_PACKETS-1, 
            variable=self.packet_count, 
            command=self.update_packet_configs
        ).pack(side="left", padx=5, fill="x", expand=True)
        
        self.packet_count_label = ctk.CTkLabel(packets_frame, text="1", width=50)
        self.packet_count_label.pack(side="left", padx=5)
        
        # Normalization
        normalize_frame = ctk.CTkFrame(general_frame)
        normalize_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkCheckBox(
            normalize_frame, 
            text="Normalize Final Vector", 
            variable=self.normalize
        ).pack(side="left", padx=5)
        
        # Packets container
        self.packets_container = ctk.CTkScrollableFrame(main_frame)
        self.packets_container.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Generate buttons
        buttons_frame = ctk.CTkFrame(main_frame)
        buttons_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkButton(
            buttons_frame,
            text="Create MAT Vector",
            command=self.generate_mat_vector,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(side="left", padx=10, pady=5)
        
        ctk.CTkButton(
            buttons_frame,
            text="Create WV Vector",
            command=self.generate_wv_vector,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(side="left", padx=10, pady=5)
        
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
            current_values = self.packet_files if self.packet_files else ["No packets available"]
            pc.packet_menu.configure(values=current_values)
            if self.packet_files and not pc.packet_var.get():
                pc.packet_var.set(self.packet_files[0])
                
        # Also refresh the extractor's packet list if it exists
        if hasattr(self, 'packet_extractor') and hasattr(self.packet_extractor, 'refresh_packet_list'):
            self.packet_extractor.refresh_packet_list()
                
        print(f"Packet list updated! Found {len(self.packet_files)} packets.")
        
    def auto_refresh_packets(self):
        """Auto-refresh packets without showing message"""
        self.update_packet_files()
        
        # Update all menus silently
        for pc in self.packet_configs:
            current_values = self.packet_files if self.packet_files else ["No packets available"]
            pc.packet_menu.configure(values=current_values)
            if self.packet_files and not pc.packet_var.get():
                pc.packet_var.set(self.packet_files[0])
                
        # Also refresh the extractor's packet list if it exists
        if hasattr(self, 'packet_extractor') and hasattr(self.packet_extractor, 'refresh_packet_list'):
            self.packet_extractor.refresh_packet_list()
            
    def start_auto_refresh_timer(self):
        """Start automatic refresh timer"""
        def check_for_updates():
            try:
                # Store current packet count
                if not hasattr(self, '_last_packet_count'):
                    self._last_packet_count = len(self.packet_files)
                
                # Check if packet count changed
                self.update_packet_files()
                current_count = len(self.packet_files)
                
                if current_count != self._last_packet_count:
                    print(f"Packet count changed: {self._last_packet_count} -> {current_count}")
                    self.auto_refresh_packets()
                    self._last_packet_count = current_count
                    
            except Exception as e:
                print(f"Auto-refresh error: {e}")
            
            # Schedule next check
            self.root.after(2000, check_for_updates)  # Check every 2 seconds
            
        # Start the timer
        self.root.after(1000, check_for_updates)  # First check after 1 second
        
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
                messagebox.showerror("Error", "No packets configured")
                return
                
            vector_length_ms = float(self.vector_length_var.get())
            if vector_length_ms <= 0:
                messagebox.showerror("Error", "Vector length must be positive")
                return
                
            # Convert from milliseconds to seconds
            vector_length = vector_length_ms / 1000.0
            
            # Show processing message
            progress_msg = messagebox.showinfo("Processing...", "Creating vector, please wait...")
            
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
                y, pre_buf = load_packet_info(cfg['file'])
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
                    messagebox.showerror("Error", f"Invalid period for packet {idx+1}")
                    return

                # Time offset of first insertion in samples relative to start of vector
                start_offset = max(0, int(round(cfg['start_time'] * TARGET_SAMPLE_RATE)) - pre_buf)

                # Insert packet instances with correct timing
                current_pos = start_offset
                instance_count = 0
                
                while current_pos + len(y) <= total_samples:
                    end_pos = current_pos + len(y)
                    vector[current_pos:end_pos] += y
                    
                    # Add marker for this instance - convert sample position to time in seconds
                    marker_time = (current_pos + pre_buf) / TARGET_SAMPLE_RATE
                    markers.append((marker_time, cfg['freq_shift'], base_name, marker_style, marker_color))
                    
                    instance_count += 1
                    current_pos += period_samples
                
                print(f"  Inserted {instance_count} instances with period {cfg['period']*1000:.1f} ms")
            
            if valid_configs == 0:
                messagebox.showerror("Error", "No valid packets to include in vector")
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
                f, t, Sxx = create_spectrogram(vector, TARGET_SAMPLE_RATE, time_resolution_us=1)
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
                "Success", 
                f"Vector created and saved successfully!\nFormat: {output_format.upper()}\nLength: {vector_length_ms} milliseconds\nSamples: {len(vector):,}\nValid packets: {valid_configs}"
            )
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid value: {e}")
            print(f"Value error: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Error creating vector: {e}")
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