"""
Long Recording Analyzer
Module for loading long recordings (1-2 seconds, 56 MSps), 
automatically detecting packets, cutting them with safety margins,
and selecting the highest quality packet of each type.
"""

import os
import numpy as np
import scipy.io as sio
from scipy import signal
from scipy.ndimage import label, find_objects
import matplotlib.pyplot as plt
from collections import defaultdict
import shutil
from datetime import datetime

class LongRecordingAnalyzer:
    def __init__(self, sample_rate=56e6, safety_margin_ms=0.5):
        """
        Initialize long recording analyzer
        
        Args:
            sample_rate: Sample rate (default 56 MHz)
            safety_margin_ms: Safety margin for packet cutting in milliseconds
        """
        self.sample_rate = sample_rate
        self.safety_margin_samples = int(safety_margin_ms * sample_rate / 1000)
        
        # Packet detection parameters
        self.power_threshold_db = -40  # Relative power threshold for packet detection
        self.min_packet_samples = int(0.01 * sample_rate)  # Minimum 10ms for packet
        self.max_packet_samples = int(0.5 * sample_rate)   # Maximum 500ms for packet
        
        # Packet grouping parameters
        self.frequency_tolerance_hz = 50e3  # Frequency tolerance for packet grouping (50 kHz)
        self.bandwidth_tolerance = 0.2  # Bandwidth tolerance (20%)
        
    def load_recording(self, file_path):
        """
        Load long recording from MAT file
        """
        print(f"📁 Loading recording: {os.path.basename(file_path)}")
        
        try:
            # Check file size
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"🗂️ File size: {file_size_mb:.1f}MB")
            
            # Load data
            data = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
            
            # Find main data
            if 'Y' in data:
                recording = data['Y']
            else:
                candidates = [k for k in data.keys() if not k.startswith('__')]
                if candidates:
                    recording = data[candidates[0]]
                else:
                    raise ValueError("No data found in file")
            
            # Ensure data is a vector
            if recording.ndim > 1:
                recording = recording.flatten()
            
            # Convert to more efficient type
            recording = recording.astype(np.complex64)
            
            duration_sec = len(recording) / self.sample_rate
            print(f"⏱️ Recording duration: {duration_sec:.2f} seconds")
            print(f"📊 Number of samples: {len(recording):,}")
            
            return recording
            
        except Exception as e:
            print(f"❌ Error loading recording: {e}")
            raise
    
    def detect_packets(self, recording):
        """
        Automatically detect packets in recording
        Returns list of tuples (start_idx, end_idx, power_db, center_freq, bandwidth)
        """
        print("🔍 Detecting packets in recording...")
        
        # Calculate instantaneous power
        instant_power = np.abs(recording)**2
        
        # Smooth to prevent noise
        window_size = max(1, int(0.001 * self.sample_rate))  # 1ms window
        power_smooth = np.convolve(instant_power, np.ones(window_size)/window_size, mode='same')
        
        # Convert to dB
        power_db = 10 * np.log10(power_smooth + 1e-12)
        noise_floor_db = np.percentile(power_db, 10)  # Noise floor level
        
        # Detect areas above threshold
        threshold_db = noise_floor_db + self.power_threshold_db
        above_threshold = power_db > threshold_db
        
        # Find continuous areas
        labeled, num_features = label(above_threshold)
        objects = find_objects(labeled)
        
        packets = []
        print(f"📡 Found {num_features} potential packet areas")
        
        for i, obj in enumerate(objects):
            if i % 10 == 0 and num_features > 20:
                print(f"🔄 Processing packet area {i+1}/{num_features}")
            if obj[0] is None:
                continue
                
            start_idx = obj[0].start
            end_idx = obj[0].stop
            duration_samples = end_idx - start_idx
            
            # Filter by packet length
            if duration_samples < self.min_packet_samples or duration_samples > self.max_packet_samples:
                continue
            
            # Add safety margin
            safe_start = max(0, start_idx - self.safety_margin_samples)
            safe_end = min(len(recording), end_idx + self.safety_margin_samples)
            
            # Calculate packet characteristics
            packet_data = recording[safe_start:safe_end]
            
            # Calculate average power
            avg_power_db = np.mean(power_db[start_idx:end_idx])
            
            # Optimized spectral analysis - use decimation for faster processing
            if len(packet_data) > 8192:
                # Decimate for faster processing while maintaining accuracy
                decimation_factor = len(packet_data) // 4096
                packet_data_decimated = packet_data[::decimation_factor]
                effective_sample_rate = self.sample_rate / decimation_factor
            else:
                packet_data_decimated = packet_data
                effective_sample_rate = self.sample_rate
            
            # Calculate center frequency and bandwidth using optimized FFT
            fft = np.fft.fft(packet_data_decimated)
            freqs = np.fft.fftfreq(len(packet_data_decimated), 1/effective_sample_rate)
            psd = np.abs(fft)**2
            
            # Find center frequency (frequency with maximum power)
            peak_idx = np.argmax(psd[:len(psd)//2])  # First half of the spectrum
            center_freq = abs(freqs[peak_idx])
            
            # Calculate bandwidth (3dB bandwidth) - optimized calculation
            max_power = np.max(psd[:len(psd)//2])
            half_power = max_power / 2
            above_half = psd[:len(psd)//2] > half_power
            
            if np.any(above_half):
                freq_indices = np.where(above_half)[0]
                bandwidth = (freq_indices[-1] - freq_indices[0]) * effective_sample_rate / len(packet_data_decimated)
            else:
                bandwidth = effective_sample_rate / len(packet_data_decimated)  # Minimum resolution
            
            packets.append({
                'start_idx': safe_start,
                'end_idx': safe_end,
                'original_start': start_idx,
                'original_end': end_idx,
                'power_db': avg_power_db,
                'center_freq': center_freq,
                'bandwidth': bandwidth,
                'duration_ms': (end_idx - start_idx) / self.sample_rate * 1000,
                'snr_db': avg_power_db - noise_floor_db
            })
        
        print(f"✅ Detected {len(packets)} valid packets")
        return packets
    
    def group_similar_packets(self, packets):
        """
        Group similar packets by frequency and bandwidth
        """
        print("🔗 Grouping similar packets...")
        
        groups = defaultdict(list)
        
        for packet in packets:
            # Find a matching group
            group_key = None
            for existing_key in groups.keys():
                ref_freq, ref_bw = existing_key
                
                # Check frequency proximity
                freq_diff = abs(packet['center_freq'] - ref_freq)
                if freq_diff > self.frequency_tolerance_hz:
                    continue
                
                # Check bandwidth proximity
                bw_ratio = abs(packet['bandwidth'] - ref_bw) / max(ref_bw, packet['bandwidth'])
                if bw_ratio > self.bandwidth_tolerance:
                    continue
                
                group_key = existing_key
                break
            
            # Create new group if no matching group found
            if group_key is None:
                group_key = (packet['center_freq'], packet['bandwidth'])
            
            groups[group_key].append(packet)
        
        print(f"📋 Created {len(groups)} packet groups")
        return groups
    
    def select_best_packet(self, packet_group):
        """
        Select the highest quality packet from a group
        """
        if len(packet_group) == 1:
            return packet_group[0]
        
        # Complex quality score
        best_packet = None
        best_score = -float('inf')
        
        for packet in packet_group:
            # Score based on:
            # 1. SNR (50% of score)
            # 2. Power (30% of score)  
            # 3. Duration (20% of score - longer packets are preferred)
            
            snr_score = packet['snr_db'] * 0.5
            power_score = packet['power_db'] * 0.3
            duration_score = packet['duration_ms'] * 0.2
            
            total_score = snr_score + power_score + duration_score
            
            if total_score > best_score:
                best_score = total_score
                best_packet = packet
        
        return best_packet
    
    def extract_packets(self, recording, packets):
        """
        Extract packets from the original recording
        """
        extracted_packets = []
        
        for packet in packets:
            start_idx = packet['start_idx']
            end_idx = packet['end_idx']
            
            packet_data = recording[start_idx:end_idx]
            extracted_packets.append({
                'data': packet_data,
                'metadata': packet
            })
        
        return extracted_packets
    
    def save_packets(self, extracted_packets, output_dir, base_name="packet"):
        """
        Save packets to a directory with the same MAT format as the original application
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        saved_files = []
        
        for i, packet_info in enumerate(extracted_packets):
            # Create filename with information about frequency and bandwidth
            freq_mhz = packet_info['metadata']['center_freq'] / 1e6
            bw_mhz = packet_info['metadata']['bandwidth'] / 1e6
            
            filename = f"{base_name}_{i+1:02d}_freq_{freq_mhz:.1f}MHz_bw_{bw_mhz:.1f}MHz.mat"
            filepath = os.path.join(output_dir, filename)
            
            # Save in the same format as the original application
            save_data = {
                'Y': packet_info['data'],
                'metadata': {
                    'center_freq': packet_info['metadata']['center_freq'],
                    'bandwidth': packet_info['metadata']['bandwidth'],
                    'power_db': packet_info['metadata']['power_db'],
                    'snr_db': packet_info['metadata']['snr_db'],
                    'duration_ms': packet_info['metadata']['duration_ms'],
                    'sample_rate': self.sample_rate,
                    'extraction_time': datetime.now().isoformat()
                }
            }
            
            sio.savemat(filepath, save_data)
            saved_files.append(filepath)
            
            print(f"💾 Saved: {filename} (Frequency: {freq_mhz:.1f}MHz, Power: {packet_info['metadata']['power_db']:.1f}dB)")
        
        return saved_files
    
    def analyze_recording(self, input_file, output_dir=None):
        """
        Primary function for analyzing long recording
        """
        print("🚀 Starting long recording analysis...")
        
        # Create output directory if not provided
        if output_dir is None:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join("extraction", f"long_recording_analysis_{base_name}_{timestamp}")
        
        # Ensure the extraction directory exists
        extraction_dir = os.path.dirname(output_dir) if os.path.dirname(output_dir) else "extraction"
        if not os.path.exists(extraction_dir):
            os.makedirs(extraction_dir)
        
        try:
            # Step 1: Load the recording
            recording = self.load_recording(input_file)
            
            # Step 2: Detect packets
            packets = self.detect_packets(recording)
            
            if not packets:
                print("⚠️ No packets found in recording")
                return
            
            # Step 3: Group similar packets
            groups = self.group_similar_packets(packets)
            
            # Step 4: Select the best packet from each group
            best_packets = []
            for group_key, packet_group in groups.items():
                best_packet = self.select_best_packet(packet_group)
                best_packets.append(best_packet)
                
                freq_mhz = group_key[0] / 1e6
                bw_mhz = group_key[1] / 1e6
                print(f"📊 Frequency group {freq_mhz:.1f}MHz: Selected packet with SNR {best_packet['snr_db']:.1f}dB from {len(packet_group)} packets")
            
            # Step 5: Extract the selected packets
            extracted_packets = self.extract_packets(recording, best_packets)
            
            # Step 6: Save
            saved_files = self.save_packets(extracted_packets, output_dir)
            
            # Summary
            print(f"\n✅ Analysis complete successfully!")
            print(f"📁 Output directory: {output_dir}")
            print(f"📦 Saved {len(saved_files)} high-quality packets")
            print(f"💽 Average packet size: {np.mean([len(ep['data']) for ep in extracted_packets]):.0f} samples")
            
            return {
                'output_dir': output_dir,
                'saved_files': saved_files,
                'packet_count': len(saved_files),
                'groups_found': len(groups),
                'total_packets_detected': len(packets)
            }
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            raise

def main():
    """Direct execution function from the command line"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python long_recording_analyzer.py <input_recording.mat>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        sys.exit(1)
    
    # Create analyzer
    analyzer = LongRecordingAnalyzer()
    
    # Analyze the recording
    result = analyzer.analyze_recording(input_file)
    
    if result:
        print(f"\n🎯 Analysis results:")
        for key, value in result.items():
            print(f"   {key}: {value}")

if __name__ == "__main__":
    main()