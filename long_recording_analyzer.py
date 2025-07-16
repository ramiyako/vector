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
import time # Added for performance tracking

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
        Fast packet detection optimized for speed (<10 seconds)
        Returns list of tuples (start_idx, end_idx, power_db, center_freq, bandwidth)
        """
        print("🔍 Fast packet detection starting...")
        start_time = time.time()
        
        # OPTIMIZATION 1: Downsample for initial detection (10x faster)
        downsample_factor = 10
        recording_fast = recording[::downsample_factor]
        fast_sample_rate = self.sample_rate / downsample_factor
        
        print(f"⚡ Using {downsample_factor}x downsampling for fast detection")
        
        # Calculate instantaneous power on downsampled signal
        instant_power = np.abs(recording_fast)**2
        
        # OPTIMIZATION 2: Much faster smoothing using simple moving average
        window_size = max(1, int(0.001 * fast_sample_rate))  # 1ms window
        if window_size > 1:
            # Use scipy uniform filter for faster smoothing
            from scipy.ndimage import uniform_filter1d
            power_smooth = uniform_filter1d(instant_power.astype(np.float32), size=window_size)
        else:
            power_smooth = instant_power
        
        # Convert to dB
        power_db = 10 * np.log10(power_smooth + 1e-12)
        noise_floor_db = np.percentile(power_db, 10)  # Noise floor level
        
        # OPTIMIZATION 3: More aggressive threshold for faster processing
        threshold_db = noise_floor_db + self.power_threshold_db + 10  # 10dB higher threshold
        above_threshold = power_db > threshold_db
        
        # Find continuous areas
        labeled, num_features = label(above_threshold)
        objects = find_objects(labeled)
        
        print(f"📡 Found {num_features} potential packet areas (downsampled)")
        
        # OPTIMIZATION 4: Limit to strongest signals only (max 20 packets)
        if num_features > 20:
            print("⚡ Too many areas detected, selecting strongest 20 signals")
            # Calculate power for each area and keep only the strongest
            area_powers = []
            for obj in objects:
                if obj[0] is not None:
                    area_power = np.mean(power_db[obj[0]])
                    area_powers.append((area_power, obj))
            
            # Sort by power and keep top 20
            area_powers.sort(reverse=True)
            objects = [obj for _, obj in area_powers[:20]]
            num_features = len(objects)
        
        packets = []
        
        for i, obj in enumerate(objects):
            if obj[0] is None:
                continue
                
            # Convert back to original sample indices
            start_idx = obj[0].start * downsample_factor
            end_idx = obj[0].stop * downsample_factor
            duration_samples = end_idx - start_idx
            
            # Filter by packet length
            min_samples = self.min_packet_samples
            max_samples = self.max_packet_samples
            if duration_samples < min_samples or duration_samples > max_samples:
                continue
            
            # Add safety margin
            safe_start = max(0, start_idx - self.safety_margin_samples)
            safe_end = min(len(recording), end_idx + self.safety_margin_samples)
            
            # OPTIMIZATION 5: Skip expensive FFT analysis, use simple estimates
            packet_data = recording[safe_start:safe_end]
            
            # Calculate average power from original signal
            packet_power = np.abs(packet_data)**2
            avg_power_db = 10 * np.log10(np.mean(packet_power) + 1e-12)
            
            # OPTIMIZATION 6: Fast frequency estimation without FFT
            # Use simple autocorrelation peak for dominant frequency
            if len(packet_data) > 1000:
                # Sample only a small portion for frequency estimation
                sample_size = min(1000, len(packet_data))
                sample_data = packet_data[:sample_size]
                
                # Simple spectral centroid estimation (much faster than FFT)
                # Use phase differences to estimate dominant frequency
                phase_diff = np.angle(sample_data[1:] * np.conj(sample_data[:-1]))
                mean_phase_diff = np.mean(phase_diff)
                estimated_freq = abs(mean_phase_diff * self.sample_rate / (2 * np.pi))
                
                # Simple bandwidth estimation based on power spread
                power_envelope = np.abs(sample_data)
                power_var = np.var(power_envelope)
                estimated_bandwidth = min(self.sample_rate / 4, power_var * 1e6)  # Rough estimate
            else:
                estimated_freq = self.sample_rate / 8  # Default estimate
                estimated_bandwidth = self.sample_rate / 16
            
            packets.append({
                'start_idx': safe_start,
                'end_idx': safe_end,
                'power_db': avg_power_db,
                'center_freq': estimated_freq,
                'bandwidth': estimated_bandwidth,
                'snr_db': avg_power_db - noise_floor_db
            })
        
        elapsed_time = time.time() - start_time
        print(f"✅ Fast detection completed in {elapsed_time:.2f} seconds")
        print(f"✅ Detected {len(packets)} valid packets")
        return packets
    
    def group_similar_packets(self, packets):
        """
        Fast packet grouping optimized for speed
        """
        print("🔗 Fast packet grouping...")
        start_time = time.time()
        
        # OPTIMIZATION: Skip complex grouping for small numbers of packets
        if len(packets) <= 5:
            print("⚡ Few packets detected, skipping complex grouping")
            groups = {}
            for i, packet in enumerate(packets):
                group_key = (packet['center_freq'], packet['bandwidth'])
                groups[group_key] = [packet]
            elapsed_time = time.time() - start_time
            print(f"✅ Fast grouping completed in {elapsed_time:.3f} seconds")
            return groups
        
        # OPTIMIZATION: Use simpler frequency-only grouping for speed
        groups = defaultdict(list)
        freq_tolerance = self.frequency_tolerance_hz * 2  # More relaxed tolerance for speed
        
        for packet in packets:
            # Simple frequency-based grouping (ignore bandwidth for speed)
            packet_freq = packet['center_freq']
            
            # Find closest existing group
            best_group_key = None
            min_freq_diff = float('inf')
            
            for existing_key in groups.keys():
                ref_freq = existing_key[0]
                freq_diff = abs(packet_freq - ref_freq)
                
                if freq_diff < freq_tolerance and freq_diff < min_freq_diff:
                    min_freq_diff = freq_diff
                    best_group_key = existing_key
            
            # Use existing group or create new one
            if best_group_key is not None:
                groups[best_group_key].append(packet)
            else:
                new_key = (packet_freq, packet['bandwidth'])
                groups[new_key].append(packet)
        
        elapsed_time = time.time() - start_time
        print(f"✅ Fast grouping completed in {elapsed_time:.3f} seconds")
        print(f"📋 Created {len(groups)} packet groups")
        return groups
    
    def select_best_packet(self, packet_group):
        """
        Fast packet selection - simply pick the highest SNR packet
        """
        if len(packet_group) == 1:
            return packet_group[0]
        
        # OPTIMIZATION: Simple SNR-based selection for speed
        best_packet = max(packet_group, key=lambda p: p['snr_db'])
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
            
            # Calculate duration in milliseconds
            duration_ms = len(packet_info['data']) / self.sample_rate * 1000
            
            # Save in the same format as the original application
            save_data = {
                'Y': packet_info['data'],
                'metadata': {
                    'center_freq': packet_info['metadata']['center_freq'],
                    'bandwidth': packet_info['metadata']['bandwidth'],
                    'power_db': packet_info['metadata']['power_db'],
                    'snr_db': packet_info['metadata']['snr_db'],
                    'duration_ms': duration_ms,
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
        Fast recording analysis optimized for <10 seconds
        """
        analysis_start_time = time.time()
        print("🚀 Starting FAST long recording analysis...")
        
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
            step_time = time.time()
            recording = self.load_recording(input_file)
            print(f"⏱️ Loading took: {time.time() - step_time:.2f}s")
            
            # Step 2: Fast packet detection
            step_time = time.time()
            packets = self.detect_packets(recording)
            print(f"⏱️ Detection took: {time.time() - step_time:.2f}s")
            
            if not packets:
                print("⚠️ No packets found in recording")
                return
            
            # Step 3: Fast grouping
            step_time = time.time()
            groups = self.group_similar_packets(packets)
            print(f"⏱️ Grouping took: {time.time() - step_time:.2f}s")
            
            # Step 4: Fast selection
            step_time = time.time()
            best_packets = []
            for group_key, packet_group in groups.items():
                best_packet = self.select_best_packet(packet_group)
                best_packets.append(best_packet)
                
                freq_mhz = group_key[0] / 1e6
                print(f"📊 Selected {freq_mhz:.1f}MHz packet (SNR: {best_packet['snr_db']:.1f}dB)")
            print(f"⏱️ Selection took: {time.time() - step_time:.2f}s")
            
            # Step 5: Fast extraction
            step_time = time.time()
            extracted_packets = self.extract_packets(recording, best_packets)
            print(f"⏱️ Extraction took: {time.time() - step_time:.2f}s")
            
            # Step 6: Fast save
            step_time = time.time()
            saved_files = self.save_packets(extracted_packets, output_dir)
            print(f"⏱️ Saving took: {time.time() - step_time:.2f}s")
            
            # Summary
            total_time = time.time() - analysis_start_time
            print(f"\n✅ FAST Analysis completed in {total_time:.2f} seconds!")
            print(f"📁 Output directory: {output_dir}")
            print(f"📦 Saved {len(saved_files)} high-quality packets")
            
            # Performance check
            if total_time > 10:
                print(f"⚠️ Warning: Analysis took {total_time:.2f}s (target: <10s)")
            else:
                print(f"🎯 SUCCESS: Analysis completed within {total_time:.2f}s target!")
            
            return {
                'output_dir': output_dir,
                'saved_files': saved_files,
                'packet_count': len(saved_files),
                'groups_found': len(groups),
                'total_packets_detected': len(packets),
                'analysis_time_seconds': total_time
            }
            
        except Exception as e:
            total_time = time.time() - analysis_start_time
            print(f"❌ Error after {total_time:.2f}s: {e}")
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