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
        
        # Packet detection parameters - VERY SENSITIVE for missing packets issue
        self.power_threshold_db = -60  # Much lower threshold (was -40dB) 
        self.min_packet_samples = int(0.0005 * sample_rate)  # Minimum 0.5ms for packet
        self.max_packet_samples = int(1.0 * sample_rate)   # Maximum 1000ms for packet
        
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
        
        # OPTIMIZATION 1: Smart downsampling - less aggressive to catch more packets
        if len(recording) > 50_000_000:  # Very large files
            downsample_factor = 10
        elif len(recording) > 10_000_000:  # Large files  
            downsample_factor = 5
        else:  # Smaller files - minimal downsampling
            downsample_factor = 2
            
        recording_fast = recording[::downsample_factor]
        fast_sample_rate = self.sample_rate / downsample_factor
        
        print(f"⚡ Using {downsample_factor}x downsampling for fast detection")
        
        # Calculate instantaneous power on downsampled signal
        instant_power = np.abs(recording_fast)**2
        
        # OPTIMIZATION 2: Much faster smoothing with smaller window to separate packets
        window_size = max(1, int(0.0001 * fast_sample_rate))  # 0.1ms window (was 1ms)
        if window_size > 1:
            # Use scipy uniform filter for faster smoothing
            from scipy.ndimage import uniform_filter1d
            power_smooth = uniform_filter1d(instant_power.astype(np.float32), size=window_size)
        else:
            power_smooth = instant_power
        
        # Convert to dB
        power_db = 10 * np.log10(power_smooth + 1e-12)
        noise_floor_db = np.percentile(power_db, 10)  # Noise floor level
        max_power_db = np.max(power_db)
        
        print(f"🔍 Detection stats: Noise floor: {noise_floor_db:.1f}dB, Max power: {max_power_db:.1f}dB, Dynamic range: {max_power_db-noise_floor_db:.1f}dB")
        
        # OPTIMIZATION 3: VERY sensitive threshold to catch all packets  
        # For missing packets issue: use relative threshold from max power instead of noise floor
        threshold_db = max_power_db - 20  # 20dB below max power (very sensitive)
        print(f"🎯 Using threshold: {threshold_db:.1f}dB")
        above_threshold = power_db > threshold_db
        
        # Debug: Show how many samples are above threshold
        samples_above = np.sum(above_threshold)
        print(f"🔍 Samples above threshold: {samples_above:,} out of {len(power_db):,} ({samples_above/len(power_db)*100:.2f}%)")
        
        # Find continuous areas
        labeled, num_features = label(above_threshold)
        objects = find_objects(labeled)
        
        print(f"📡 Found {num_features} potential packet areas (downsampled)")
        
        # ADDITIONAL: Try peak-based detection for better separation
        if num_features <= 2:  # If we found few areas, try peak detection
            from scipy.signal import find_peaks
            
            # Find peaks in power signal - VERY SENSITIVE
            peak_indices, _ = find_peaks(power_db, 
                                       height=threshold_db - 10,  # 10dB BELOW threshold (very sensitive)
                                       distance=int(0.001 * fast_sample_rate))  # Min 1ms between peaks
            
            print(f"🔍 Peak detection found {len(peak_indices)} potential packet centers")
            
            # Convert peaks to packet areas
            peak_objects = []
            for peak_idx in peak_indices:
                # Find the extent around each peak
                start_search = max(0, peak_idx - int(0.05 * fast_sample_rate))  # Search 50ms before peak
                end_search = min(len(power_db), peak_idx + int(0.05 * fast_sample_rate))  # Search 50ms after peak
                
                # Find where signal drops below threshold around the peak
                start_idx = peak_idx
                for i in range(peak_idx, start_search, -1):
                    if power_db[i] < threshold_db:
                        start_idx = i + 1
                        break
                        
                end_idx = peak_idx
                for i in range(peak_idx, end_search):
                    if power_db[i] < threshold_db:
                        end_idx = i
                        break
                
                if end_idx > start_idx + int(0.0005 * fast_sample_rate):  # At least 0.5ms duration
                    peak_objects.append((slice(start_idx, end_idx),))
                    
            if len(peak_objects) > len(objects):
                print(f"⚡ Using peak-based detection: {len(peak_objects)} packets vs {len(objects)} from threshold")
                objects = peak_objects
                num_features = len(objects)
        
        # OPTIMIZATION 4: Smart signal selection
        if num_features == 0:
            print("⚠️ No packets found with higher threshold, trying lower threshold...")
            # Try with even more sensitive threshold
            threshold_db = max_power_db - 30  # 30dB below max power (extremely sensitive)
            above_threshold = power_db > threshold_db
            labeled, num_features = label(above_threshold)
            objects = find_objects(labeled)
            print(f"📡 Found {num_features} potential packet areas with lower threshold")
            
            # If still no packets, try even lower threshold
            if num_features == 0:
                print("⚠️ Still no packets, trying EXTREMELY sensitive detection...")
                threshold_db = max_power_db - 50  # 50dB below max power (catch everything!)
                above_threshold = power_db > threshold_db
                labeled, num_features = label(above_threshold)
                objects = find_objects(labeled)
                print(f"📡 Found {num_features} potential packet areas with very low threshold")
        
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
            duration_ms = duration_samples / self.sample_rate * 1000
            
            if duration_samples < min_samples:
                print(f"⚠️ Skipping short packet: {duration_ms:.1f}ms (min: {min_samples/self.sample_rate*1000:.1f}ms)")
                continue
            if duration_samples > max_samples:
                print(f"⚠️ Skipping long packet: {duration_ms:.1f}ms (max: {max_samples/self.sample_rate*1000:.1f}ms)")
                continue
                
            print(f"✅ Valid packet found: {duration_ms:.1f}ms duration")
            
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
        
        # OPTIMIZATION: If we found very few packets, try without downsampling
        if len(packets) <= 1 and downsample_factor > 1:
            print(f"⚠️ Found only {len(packets)} packet(s), trying full-resolution detection...")
            
            # Calculate power on full resolution
            instant_power_full = np.abs(recording)**2
            
            # Faster smoothing for full resolution with small window
            window_size_full = max(1, int(0.0001 * self.sample_rate))  # 0.1ms window
            if window_size_full > 1:
                from scipy.ndimage import uniform_filter1d
                power_smooth_full = uniform_filter1d(instant_power_full.astype(np.float32), size=window_size_full)
            else:
                power_smooth_full = instant_power_full
            
            # Convert to dB
            power_db_full = 10 * np.log10(power_smooth_full + 1e-12)
            noise_floor_db_full = np.percentile(power_db_full, 10)
            
            # Use very sensitive threshold for full resolution  
            max_power_db_full = np.max(power_db_full)
            threshold_db_full = max_power_db_full - 25  # 25dB below max (very sensitive)
            above_threshold_full = power_db_full > threshold_db_full
            
            # Find areas
            labeled_full, num_features_full = label(above_threshold_full)
            objects_full = find_objects(labeled_full)
            
            print(f"📡 Full resolution found {num_features_full} potential packet areas")
            
            # Process full resolution packets
            for obj in objects_full:
                if obj[0] is None:
                    continue
                    
                start_idx = obj[0].start
                end_idx = obj[0].stop
                duration_samples = end_idx - start_idx
                duration_ms = duration_samples / self.sample_rate * 1000
                
                # Check length constraints
                if duration_samples < self.min_packet_samples:
                    continue
                if duration_samples > self.max_packet_samples:
                    continue
                
                # Add safety margin
                safe_start = max(0, start_idx - self.safety_margin_samples)
                safe_end = min(len(recording), end_idx + self.safety_margin_samples)
                
                # Calculate packet characteristics
                packet_data = recording[safe_start:safe_end]
                packet_power = np.abs(packet_data)**2
                avg_power_db = 10 * np.log10(np.mean(packet_power) + 1e-12)
                
                # Simple frequency estimation
                if len(packet_data) > 100:
                    sample_size = min(1000, len(packet_data))
                    sample_data = packet_data[:sample_size]
                    phase_diff = np.angle(sample_data[1:] * np.conj(sample_data[:-1]))
                    mean_phase_diff = np.mean(phase_diff)
                    estimated_freq = abs(mean_phase_diff * self.sample_rate / (2 * np.pi))
                    power_envelope = np.abs(sample_data)
                    power_var = np.var(power_envelope)
                    estimated_bandwidth = min(self.sample_rate / 4, power_var * 1e6)
                else:
                    estimated_freq = self.sample_rate / 8
                    estimated_bandwidth = self.sample_rate / 16
                
                packet_info = {
                    'start_idx': safe_start,
                    'end_idx': safe_end,
                    'power_db': avg_power_db,
                    'center_freq': estimated_freq,
                    'bandwidth': estimated_bandwidth,
                    'snr_db': avg_power_db - noise_floor_db_full
                }
                
                # Check if this is a new packet (not already found)
                is_new = True
                for existing_packet in packets:
                    overlap_start = max(packet_info['start_idx'], existing_packet['start_idx'])
                    overlap_end = min(packet_info['end_idx'], existing_packet['end_idx'])
                    if overlap_end > overlap_start:
                        overlap_ratio = (overlap_end - overlap_start) / (packet_info['end_idx'] - packet_info['start_idx'])
                        if overlap_ratio > 0.5:  # More than 50% overlap
                            is_new = False
                            break
                
                if is_new:
                    packets.append(packet_info)
                    print(f"✅ Found additional packet: {duration_ms:.1f}ms duration, {estimated_freq/1e6:.1f}MHz")
        
        elapsed_time = time.time() - start_time
        print(f"✅ Detection completed in {elapsed_time:.2f} seconds")
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