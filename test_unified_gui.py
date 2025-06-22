#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive test suite for the unified GUI application
Tests all components, functionality, and edge cases
"""

import unittest
import numpy as np
import os
import time
import tempfile
import shutil
from unittest.mock import patch, MagicMock

class TestUnifiedGUI(unittest.TestCase):
    """Comprehensive test suite for unified GUI"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        os.makedirs("data", exist_ok=True)
        
    def tearDown(self):
        """Clean up test environment"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)

class TestDependencies(TestUnifiedGUI):
    """Test 1: Dependencies and imports"""
    
    def test_dependencies_available(self):
        """Test if all required dependencies are available"""
        try:
            import customtkinter
            import numpy
            import scipy
            import matplotlib
            self.assertTrue(True, "All dependencies available")
        except ImportError as e:
            self.fail(f"Missing dependency: {e}")
    
    def test_utils_import(self):
        """Test if utils module can be imported"""
        try:
            from utils import (
                load_packet, 
                apply_frequency_shift, 
                create_spectrogram,
                save_vector
            )
            self.assertTrue(True, "Utils module imported successfully")
        except ImportError as e:
            self.fail(f"Cannot import utils: {e}")

class TestFileOperations(TestUnifiedGUI):
    """Test 2: File operations and data handling"""
    
    def test_data_directory_creation(self):
        """Test data directory creation"""
        test_data_dir = "test_data"
        os.makedirs(test_data_dir, exist_ok=True)
        self.assertTrue(os.path.exists(test_data_dir))
        
    def test_mat_file_save_load(self):
        """Test MAT file saving and loading"""
        import scipy.io as sio
        
        # Create test data
        test_signal = np.random.random(1000) + 1j * np.random.random(1000)
        test_signal = test_signal.astype(np.complex64)
        
        # Save to MAT file
        test_file = "test_signal.mat"
        sio.savemat(test_file, {'Y': test_signal})
        
        # Load from MAT file
        loaded_data = sio.loadmat(test_file, squeeze_me=True)
        loaded_signal = loaded_data['Y']
        
        # Verify data integrity
        np.testing.assert_array_almost_equal(test_signal, loaded_signal)
        
        # Clean up
        os.remove(test_file)

class TestNumpyOperations(TestUnifiedGUI):
    """Test 3: NumPy array operations"""
    
    def test_complex_array_creation(self):
        """Test complex array creation and manipulation"""
        # Test array creation
        signal = np.zeros(1000, dtype=np.complex64)
        self.assertEqual(signal.dtype, np.complex64)
        self.assertEqual(len(signal), 1000)
        
        # Test complex signal generation
        t = np.linspace(0, 0.001, 1000, endpoint=False)  # 1ms signal
        freq = 1e6  # 1 MHz
        complex_signal = np.exp(2j * np.pi * freq * t)
        self.assertEqual(complex_signal.dtype, np.complex128)
        
        # Convert to complex64
        complex_signal = complex_signal.astype(np.complex64)
        self.assertEqual(complex_signal.dtype, np.complex64)
    
    def test_fft_operations(self):
        """Test FFT operations for spectrogram creation"""
        # Create test signal
        fs = 56e6  # 56 MHz sample rate
        duration = 0.001  # 1 ms
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        freq = 5e6  # 5 MHz signal
        signal = np.exp(2j * np.pi * freq * t).astype(np.complex64)
        
        # Perform FFT
        fft_result = np.fft.fft(signal)
        self.assertEqual(len(fft_result), len(signal))
        
        # Check frequency domain properties
        freqs = np.fft.fftfreq(len(signal), 1/fs)
        peak_idx = np.argmax(np.abs(fft_result))
        peak_freq = abs(freqs[peak_idx])
        
        # Peak should be near the signal frequency (within 1% tolerance)
        self.assertLess(abs(peak_freq - freq) / freq, 0.01)

class TestUtilsFunctions(TestUnifiedGUI):
    """Test 4: Utils module functions"""
    
    def test_spectrogram_creation(self):
        """Test spectrogram creation"""
        from utils import create_spectrogram
        
        # Create test signal
        fs = 56e6
        duration = 0.001  # 1 ms
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        signal = np.exp(2j * np.pi * 5e6 * t).astype(np.complex64)
        
        # Create spectrogram
        f, t_spec, Sxx = create_spectrogram(signal, fs)
        
        # Verify output shapes
        self.assertGreater(len(f), 0)
        self.assertGreater(len(t_spec), 0)
        self.assertEqual(Sxx.shape, (len(f), len(t_spec)))
    
    def test_frequency_shift(self):
        """Test frequency shift operation"""
        from utils import apply_frequency_shift
        
        # Create test signal at 5 MHz
        fs = 56e6
        duration = 0.001
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        original_freq = 5e6
        signal = np.exp(2j * np.pi * original_freq * t).astype(np.complex64)
        
        # Apply frequency shift of 2 MHz
        shift_freq = 2e6
        shifted_signal = apply_frequency_shift(signal, shift_freq, fs)
        
        # Verify shift was applied (check that signal changed)
        self.assertFalse(np.allclose(signal, shifted_signal))
        
        # Verify output type
        self.assertEqual(shifted_signal.dtype, np.complex64)
    
    def test_packet_loading(self):
        """Test packet loading functionality"""
        from utils import load_packet
        import scipy.io as sio
        
        # Create test packet
        test_packet = np.random.random(1000) + 1j * np.random.random(1000)
        test_packet = test_packet.astype(np.complex64)
        
        # Save to file
        test_file = "data/test_packet.mat"
        sio.savemat(test_file, {'Y': test_packet})
        
        # Load using utils function
        loaded_packet = load_packet(test_file)
        
        # Verify data integrity
        np.testing.assert_array_almost_equal(test_packet, loaded_packet)
        self.assertEqual(loaded_packet.dtype, np.complex64)

class TestFullWorkflow(TestUnifiedGUI):
    """Test 5: Complete workflow from packet extraction to vector creation"""
    
    def test_complete_packet_to_vector_workflow(self):
        """Test the complete workflow: create signal -> extract packets -> build vector"""
        from utils import apply_frequency_shift, save_vector
        import scipy.io as sio
        
        print("Testing complete workflow...")
        
        # Step 1: Create original signal with multiple frequencies and noise
        fs = 56e6  # Sample rate
        duration = 0.0001  # 100 us (shorter for faster test)
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        
        # Create signal with 3 frequency components + noise
        freq1, freq2, freq3 = 5e6, 10e6, 15e6
        signal = (np.exp(2j * np.pi * freq1 * t) + 
                 0.7 * np.exp(2j * np.pi * freq2 * t) + 
                 0.5 * np.exp(2j * np.pi * freq3 * t))
        
        # Add noise
        noise = 0.1 * (np.random.random(len(t)) + 1j * np.random.random(len(t)))
        signal = (signal + noise).astype(np.complex64)
        
        print(f"Created original signal: {len(signal)} samples, {duration*1000:.1f} ms")
        
        # Step 2: Extract 3 different packets with different lengths
        packets = []
        packet_lengths = [500, 750, 1000]  # Different packet lengths
        
        for i, length in enumerate(packet_lengths):
            start_idx = i * 1000  # Non-overlapping packets
            end_idx = start_idx + length
            
            if end_idx <= len(signal):
                packet = signal[start_idx:end_idx]
                packet_file = f"data/test_packet_{i+1}.mat"
                sio.savemat(packet_file, {'Y': packet})
                packets.append({
                    'file': packet_file,
                    'data': packet,
                    'length': length
                })
                print(f"Extracted packet {i+1}: {length} samples")
        
        self.assertGreaterEqual(len(packets), 2, "Should have extracted at least 2 packets")
        
        # Step 3: Create complex vector with different parameters
        vector_length_ms = 0.2  # 200 us (shorter for test)
        vector_length = vector_length_ms / 1000.0  # Convert to seconds
        total_samples = int(vector_length * fs)
        vector = np.zeros(total_samples, dtype=np.complex64)
        
        # Packet configurations with different shifts, periods, and start times
        configs = [
            {'freq_shift': 2e6, 'period_ms': 50, 'start_time_ms': 0},    # Every 50 us, start immediately
            {'freq_shift': -3e6, 'period_ms': 75, 'start_time_ms': 10},  # Every 75 us, start at 10 us
            {'freq_shift': 1e6, 'period_ms': 60, 'start_time_ms': 25}   # Every 60 us, start at 25 us
        ]
        
        instances_count = 0
        
        for i, config in enumerate(configs[:len(packets)]):
            packet_data = packets[i]['data']
            
            # Apply frequency shift
            if config['freq_shift'] != 0:
                packet_data = apply_frequency_shift(packet_data, config['freq_shift'], fs)
                print(f"Applied {config['freq_shift']/1e6:.1f} MHz shift to packet {i+1}")
            
            # Calculate timing parameters
            period_samples = int(config['period_ms'] / 1000.0 * fs)
            start_offset = int(config['start_time_ms'] / 1000.0 * fs)
            
            # Insert packet instances
            current_pos = start_offset
            packet_instances = 0
            
            while current_pos + len(packet_data) <= total_samples:
                end_pos = current_pos + len(packet_data)
                vector[current_pos:end_pos] += packet_data
                packet_instances += 1
                instances_count += 1
                current_pos += period_samples
            
            print(f"Packet {i+1}: {packet_instances} instances, period {config['period_ms']} us")
        
        self.assertGreater(instances_count, 0, "Should have inserted at least one packet instance")
        
        # Step 4: Normalize vector
        max_val = np.max(np.abs(vector))
        if max_val > 0:
            vector = vector / max_val
            print("Vector normalized")
        
        # Step 5: Verify vector properties
        self.assertEqual(len(vector), total_samples)
        self.assertEqual(vector.dtype, np.complex64)
        self.assertLessEqual(np.max(np.abs(vector)), 1.0, "Normalized vector should have max amplitude <= 1")
        
        # Check that vector has significant energy (not mostly zeros)
        energy = np.sum(np.abs(vector)**2)
        energy_ratio = energy / len(vector)
        self.assertGreater(energy_ratio, 0.01, "Vector should have significant energy")
        
        # Step 6: Save vector
        vector_file = "test_vector_complete.mat"
        save_vector(vector, vector_file)
        self.assertTrue(os.path.exists(vector_file))
        
        # Verify saved vector
        loaded_data = sio.loadmat(vector_file, squeeze_me=True)
        loaded_vector = loaded_data['Y']
        np.testing.assert_array_almost_equal(vector, loaded_vector)
        
        print(f"Complete workflow test passed!")
        print(f"Final vector: {len(vector)} samples, {vector_length_ms} ms")
        print(f"Total packet instances: {instances_count}")
        print(f"Energy ratio: {energy_ratio:.4f}")
        
        # Clean up
        os.remove(vector_file)
        for packet in packets:
            if os.path.exists(packet['file']):
                os.remove(packet['file'])

class TestPerformance(TestUnifiedGUI):
    """Test 6: Performance and optimization"""
    
    def test_large_vector_creation(self):
        """Test creation of large vector (10 seconds)"""
        from utils import save_vector
        
        print("Testing large vector creation...")
        start_time = time.time()
        
        # Create 10 second vector at 56 MHz (but use lower sample rate for test)
        test_fs = 1e6  # 1 MHz for faster test
        duration = 0.01  # 10 ms (scaled down)
        total_samples = int(duration * test_fs)
        
        # Create vector
        vector = np.zeros(total_samples, dtype=np.complex64)
        
        # Add some test data
        test_packet = np.random.random(1000).astype(np.complex64)
        for i in range(0, total_samples - 1000, 5000):
            vector[i:i+1000] = test_packet
        
        creation_time = time.time() - start_time
        
        # Save vector
        save_time_start = time.time()
        save_vector(vector, "large_test_vector.mat")
        save_time = time.time() - save_time_start
        
        print(f"Large vector creation: {creation_time:.3f}s")
        print(f"Large vector save: {save_time:.3f}s")
        
        # Performance assertions
        self.assertLess(creation_time, 5.0, "Large vector creation should be under 5 seconds")
        self.assertLess(save_time, 5.0, "Large vector save should be under 5 seconds")
        
        # Clean up
        os.remove("large_test_vector.mat")
    
    def test_spectrogram_performance(self):
        """Test spectrogram creation performance"""
        from utils import create_spectrogram
        
        print("Testing spectrogram performance...")
        
        # Create test signal
        fs = 56e6
        duration = 0.001  # 1 ms
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        signal = np.exp(2j * np.pi * 5e6 * t).astype(np.complex64)
        
        start_time = time.time()
        f, t_spec, Sxx = create_spectrogram(signal, fs)
        spectrogram_time = time.time() - start_time
        
        print(f"Spectrogram creation: {spectrogram_time:.3f}s")
        
        # Performance assertion
        self.assertLess(spectrogram_time, 2.0, "Spectrogram creation should be under 2 seconds")
    
    def test_frequency_shift_performance(self):
        """Test frequency shift performance"""
        from utils import apply_frequency_shift
        
        print("Testing frequency shift performance...")
        
        # Create test signal
        fs = 56e6
        duration = 0.001  # 1 ms
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        signal = np.exp(2j * np.pi * 5e6 * t).astype(np.complex64)
        
        start_time = time.time()
        shifted_signal = apply_frequency_shift(signal, 2e6, fs)
        shift_time = time.time() - start_time
        
        print(f"Frequency shift: {shift_time:.3f}s")
        
        # Performance assertion
        self.assertLess(shift_time, 1.0, "Frequency shift should be under 1 second")

class TestEdgeCases(TestUnifiedGUI):
    """Test 7: Edge cases and error handling"""
    
    def test_empty_signal_handling(self):
        """Test handling of empty signals"""
        from utils import create_spectrogram
        
        # Test empty signal
        empty_signal = np.array([], dtype=np.complex64)
        
        # Should handle gracefully or raise appropriate error
        with self.assertRaises((ValueError, IndexError)):
            create_spectrogram(empty_signal, 56e6)
    
    def test_extreme_frequency_values(self):
        """Test handling of extreme frequency shift values"""
        from utils import apply_frequency_shift
        
        # Create test signal
        fs = 56e6
        signal = np.ones(1000, dtype=np.complex64)
        
        # Test very large frequency shift
        large_shift = fs  # Nyquist frequency
        try:
            shifted = apply_frequency_shift(signal, large_shift, fs)
            self.assertEqual(shifted.dtype, np.complex64)
        except Exception as e:
            # Should handle gracefully
            self.assertIsInstance(e, (ValueError, OverflowError))
        
        # Test negative frequency shift
        negative_shift = -10e6
        shifted_neg = apply_frequency_shift(signal, negative_shift, fs)
        self.assertEqual(shifted_neg.dtype, np.complex64)
    
    def test_invalid_file_handling(self):
        """Test handling of invalid or missing files"""
        from utils import load_packet
        
        # Test non-existent file
        with self.assertRaises((FileNotFoundError, OSError)):
            load_packet("non_existent_file.mat")
        
        # Test invalid file format
        invalid_file = "invalid_file.txt"
        with open(invalid_file, 'w') as f:
            f.write("This is not a MAT file")
        
        with self.assertRaises(Exception):
            load_packet(invalid_file)
        
        # Clean up
        os.remove(invalid_file)

def run_all_tests():
    """Run all test categories with detailed output"""
    print("="*60)
    print("UNIFIED GUI COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    test_categories = [
        ("Dependencies", TestDependencies),
        ("File Operations", TestFileOperations), 
        ("NumPy Operations", TestNumpyOperations),
        ("Utils Functions", TestUtilsFunctions),
        ("Complete Workflow", TestFullWorkflow),
        ("Performance", TestPerformance),
        ("Edge Cases", TestEdgeCases)
    ]
    
    total_start_time = time.time()
    all_results = []
    
    for category_name, test_class in test_categories:
        print(f"\n[{category_name}] Running tests...")
        print("-" * 40)
        
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=2, stream=open(os.devnull, 'w'))
        
        category_start = time.time()
        result = runner.run(suite)
        category_time = time.time() - category_start
        
        success = result.wasSuccessful()
        all_results.append((category_name, success, category_time, len(result.failures), len(result.errors)))
        
        status = "[PASS]" if success else "[FAIL]"
        print(f"[{category_name}] {status} ({category_time:.3f}s)")
        
        if not success:
            print(f"  Failures: {len(result.failures)}, Errors: {len(result.errors)}")
            for test, traceback in result.failures + result.errors:
                print(f"  - {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")
    
    # Summary
    total_time = time.time() - total_start_time
    passed_count = sum(1 for _, success, _, _, _ in all_results if success)
    total_count = len(all_results)
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for category, success, time_taken, failures, errors in all_results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} {category:20} ({time_taken:.3f}s)")
        if not success:
            print(f"    Failures: {failures}, Errors: {errors}")
    
    print("-" * 60)
    print(f"TOTAL: {passed_count}/{total_count} categories passed")
    print(f"TIME: {total_time:.3f} seconds")
    
    if passed_count == total_count:
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print("[FAIL] SOME TESTS FAILED")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1) 