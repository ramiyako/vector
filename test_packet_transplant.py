#!/usr/bin/env python3
"""
Comprehensive tests for the packet transplant feature
Tests both the core functionality and integration with existing features
"""

import os
import sys
import unittest
import numpy as np
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add the parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    cross_correlate_signals,
    find_correlation_peak,
    extract_reference_segment,
    find_packet_location_in_vector,
    transplant_packet_in_vector,
    validate_transplant_quality,
    generate_sample_packet,
    save_vector,
    load_packet,
    get_sample_rate_from_mat
)

class TestCrossCorrelation(unittest.TestCase):
    """Test cross-correlation functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.sample_rate = 56e6
        self.test_signal1 = np.random.randn(1000) + 1j * np.random.randn(1000)
        self.test_signal2 = np.random.randn(2000) + 1j * np.random.randn(2000)
        
    def test_cross_correlate_signals_basic(self):
        """Test basic cross-correlation functionality"""
        correlation, lags = cross_correlate_signals(self.test_signal1, self.test_signal2)
        
        # Check output dimensions
        self.assertEqual(len(correlation), len(self.test_signal1) + len(self.test_signal2) - 1)
        self.assertEqual(len(lags), len(correlation))
        
        # Check that correlation is complex
        self.assertTrue(np.iscomplexobj(correlation))
        
    def test_cross_correlate_signals_modes(self):
        """Test different correlation modes"""
        for mode in ['full', 'valid', 'same']:
            correlation, lags = cross_correlate_signals(self.test_signal1, self.test_signal2, mode=mode)
            self.assertIsInstance(correlation, np.ndarray)
            self.assertIsInstance(lags, np.ndarray)
            self.assertEqual(len(correlation), len(lags))
            
    def test_cross_correlate_identical_signals(self):
        """Test cross-correlation with identical signals (should have high peak)"""
        signal = np.random.randn(500) + 1j * np.random.randn(500)
        correlation, lags = cross_correlate_signals(signal, signal)
        
        # Find peak
        peak_idx = np.argmax(np.abs(correlation))
        peak_lag = lags[peak_idx]
        
        # For identical signals, peak should be at zero lag
        self.assertEqual(peak_lag, 0)
        
    def test_find_correlation_peak(self):
        """Test correlation peak finding"""
        # Create a signal with a known peak
        correlation = np.array([0.1, 0.2, 0.9, 0.3, 0.1])
        lags = np.array([-2, -1, 0, 1, 2])
        
        peak_lag, peak_val, confidence = find_correlation_peak(correlation, lags)
        
        self.assertEqual(peak_lag, 0)
        self.assertEqual(peak_val, 0.9)
        self.assertGreater(confidence, 0.0)
        
    def test_find_correlation_peak_threshold(self):
        """Test correlation peak finding with threshold"""
        # Create a signal with low correlation
        correlation = np.array([0.1, 0.2, 0.25, 0.2, 0.1])
        lags = np.array([-2, -1, 0, 1, 2])
        
        peak_lag, peak_val, confidence = find_correlation_peak(correlation, lags, threshold_ratio=0.9)
        
        # Should have low confidence due to threshold
        self.assertEqual(confidence, 0.0)


class TestReferenceSegment(unittest.TestCase):
    """Test reference segment extraction"""
    
    def setUp(self):
        """Set up test data"""
        self.test_signal = np.random.randn(1000) + 1j * np.random.randn(1000)
        
    def test_extract_reference_segment(self):
        """Test reference segment extraction"""
        start_sample = 100
        end_sample = 200
        
        reference = extract_reference_segment(self.test_signal, start_sample, end_sample)
        
        self.assertEqual(len(reference), end_sample - start_sample)
        np.testing.assert_array_equal(reference, self.test_signal[start_sample:end_sample])
        
    def test_extract_reference_segment_bounds(self):
        """Test reference segment extraction with boundary conditions"""
        # Test with negative start
        reference = extract_reference_segment(self.test_signal, -10, 50)
        self.assertEqual(len(reference), 50)
        
        # Test with end beyond signal length
        reference = extract_reference_segment(self.test_signal, 500, 2000)
        self.assertEqual(len(reference), 500)
        
    def test_extract_reference_segment_invalid(self):
        """Test reference segment extraction with invalid parameters"""
        with self.assertRaises(ValueError):
            extract_reference_segment(self.test_signal, 200, 100)


class TestPacketTransplant(unittest.TestCase):
    """Test packet transplant core functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.sample_rate = 56e6
        
        # Create a vector with a known packet embedded
        self.vector_length = 10000
        self.packet_length = 1000
        self.packet_start = 3000
        
        # Create clean packet
        self.clean_packet = generate_sample_packet(
            self.packet_length / self.sample_rate, 
            self.sample_rate, 
            frequency=5e6
        )
        
        # Create corrupted packet (add noise)
        self.corrupted_packet = self.clean_packet + 0.5 * (
            np.random.randn(len(self.clean_packet)) + 1j * np.random.randn(len(self.clean_packet))
        )
        
        # Create vector with corrupted packet
        self.vector = np.random.randn(self.vector_length) + 1j * np.random.randn(self.vector_length)
        self.vector[self.packet_start:self.packet_start + self.packet_length] = self.corrupted_packet
        
        # Create reference segment from clean packet
        self.reference_start = 100
        self.reference_end = 200
        self.reference_segment = extract_reference_segment(
            self.clean_packet, 
            self.reference_start, 
            self.reference_end
        )
        
    def test_find_packet_location_in_vector(self):
        """Test finding packet location in vector"""
        vector_location, packet_location, confidence = find_packet_location_in_vector(
            self.vector, 
            self.clean_packet, 
            self.reference_segment
        )
        
        # Should find the packet location within reasonable tolerance
        self.assertAlmostEqual(vector_location, self.packet_start, delta=50)
        self.assertGreater(confidence, 0.3)
        
    def test_find_packet_location_with_search_window(self):
        """Test finding packet location with search window"""
        search_window = (self.packet_start - 500, self.packet_start + 500)
        
        vector_location, packet_location, confidence = find_packet_location_in_vector(
            self.vector, 
            self.clean_packet, 
            self.reference_segment,
            search_window=search_window
        )
        
        # Should find the packet location within search window
        self.assertGreaterEqual(vector_location, search_window[0])
        self.assertLessEqual(vector_location, search_window[1])
        
    def test_transplant_packet_in_vector(self):
        """Test packet transplant operation"""
        # Find optimal location
        vector_location, packet_location, confidence = find_packet_location_in_vector(
            self.vector, 
            self.clean_packet, 
            self.reference_segment
        )
        
        # Perform transplant
        transplanted_vector = transplant_packet_in_vector(
            self.vector, 
            self.clean_packet, 
            vector_location, 
            packet_location
        )
        
        # Check that vector length is preserved
        self.assertEqual(len(transplanted_vector), len(self.vector))
        
        # Check that transplanted region is different from original
        transplant_end = min(vector_location + len(self.clean_packet), len(self.vector))
        original_region = self.vector[vector_location:transplant_end]
        transplanted_region = transplanted_vector[vector_location:transplant_end]
        
        self.assertFalse(np.array_equal(original_region, transplanted_region))
        
    def test_transplant_packet_boundary_conditions(self):
        """Test packet transplant with boundary conditions"""
        # Test transplant at vector boundaries
        vector_location = len(self.vector) - 100
        
        transplanted_vector = transplant_packet_in_vector(
            self.vector, 
            self.clean_packet, 
            vector_location
        )
        
        # Should handle boundary gracefully
        self.assertEqual(len(transplanted_vector), len(self.vector))
        
    def test_validate_transplant_quality(self):
        """Test transplant quality validation"""
        # Perform transplant
        vector_location, packet_location, confidence = find_packet_location_in_vector(
            self.vector, 
            self.clean_packet, 
            self.reference_segment
        )
        
        transplanted_vector = transplant_packet_in_vector(
            self.vector, 
            self.clean_packet, 
            vector_location, 
            packet_location
        )
        
        # Validate quality
        validation_results = validate_transplant_quality(
            self.vector, 
            transplanted_vector, 
            self.clean_packet,
            vector_location, 
            self.reference_segment, 
            self.sample_rate
        )
        
        # Check validation results structure
        required_keys = [
            'reference_correlation', 'reference_confidence', 'power_ratio',
            'snr_improvement_db', 'transplant_length_samples', 'transplant_length_us',
            'time_precision_us', 'vector_location', 'success'
        ]
        
        for key in required_keys:
            self.assertIn(key, validation_results)
            
        # Check that time precision is reasonable (should be ~0.018 μs for 56MHz)
        expected_precision = 1e6 / self.sample_rate
        self.assertAlmostEqual(validation_results['time_precision_us'], expected_precision, places=3)


class TestExistingFunctionality(unittest.TestCase):
    """Test that existing functionality still works after adding transplant feature"""
    
    def setUp(self):
        """Set up test data"""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_rate = 56e6
        
    def tearDown(self):
        """Clean up temp directory"""
        shutil.rmtree(self.temp_dir)
        
    def test_generate_sample_packet(self):
        """Test that sample packet generation still works"""
        duration = 1e-3  # 1 ms
        frequency = 5e6  # 5 MHz
        
        packet = generate_sample_packet(duration, self.sample_rate, frequency)
        
        expected_length = int(duration * self.sample_rate)
        self.assertEqual(len(packet), expected_length)
        self.assertTrue(np.iscomplexobj(packet))
        
    def test_save_and_load_vector(self):
        """Test vector save and load functionality"""
        # Create test vector
        test_vector = np.random.randn(1000) + 1j * np.random.randn(1000)
        
        # Save vector
        file_path = os.path.join(self.temp_dir, "test_vector.mat")
        save_vector(test_vector, file_path)
        
        # Load vector
        loaded_vector = load_packet(file_path)
        
        # Check that loaded vector matches original
        np.testing.assert_array_almost_equal(test_vector, loaded_vector)
        
    def test_get_sample_rate_from_mat(self):
        """Test sample rate extraction from MAT file"""
        # Create test vector with embedded sample rate
        test_vector = np.random.randn(1000) + 1j * np.random.randn(1000)
        
        # Save vector
        file_path = os.path.join(self.temp_dir, "test_56MHz_vector.mat")
        save_vector(test_vector, file_path)
        
        # Extract sample rate from filename
        sample_rate = get_sample_rate_from_mat(file_path)
        
        # Should extract 56MHz from filename
        self.assertEqual(sample_rate, 56e6)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete transplant workflow"""
    
    def setUp(self):
        """Set up integration test data"""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_rate = 56e6
        
        # Create realistic test scenario
        self.create_test_scenario()
        
    def tearDown(self):
        """Clean up temp directory"""
        shutil.rmtree(self.temp_dir)
        
    def create_test_scenario(self):
        """Create realistic test scenario with vector and packet files"""
        # Create a clean packet
        packet_duration = 2e-3  # 2 ms
        packet_frequency = 10e6  # 10 MHz
        
        self.clean_packet = generate_sample_packet(
            packet_duration, 
            self.sample_rate, 
            packet_frequency
        )
        
        # Create a vector with multiple packets
        vector_duration = 20e-3  # 20 ms
        vector_length = int(vector_duration * self.sample_rate)
        
        self.vector = np.random.randn(vector_length) + 1j * np.random.randn(vector_length)
        
        # Insert clean packet at known location
        self.packet_location = int(5e-3 * self.sample_rate)  # 5 ms from start
        packet_end = self.packet_location + len(self.clean_packet)
        self.vector[self.packet_location:packet_end] = self.clean_packet
        
        # Create corrupted version of the packet
        self.corrupted_packet = self.clean_packet + 0.3 * (
            np.random.randn(len(self.clean_packet)) + 1j * np.random.randn(len(self.clean_packet))
        )
        
        # Create vector with corrupted packet
        self.corrupted_vector = self.vector.copy()
        self.corrupted_vector[self.packet_location:packet_end] = self.corrupted_packet
        
        # Save files
        self.vector_file = os.path.join(self.temp_dir, "test_vector.mat")
        self.packet_file = os.path.join(self.temp_dir, "clean_packet.mat")
        
        save_vector(self.corrupted_vector, self.vector_file)
        save_vector(self.clean_packet, self.packet_file)
        
    def test_complete_transplant_workflow(self):
        """Test complete packet transplant workflow"""
        # Load files
        vector_signal = load_packet(self.vector_file)
        packet_signal = load_packet(self.packet_file)
        
        # Extract reference segment
        ref_start = 100
        ref_end = 300
        reference_segment = extract_reference_segment(packet_signal, ref_start, ref_end)
        
        # Find optimal transplant location
        vector_location, packet_location, confidence = find_packet_location_in_vector(
            vector_signal, 
            packet_signal, 
            reference_segment
        )
        
        # Should find the packet location with good confidence
        self.assertGreater(confidence, 0.5)
        self.assertAlmostEqual(vector_location, self.packet_location, delta=100)
        
        # Perform transplant
        transplanted_vector = transplant_packet_in_vector(
            vector_signal, 
            packet_signal, 
            vector_location, 
            packet_location
        )
        
        # Validate results
        validation_results = validate_transplant_quality(
            vector_signal, 
            transplanted_vector, 
            packet_signal,
            vector_location, 
            reference_segment, 
            self.sample_rate
        )
        
        # Should report successful transplant
        self.assertTrue(validation_results['success'])
        self.assertGreater(validation_results['reference_confidence'], 0.5)
        self.assertGreater(validation_results['power_ratio'], 0.1)
        
    def test_microsecond_precision(self):
        """Test that transplant achieves microsecond precision"""
        # Load files
        vector_signal = load_packet(self.vector_file)
        packet_signal = load_packet(self.packet_file)
        
        # Extract reference segment
        reference_segment = extract_reference_segment(packet_signal, 50, 150)
        
        # Find optimal transplant location
        vector_location, packet_location, confidence = find_packet_location_in_vector(
            vector_signal, 
            packet_signal, 
            reference_segment
        )
        
        # Calculate precision
        time_precision = 1e6 / self.sample_rate  # in microseconds
        
        # At 56MHz, precision should be ~0.018 μs
        self.assertLess(time_precision, 1.0)  # Should be sub-microsecond
        
        # Verify that we can achieve sample-accurate positioning
        position_error_samples = abs(vector_location - self.packet_location)
        position_error_us = position_error_samples * time_precision
        
        # Should be within a few microseconds
        self.assertLess(position_error_us, 5.0)


class TestGUIIntegration(unittest.TestCase):
    """Test GUI integration (mocked)"""
    
    def setUp(self):
        """Set up GUI test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up temp directory"""
        shutil.rmtree(self.temp_dir)
        
    @patch('tkinter.filedialog.askopenfilename')
    @patch('tkinter.filedialog.asksaveasfilename')
    def test_gui_workflow_mock(self, mock_save_dialog, mock_open_dialog):
        """Test GUI workflow with mocked file dialogs"""
        # This would test the GUI components, but since we're running headless,
        # we'll mock the file dialogs and test the underlying logic
        
        # Create test files
        test_vector = np.random.randn(1000) + 1j * np.random.randn(1000)
        test_packet = np.random.randn(200) + 1j * np.random.randn(200)
        
        vector_file = os.path.join(self.temp_dir, "vector.mat")
        packet_file = os.path.join(self.temp_dir, "packet.mat")
        result_file = os.path.join(self.temp_dir, "result.mat")
        
        save_vector(test_vector, vector_file)
        save_vector(test_packet, packet_file)
        
        # Mock file dialogs
        mock_open_dialog.side_effect = [vector_file, packet_file]
        mock_save_dialog.return_value = result_file
        
        # Test that files can be loaded and saved
        self.assertTrue(os.path.exists(vector_file))
        self.assertTrue(os.path.exists(packet_file))
        
        # Test loading
        loaded_vector = load_packet(vector_file)
        loaded_packet = load_packet(packet_file)
        
        np.testing.assert_array_almost_equal(test_vector, loaded_vector)
        np.testing.assert_array_almost_equal(test_packet, loaded_packet)


def run_comprehensive_tests():
    """Run all tests and provide detailed report"""
    print("=" * 60)
    print("COMPREHENSIVE PACKET TRANSPLANT TESTS")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestCrossCorrelation,
        TestReferenceSegment,
        TestPacketTransplant,
        TestExistingFunctionality,
        TestIntegration,
        TestGUIIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, failure in result.failures:
            print(f"  - {test}: {failure}")
    
    if result.errors:
        print("\nERRORS:")
        for test, error in result.errors:
            print(f"  - {test}: {error}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOVERALL: {'PASSED' if success else 'FAILED'}")
    
    return success


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)