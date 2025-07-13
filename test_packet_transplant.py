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
        # Test full mode
        correlation_full, lags_full = cross_correlate_signals(self.test_signal1, self.test_signal2, mode='full')
        self.assertEqual(len(correlation_full), len(self.test_signal1) + len(self.test_signal2) - 1)
        
    def test_cross_correlate_identical_signals(self):
        """Test cross-correlation of identical signals"""
        # Create identical signals
        signal = np.random.randn(500) + 1j * np.random.randn(500)
        correlation, lags = cross_correlate_signals(signal, signal)
        
        # Find peak
        peak_lag, peak_val, confidence = find_correlation_peak(correlation, lags)
        
        # For identical signals, peak should be at zero lag with high confidence
        self.assertAlmostEqual(peak_lag, 0, places=0)
        self.assertGreater(confidence, 0.8)
        
    def test_find_correlation_peak(self):
        """Test correlation peak finding"""
        # Create a simple correlation with known peak
        lags = np.arange(-10, 11)
        correlation = np.exp(-lags**2 / 4)  # Gaussian peak at lag 0
        
        peak_lag, peak_val, confidence = find_correlation_peak(correlation, lags)
        
        self.assertAlmostEqual(peak_lag, 0, places=0)
        self.assertAlmostEqual(peak_val, 1.0, places=2)
        self.assertGreater(confidence, 0.9)
        
    def test_find_correlation_peak_threshold(self):
        """Test correlation peak finding with threshold"""
        # Create weak correlation
        lags = np.arange(-10, 11)
        correlation = np.ones_like(lags) * 0.1  # Weak uniform correlation
        
        peak_lag, peak_val, confidence = find_correlation_peak(correlation, lags, threshold_ratio=0.5)
        
        # Should have low confidence
        self.assertLess(confidence, 0.5)

class TestReferenceSegment(unittest.TestCase):
    """Test reference segment extraction"""
    
    def setUp(self):
        """Set up test data"""
        self.test_signal = np.random.randn(1000) + 1j * np.random.randn(1000)
        
    def test_extract_reference_segment(self):
        """Test basic reference segment extraction"""
        segment = extract_reference_segment(self.test_signal, 100, 200)
        
        self.assertEqual(len(segment), 100)
        np.testing.assert_array_equal(segment, self.test_signal[100:200])
        
    def test_extract_reference_segment_bounds(self):
        """Test reference segment extraction with boundary conditions"""
        # Test end beyond signal length
        segment = extract_reference_segment(self.test_signal, 900, 1100)
        self.assertEqual(len(segment), 100)  # Should be truncated
        
    def test_extract_reference_segment_invalid(self):
        """Test invalid reference segment parameters"""
        with self.assertRaises(ValueError):
            extract_reference_segment(self.test_signal, 200, 100)  # start > end

class TestPacketTransplant(unittest.TestCase):
    """Test packet transplant functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.sample_rate = 56e6
        
        # Create test vector with embedded packet
        self.vector_length = 5000
        self.packet_length = 500
        self.vector = np.random.randn(self.vector_length) + 1j * np.random.randn(self.vector_length)
        
        # Create a clean packet (sine wave)
        t = np.arange(self.packet_length) / self.sample_rate
        self.clean_packet = np.exp(1j * 2 * np.pi * 1e6 * t)  # 1 MHz tone
        
        # Embed the packet in the vector at known location
        self.embed_location = 1000
        self.vector[self.embed_location:self.embed_location + self.packet_length] = self.clean_packet
        
        # Create reference segment
        self.reference_segment = extract_reference_segment(self.clean_packet, 0, 100)
        
    def test_find_packet_location_in_vector(self):
        """Test finding packet location in vector"""
        vector_location, packet_location, confidence = find_packet_location_in_vector(
            self.vector, 
            self.clean_packet, 
            self.reference_segment
        )
        
        # Should find the embedded packet
        self.assertAlmostEqual(vector_location, self.embed_location, delta=10)
        self.assertGreater(confidence, 0.8)
        
    def test_find_packet_location_with_search_window(self):
        """Test finding packet location with search window"""
        search_window = (900, 1100)
        
        vector_location, packet_location, confidence = find_packet_location_in_vector(
            self.vector, 
            self.clean_packet, 
            self.reference_segment,
            search_window=search_window
        )
        
        # Should find the embedded packet within search window
        self.assertAlmostEqual(vector_location, self.embed_location, delta=10)
        self.assertGreater(confidence, 0.8)
        
    def test_transplant_packet_in_vector(self):
        """Test basic packet transplant functionality"""
        # Find packet location first
        vector_location, packet_location, confidence = find_packet_location_in_vector(
            self.vector, 
            self.clean_packet, 
            self.reference_segment
        )
        
        # Perform transplant with power normalization
        transplanted_vector = transplant_packet_in_vector(
            self.vector, 
            self.clean_packet, 
            vector_location, 
            packet_location,
            normalize_power=True
        )
        
        # Check that vector length is preserved
        self.assertEqual(len(transplanted_vector), len(self.vector))
        
        # Check that the transplanted region matches the packet
        transplanted_region = transplanted_vector[vector_location:vector_location + len(self.clean_packet)]
        # Allow for some difference due to power normalization
        correlation = np.abs(np.corrcoef(transplanted_region, self.clean_packet)[0, 1])
        self.assertGreater(correlation, 0.9)
        
    def test_transplant_packet_boundary_conditions(self):
        """Test transplant with boundary conditions"""
        # Test transplant near end of vector
        vector_location = len(self.vector) - 100
        
        transplanted_vector = transplant_packet_in_vector(
            self.vector, 
            self.clean_packet, 
            vector_location, 
            0,
            normalize_power=True
        )
        
        # Should handle boundary correctly
        self.assertEqual(len(transplanted_vector), len(self.vector))
        
    def test_validate_transplant_quality(self):
        """Test transplant quality validation"""
        # Find packet location first
        vector_location, packet_location, confidence = find_packet_location_in_vector(
            self.vector, 
            self.clean_packet, 
            self.reference_segment
        )
        
        # Perform transplant with power normalization
        transplanted_vector = transplant_packet_in_vector(
            self.vector, 
            self.clean_packet, 
            vector_location, 
            packet_location,
            normalize_power=True
        )
        
        # Validate transplant quality
        validation_results = validate_transplant_quality(
            self.vector,
            transplanted_vector,
            self.clean_packet,
            vector_location,
            self.reference_segment,
            self.sample_rate
        )
        
        # Check validation results
        self.assertIsInstance(validation_results, dict)
        self.assertIn('success', validation_results)
        self.assertIn('reference_correlation', validation_results)
        self.assertIn('power_ratio', validation_results)
        self.assertIn('snr_improvement_db', validation_results)
        self.assertIn('criteria', validation_results)
        
        # With power normalization, should have better results
        self.assertGreater(validation_results['power_ratio'], 0.01)
        
        # Check criteria details
        criteria = validation_results['criteria']
        self.assertIn('confidence_ok', criteria)
        self.assertIn('power_ok', criteria)
        self.assertIn('snr_ok', criteria)


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
    """Integration tests for complete workflow"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_rate = 56e6
        
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
        
    def create_test_scenario(self):
        """Create a test scenario with vector and packet files"""
        # Create test vector
        vector_length = 10000
        vector = np.random.randn(vector_length) + 1j * np.random.randn(vector_length)
        
        # Create test packet
        packet_length = 1000
        t = np.arange(packet_length) / self.sample_rate
        packet = np.exp(1j * 2 * np.pi * 2e6 * t)  # 2 MHz tone
        
        # Embed packet in vector at known location
        embed_location = 3000
        vector[embed_location:embed_location + packet_length] = packet
        
        # Save to files
        vector_file = os.path.join(self.temp_dir, 'test_vector.mat')
        packet_file = os.path.join(self.temp_dir, 'test_packet.mat')
        
        save_vector(vector, vector_file)
        save_vector(packet, packet_file)
        
        return {
            'vector_file': vector_file,
            'packet_file': packet_file,
            'vector': vector,
            'packet': packet,
            'embed_location': embed_location
        }
    
    def test_complete_transplant_workflow(self):
        """Test complete transplant workflow"""
        # Create test scenario
        scenario = self.create_test_scenario()
        
        # Load vector and packet
        vector = scenario['vector']
        packet = scenario['packet']
        
        # Extract reference segment
        reference_segment = extract_reference_segment(packet, 0, 100)
        
        # Find packet location
        vector_location, packet_location, confidence = find_packet_location_in_vector(
            vector, packet, reference_segment
        )
        
        # Should find the embedded packet
        self.assertAlmostEqual(vector_location, scenario['embed_location'], delta=50)
        self.assertGreater(confidence, 0.8)
        
        # Perform transplant with power normalization
        transplanted_vector = transplant_packet_in_vector(
            vector, packet, vector_location, packet_location, normalize_power=True
        )
        
        # Validate transplant
        validation_results = validate_transplant_quality(
            vector, transplanted_vector, packet, vector_location, 
            reference_segment, self.sample_rate
        )
        
        # Should pass validation with power normalization
        self.assertTrue(validation_results['success'])
        
        # Save result
        output_file = os.path.join(self.temp_dir, 'transplant_result.mat')
        save_vector(transplanted_vector, output_file)
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_file))
        
    def test_microsecond_precision(self):
        """Test microsecond precision timing"""
        scenario = self.create_test_scenario()
        
        vector = scenario['vector']
        packet = scenario['packet']
        reference_segment = extract_reference_segment(packet, 0, 100)
        
        # Find packet location
        vector_location, packet_location, confidence = find_packet_location_in_vector(
            vector, packet, reference_segment
        )
        
        # Calculate timing precision
        time_precision_us = 1e6 / self.sample_rate
        
        # Should be sub-microsecond precision at 56 MHz
        self.assertLess(time_precision_us, 1.0)
        
        # Perform transplant with power normalization
        transplanted_vector = transplant_packet_in_vector(
            vector, packet, vector_location, packet_location, normalize_power=True
        )
        
        # Validate timing precision
        validation_results = validate_transplant_quality(
            vector, transplanted_vector, packet, vector_location, 
            reference_segment, self.sample_rate
        )
        
        self.assertLess(validation_results['time_precision_us'], 1.0)


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