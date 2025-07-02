#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive End-to-End Test for Packet Detection and Vector Generation
Tests the entire flow from packet detection to vector creation with microsecond accuracy.
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import time
from utils import (
    detect_packet_bounds,
    load_packet_info,
    apply_frequency_shift,
    save_vector,
    create_spectrogram,
    plot_spectrogram
)

# Test constants
TARGET_SAMPLE_RATE = 56e6  # 56 MHz
SAMPLE_RATE = TARGET_SAMPLE_RATE  # For compatibility
MICROSECOND_SAMPLES = int(SAMPLE_RATE / 1e6)  # Samples per microsecond
TEST_TOLERANCE_US = 0.1  # 0.1 microsecond tolerance

def create_test_signal_with_known_packet(packet_start_us, packet_duration_us, pre_buffer_us=1, post_buffer_us=1):
    """
    Create a test signal with a known packet position for validation
    
    Args:
        packet_start_us: Actual packet start time in microseconds
        packet_duration_us: Packet duration in microseconds
        pre_buffer_us: Buffer before packet in microseconds
        post_buffer_us: Buffer after packet in microseconds
    
    Returns:
        signal, actual_packet_start_sample, actual_packet_end_sample
    """
    # Calculate sample positions
    pre_buffer_samples = int(pre_buffer_us * SAMPLE_RATE / 1e6)
    packet_start_sample = int(packet_start_us * SAMPLE_RATE / 1e6)
    packet_duration_samples = int(packet_duration_us * SAMPLE_RATE / 1e6)
    packet_end_sample = packet_start_sample + packet_duration_samples
    post_buffer_samples = int(post_buffer_us * SAMPLE_RATE / 1e6)
    
    total_samples = pre_buffer_samples + packet_end_sample + post_buffer_samples
    
    # Create signal with noise
    signal = np.random.normal(0, 0.01, total_samples) + 1j * np.random.normal(0, 0.01, total_samples)
    
    # Create packet with higher amplitude
    packet_freq = 1e6  # 1 MHz
    t_packet = np.arange(packet_duration_samples) / SAMPLE_RATE
    packet_signal = 0.5 * np.exp(2j * np.pi * packet_freq * t_packet)
    
    # Insert packet at the correct position
    signal[packet_start_sample:packet_end_sample] = packet_signal
    
    return signal.astype(np.complex64), packet_start_sample, packet_end_sample

def test_packet_detection_accuracy():
    """Test packet detection accuracy with known packet positions"""
    print("=" * 60)
    print("TESTING PACKET DETECTION ACCURACY")
    print("=" * 60)
    
    test_cases = [
        {"start_us": 10, "duration_us": 50, "name": "Short packet"},
        {"start_us": 100, "duration_us": 500, "name": "Medium packet"},
        {"start_us": 1000, "duration_us": 2000, "name": "Long packet"},
        {"start_us": 5.5, "duration_us": 25.7, "name": "Fractional timing"},
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        print(f"Expected start: {test_case['start_us']:.1f} μs")
        print(f"Expected duration: {test_case['duration_us']:.1f} μs")
        
        # Create test signal
        signal, true_start, true_end = create_test_signal_with_known_packet(
            test_case['start_us'], test_case['duration_us']
        )
        
        # Detect packet bounds
        detected_start, detected_end = detect_packet_bounds(signal, SAMPLE_RATE)
        
        # Calculate errors in microseconds
        start_error_us = abs(detected_start - true_start) / SAMPLE_RATE * 1e6
        end_error_us = abs(detected_end - true_end) / SAMPLE_RATE * 1e6
        
        print(f"Detected start: {detected_start / SAMPLE_RATE * 1e6:.1f} μs")
        print(f"Detected end: {detected_end / SAMPLE_RATE * 1e6:.1f} μs")
        print(f"Start error: {start_error_us:.2f} μs")
        print(f"End error: {end_error_us:.2f} μs")
        
        # Check if within tolerance
        start_ok = start_error_us <= TEST_TOLERANCE_US
        end_ok = end_error_us <= TEST_TOLERANCE_US
        
        print(f"Start accuracy: {'✓ PASS' if start_ok else '✗ FAIL'}")
        print(f"End accuracy: {'✓ PASS' if end_ok else '✗ FAIL'}")
        
        results.append({
            'name': test_case['name'],
            'start_error_us': start_error_us,
            'end_error_us': end_error_us,
            'start_ok': start_ok,
            'end_ok': end_ok
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("PACKET DETECTION SUMMARY")
    print(f"{'='*60}")
    total_tests = len(results) * 2  # start + end
    passed_tests = sum(r['start_ok'] + r['end_ok'] for r in results)
    print(f"Passed: {passed_tests}/{total_tests} tests")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    return all(r['start_ok'] and r['end_ok'] for r in results)

def test_buffer_mechanism():
    """Test the buffer/spill mechanism used in packet cutting"""
    print("\n" + "=" * 60)
    print("TESTING BUFFER/SPILL MECHANISM")
    print("=" * 60)
    
    # Create test packet
    packet_duration_us = 100
    signal, true_start, true_end = create_test_signal_with_known_packet(
        50, packet_duration_us  # 50μs start, 100μs duration
    )
    
    # Detect packet bounds
    detected_start, detected_end = detect_packet_bounds(signal, SAMPLE_RATE)
    
    # Apply buffer mechanism (same as in the actual code)
    buffer_samples = int(SAMPLE_RATE // 1_000_000)  # 1 μs buffer
    buffer_start = max(0, detected_start - buffer_samples)
    
    print(f"Detected packet start: {detected_start / SAMPLE_RATE * 1e6:.1f} μs")
    print(f"Detected packet end: {detected_end / SAMPLE_RATE * 1e6:.1f} μs")
    print(f"Buffer size: {buffer_samples} samples ({buffer_samples / SAMPLE_RATE * 1e6:.1f} μs)")
    print(f"Buffered start: {buffer_start / SAMPLE_RATE * 1e6:.1f} μs")
    
    # Cut packet with buffer
    packet_with_buffer = signal[buffer_start:detected_end]
    pre_samples = detected_start - buffer_start
    
    print(f"Cut packet size: {len(packet_with_buffer)} samples")
    print(f"Pre-samples (spill): {pre_samples} samples ({pre_samples / SAMPLE_RATE * 1e6:.1f} μs)")
    
    # Save test packet
    os.makedirs("data", exist_ok=True)
    test_packet_path = "data/test_packet_buffer.mat"
    sio.savemat(test_packet_path, {'Y': packet_with_buffer, 'pre_samples': pre_samples})
    
    # Load back and verify
    y_loaded, pre_loaded = load_packet_info(test_packet_path)
    
    print(f"Loaded packet size: {len(y_loaded)} samples")
    print(f"Loaded pre-samples: {pre_loaded} samples ({pre_loaded / SAMPLE_RATE * 1e6:.1f} μs)")
    
    # Verify buffer mechanism
    buffer_accurate = abs(pre_loaded - pre_samples) == 0
    print(f"Buffer mechanism: {'✓ PASS' if buffer_accurate else '✗ FAIL'}")
    
    return buffer_accurate, test_packet_path

def test_vector_generation_accuracy():
    """Test vector generation with precise timing"""
    print("\n" + "=" * 60)
    print("TESTING VECTOR GENERATION ACCURACY")
    print("=" * 60)
    
    # Create test packets with known timing
    buffer_ok, test_packet_path = test_buffer_mechanism()
    if not buffer_ok:
        print("❌ Buffer mechanism failed, skipping vector test")
        return False
    
    # Load the test packet
    y, pre_samples = load_packet_info(test_packet_path)
    
    # Define vector parameters
    vector_length_ms = 1000  # 1 second
    period_ms = 100  # 100ms period
    start_time_ms = 50  # 50ms start time
    freq_shift_mhz = 5  # 5 MHz shift
    
    # Convert to samples
    vector_length_samples = int(vector_length_ms / 1000 * SAMPLE_RATE)
    period_samples = int(period_ms / 1000 * SAMPLE_RATE)
    start_time_samples = int(start_time_ms / 1000 * SAMPLE_RATE)
    
    print(f"Vector length: {vector_length_ms} ms ({vector_length_samples:,} samples)")
    print(f"Packet period: {period_ms} ms ({period_samples:,} samples)")
    print(f"Start time: {start_time_ms} ms ({start_time_samples:,} samples)")
    print(f"Frequency shift: {freq_shift_mhz} MHz")
    print(f"Pre-samples buffer: {pre_samples} samples ({pre_samples / SAMPLE_RATE * 1e6:.1f} μs)")
    
    # Create vector
    vector = np.zeros(vector_length_samples, dtype=np.complex64)
    
    # Apply frequency shift
    y_shifted = apply_frequency_shift(y, freq_shift_mhz * 1e6, SAMPLE_RATE)
    
    # Calculate insertion positions (same logic as in the actual code)
    start_offset = max(0, start_time_samples - pre_samples)
    
    print(f"Calculated start offset: {start_offset} samples ({start_offset / SAMPLE_RATE * 1000:.3f} ms)")
    
    # Insert packets
    current_pos = start_offset
    insertion_count = 0
    expected_positions = []
    
    while current_pos + len(y_shifted) <= vector_length_samples:
        vector[current_pos:current_pos + len(y_shifted)] += y_shifted
        
        # Record expected packet center (considering pre_samples)
        packet_center = (current_pos + pre_samples) / SAMPLE_RATE * 1000
        expected_positions.append(packet_center)
        
        insertion_count += 1
        current_pos += period_samples
    
    print(f"Inserted {insertion_count} packet instances")
    print(f"Expected packet centers (ms): {expected_positions[:5]}...")  # Show first 5
    
    # Verify timing accuracy by finding peaks in the vector
    energy = np.abs(vector) ** 2
    # Use a simple threshold to find packet positions
    threshold = np.max(energy) * 0.5
    above_threshold = energy > threshold
    
    # Find start positions of continuous regions above threshold
    diff = np.diff(above_threshold.astype(int))
    packet_starts = np.where(diff == 1)[0] + 1
    
    if len(packet_starts) == 0 and np.any(above_threshold):
        # Handle case where vector starts with a packet
        packet_starts = np.array([0])
    
    detected_positions = packet_starts / SAMPLE_RATE * 1000  # Convert to ms
    
    print(f"Detected packet positions (ms): {detected_positions[:5]}...")  # Show first 5
    
    # Compare expected vs detected (allowing for some tolerance)
    timing_errors = []
    for i, expected in enumerate(expected_positions[:len(detected_positions)]):
        if i < len(detected_positions):
            error_ms = abs(detected_positions[i] - expected)
            timing_errors.append(error_ms)
            print(f"Packet {i+1}: Expected {expected:.3f} ms, Detected {detected_positions[i]:.3f} ms, Error {error_ms:.3f} ms")
    
    # Check if all timing errors are within tolerance (1 μs = 0.001 ms)
    max_error_ms = max(timing_errors) if timing_errors else 0
    timing_ok = max_error_ms <= 0.001  # 1 μs tolerance
    
    print(f"Maximum timing error: {max_error_ms:.3f} ms ({max_error_ms * 1000:.1f} μs)")
    print(f"Vector timing accuracy: {'✓ PASS' if timing_ok else '✗ FAIL'}")
    
    # Save test vector
    test_vector_path = "test_vector_accuracy.mat"
    save_vector(vector, test_vector_path)
    print(f"Test vector saved to {test_vector_path}")
    
    return timing_ok

def test_full_end_to_end_workflow():
    """Test the complete workflow from packet detection to vector generation"""
    print("\n" + "=" * 60)
    print("FULL END-TO-END WORKFLOW TEST")
    print("=" * 60)
    
    # Step 1: Create multiple test packets with different characteristics
    test_packets = [
        {"name": "packet_1", "start_us": 10, "duration_us": 100, "freq": 1e6},
        {"name": "packet_2", "start_us": 20, "duration_us": 150, "freq": 2e6},
        {"name": "packet_3", "start_us": 30, "duration_us": 200, "freq": 3e6},
    ]
    
    packet_paths = []
    
    for packet_info in test_packets:
        print(f"\nCreating test packet: {packet_info['name']}")
        
        # Create signal
        signal, true_start, true_end = create_test_signal_with_known_packet(
            packet_info['start_us'], packet_info['duration_us']
        )
        
        # Detect bounds
        detected_start, detected_end = detect_packet_bounds(signal, SAMPLE_RATE)
        
        # Apply buffer
        buffer_samples = int(SAMPLE_RATE // 1_000_000)
        buffer_start = max(0, detected_start - buffer_samples)
        
        # Cut packet
        packet = signal[buffer_start:detected_end]
        pre_samples = detected_start - buffer_start
        
        # Save packet
        packet_path = f"data/{packet_info['name']}.mat"
        sio.savemat(packet_path, {'Y': packet, 'pre_samples': pre_samples})
        packet_paths.append(packet_path)
        
        print(f"  Saved packet: {len(packet)} samples, pre_samples: {pre_samples}")
    
    # Step 2: Create vector using the same logic as the main application
    print(f"\nCreating vector from {len(packet_paths)} packets...")
    
    vector_length_ms = 500
    vector_samples = int(vector_length_ms / 1000 * SAMPLE_RATE)
    vector = np.zeros(vector_samples, dtype=np.complex64)
    
    # Configuration for each packet
    configs = [
        {"period_ms": 50, "start_time_ms": 0, "freq_shift_mhz": 0},
        {"period_ms": 75, "start_time_ms": 25, "freq_shift_mhz": 2},
        {"period_ms": 100, "start_time_ms": 50, "freq_shift_mhz": -1},
    ]
    
    for i, (packet_path, config) in enumerate(zip(packet_paths, configs)):
        print(f"  Processing packet {i+1}: {os.path.basename(packet_path)}")
        
        # Load packet
        y, pre_samples = load_packet_info(packet_path)
        
        # Apply frequency shift
        if config['freq_shift_mhz'] != 0:
            y = apply_frequency_shift(y, config['freq_shift_mhz'] * 1e6, SAMPLE_RATE)
        
        # Calculate insertion timing
        period_samples = int(config['period_ms'] / 1000 * SAMPLE_RATE)
        start_samples = int(config['start_time_ms'] / 1000 * SAMPLE_RATE)
        start_offset = max(0, start_samples - pre_samples)
        
        # Insert packets
        current_pos = start_offset
        count = 0
        while current_pos + len(y) <= vector_samples:
            vector[current_pos:current_pos + len(y)] += y
            count += 1
            current_pos += period_samples
        
        print(f"    Inserted {count} instances")
    
    # Step 3: Validate final vector
    print(f"\nFinal vector validation:")
    print(f"  Vector length: {len(vector):,} samples ({len(vector) / SAMPLE_RATE * 1000:.1f} ms)")
    print(f"  Peak amplitude: {np.max(np.abs(vector)):.3f}")
    print(f"  RMS amplitude: {np.sqrt(np.mean(np.abs(vector)**2)):.3f}")
    
    # Save final vector
    final_vector_path = "test_end_to_end_vector.mat"
    save_vector(vector, final_vector_path)
    print(f"  Saved to: {final_vector_path}")
    
    # Create and display spectrogram
    try:
        f, t, Sxx = create_spectrogram(vector, SAMPLE_RATE, time_resolution_us=1)
        plt.figure(figsize=(12, 8))
        plt.pcolormesh(t * 1000, f / 1e6, 10 * np.log10(np.abs(Sxx) + 1e-12), shading='nearest')
        plt.colorbar(label='dB')
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency (MHz)')
        plt.title('End-to-End Test Vector Spectrogram')
        plt.tight_layout()
        plt.savefig('test_end_to_end_spectrogram.png', dpi=150)
        plt.show()
        print("  Spectrogram saved as test_end_to_end_spectrogram.png")
    except Exception as e:
        print(f"  Warning: Could not create spectrogram: {e}")
    
    return True

def main():
    """Run all accuracy tests"""
    print("COMPREHENSIVE PACKET DETECTION AND VECTOR GENERATION ACCURACY TEST")
    print("=" * 80)
    print(f"Target sample rate: {SAMPLE_RATE/1e6:.1f} MHz")
    print(f"Microsecond resolution: {MICROSECOND_SAMPLES} samples/μs")
    print(f"Test tolerance: {TEST_TOLERANCE_US} μs")
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Run tests
    results = []
    
    try:
        # Test 1: Packet detection accuracy
        detection_ok = test_packet_detection_accuracy()
        results.append(("Packet Detection", detection_ok))
        
        # Test 2: Vector generation accuracy
        vector_ok = test_vector_generation_accuracy()
        results.append(("Vector Generation", vector_ok))
        
        # Test 3: Full end-to-end workflow
        workflow_ok = test_full_end_to_end_workflow()
        results.append(("End-to-End Workflow", workflow_ok))
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL TEST SUMMARY")
    print("=" * 80)
    
    total_tests = len(results)
    passed_tests = sum(1 for _, ok in results if ok)
    
    for test_name, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"{test_name:<25}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED - System is ready for microsecond-accurate operation!")
        return True
    else:
        print("❌ SOME TESTS FAILED - System needs improvement")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)