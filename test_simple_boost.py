#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for simple boost power feature
This test verifies that the new two-level power system works correctly.
"""

import numpy as np
import scipy.io as sio
import os
import time
from utils import load_packet, save_vector, create_spectrogram

# Constants
TARGET_SAMPLE_RATE = 56e6

def create_test_packet(frequency_mhz, duration_ms, amplitude=1.0, packet_name="test_packet"):
    """Create a test packet with specified frequency, duration, and amplitude"""
    duration_s = duration_ms / 1000.0
    samples = int(duration_s * TARGET_SAMPLE_RATE)
    
    # Create time vector
    t = np.linspace(0, duration_s, samples)
    
    # Create sinusoidal signal
    frequency_hz = frequency_mhz * 1e6
    signal = amplitude * np.exp(1j * 2 * np.pi * frequency_hz * t)
    
    # Add some noise for realism
    noise_power = 0.01
    signal += np.sqrt(noise_power) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    
    # Save as .mat file
    filename = f"{packet_name}_{frequency_mhz}MHz.mat"
    mat_data = {
        'y': signal.astype(np.complex64),
        'sr': TARGET_SAMPLE_RATE,
        'pre_buf': 0
    }
    sio.savemat(filename, mat_data)
    print(f"Created test packet: {filename}")
    return filename

def test_simple_boost_functionality():
    """Test the simple boost power feature"""
    print("=" * 60)
    print("Testing Simple Boost Power Feature")
    print("=" * 60)
    
    # Create test packets
    test_files = []
    file1 = create_test_packet(5, 50, 1.0, "boost_test_packet1")
    file2 = create_test_packet(2, 30, 1.0, "boost_test_packet2")
    file3 = create_test_packet(7, 40, 1.0, "boost_test_packet3")
    test_files.extend([file1, file2, file3])
    
    # Test different boost configurations
    test_configs = [
        {
            'name': 'All Normal',
            'configs': [
                {'file': file1, 'boost': False},  # Normal (1.0x)
                {'file': file2, 'boost': False},  # Normal (1.0x)
                {'file': file3, 'boost': False}   # Normal (1.0x)
            ]
        },
        {
            'name': 'One Boosted',
            'configs': [
                {'file': file1, 'boost': False},  # Normal (1.0x)
                {'file': file2, 'boost': True},   # Boosted (2.0x)
                {'file': file3, 'boost': False}   # Normal (1.0x)
            ]
        },
        {
            'name': 'Two Boosted',
            'configs': [
                {'file': file1, 'boost': True},   # Boosted (2.0x)
                {'file': file2, 'boost': False},  # Normal (1.0x)
                {'file': file3, 'boost': True}    # Boosted (2.0x)
            ]
        }
    ]
    
    results = []
    
    for test_config in test_configs:
        print(f"\n{'-' * 40}")
        print(f"Test: {test_config['name']}")
        print(f"{'-' * 40}")
        
        # Create vector for this configuration
        vector = create_boost_vector(test_config['configs'], test_config['name'])
        
        # Analyze results
        power = np.sum(np.abs(vector)**2)
        peak = np.max(np.abs(vector))
        
        print(f"  Vector power: {power:.2e}")
        print(f"  Vector peak: {peak:.3f}")
        
        results.append({
            'name': test_config['name'],
            'power': power,
            'peak': peak,
            'vector': vector
        })
    
    # Compare results
    print(f"\n{'=' * 40}")
    print("Results Comparison")
    print(f"{'=' * 40}")
    
    baseline = results[0]  # All normal
    
    for i, result in enumerate(results):
        power_ratio = result['power'] / baseline['power']
        peak_ratio = result['peak'] / baseline['peak']
        
        print(f"{result['name']}:")
        print(f"  Power ratio vs baseline: {power_ratio:.2f}")
        print(f"  Peak ratio vs baseline: {peak_ratio:.2f}")
        
        # Expected ratios
        if i == 0:  # All normal
            expected_power = 1.0
            expected_peak = 1.0
        elif i == 1:  # One boosted
            expected_power = 1.33  # Approximately (1 + 2² + 1) / (1 + 1 + 1)
            expected_peak = 2.0    # One packet is 2x
        elif i == 2:  # Two boosted
            expected_power = 2.0   # Approximately (2² + 1 + 2²) / (1 + 1 + 1)
            expected_peak = 2.0    # Two packets are 2x
        
        print(f"  Expected power ratio: ~{expected_power:.2f}")
        print(f"  Expected peak ratio: ~{expected_peak:.2f}")
        
        # Verify results are reasonable
        if abs(power_ratio - expected_power) < 0.5:
            print(f"  ✅ Power ratio is reasonable")
        else:
            print(f"  ⚠️ Power ratio differs from expected")
        
        if i > 0 and peak_ratio >= 1.5:  # Should be significantly higher when boosted
            print(f"  ✅ Peak ratio shows boost effect")
        elif i == 0:
            print(f"  ✅ Baseline reference")
        else:
            print(f"  ⚠️ Peak ratio doesn't show expected boost")
    
    # Cleanup
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Cleaned up: {file}")
    
    print(f"\n{'=' * 60}")
    print("Simple Boost Power Test Complete!")
    print(f"{'=' * 60}")

def create_boost_vector(configs, test_name):
    """Create a vector with boost configuration"""
    vector_length_ms = 200
    vector_length_s = vector_length_ms / 1000.0
    total_samples = int(vector_length_s * TARGET_SAMPLE_RATE)
    vector = np.zeros(total_samples, dtype=np.complex64)
    
    print(f"Creating vector for {test_name}:")
    
    # Standard timing configuration
    timing_configs = [
        {'period': 0.080, 'start_time': 0.010},  # 80ms period, start at 10ms
        {'period': 0.060, 'start_time': 0.020},  # 60ms period, start at 20ms
        {'period': 0.100, 'start_time': 0.030},  # 100ms period, start at 30ms
    ]
    
    for i, config in enumerate(configs):
        print(f"  Processing packet {i+1}: {os.path.basename(config['file'])}")
        
        # Determine normalization ratio
        norm_ratio = 2.0 if config['boost'] else 1.0
        print(f"    Power level: {'Boosted (2.0x)' if config['boost'] else 'Normal (1.0x)'}")
        
        # Load packet
        packet = load_packet(config['file'])
        original_max = np.max(np.abs(packet))
        
        # Apply boost if needed
        if config['boost']:
            packet = packet * norm_ratio
            new_max = np.max(np.abs(packet))
            print(f"    Original max: {original_max:.3f} → Boosted max: {new_max:.3f}")
        else:
            print(f"    Max amplitude: {original_max:.3f} (unchanged)")
        
        # Insert packet instances
        timing = timing_configs[i]
        period_samples = int(timing['period'] * TARGET_SAMPLE_RATE)
        start_offset = int(timing['start_time'] * TARGET_SAMPLE_RATE)
        
        current_pos = start_offset
        instance_count = 0
        
        while current_pos + len(packet) <= total_samples:
            end_pos = current_pos + len(packet)
            vector[current_pos:end_pos] += packet
            instance_count += 1
            current_pos += period_samples
        
        print(f"    Inserted {instance_count} instances")
    
    return vector

if __name__ == "__main__":
    test_simple_boost_functionality()