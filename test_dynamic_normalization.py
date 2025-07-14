#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for dynamic normalization feature
This test creates packets, applies different normalization ratios, 
and verifies the vector generation with dynamic normalization.
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import shutil
import time
from utils import (
    create_spectrogram, 
    plot_spectrogram,
    save_vector,
    load_packet
)

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
    print(f"  Frequency: {frequency_mhz} MHz")
    print(f"  Duration: {duration_ms} ms")
    print(f"  Amplitude: {amplitude}")
    print(f"  Samples: {len(signal)}")
    return filename

def test_dynamic_normalization():
    """Test the dynamic normalization feature"""
    print("=" * 60)
    print("Testing Dynamic Normalization Feature")
    print("=" * 60)
    
    # Create test packets with different properties
    test_files = []
    
    # Packet 1: High frequency, normal amplitude
    file1 = create_test_packet(5, 50, 1.0, "packet1")
    test_files.append(file1)
    
    # Packet 2: Low frequency, lower amplitude
    file2 = create_test_packet(2, 30, 0.5, "packet2")
    test_files.append(file2)
    
    # Packet 3: Mid frequency, higher amplitude
    file3 = create_test_packet(7, 40, 1.5, "packet3")
    test_files.append(file3)
    
    # Test 1: Create vector without dynamic normalization (all ratios = 1.0)
    print("\n" + "=" * 40)
    print("Test 1: Standard normalization (all ratios = 1.0)")
    print("=" * 40)
    
    vector_standard = create_test_vector(test_files, [1.0, 1.0, 1.0], "standard")
    
    # Test 2: Create vector with dynamic normalization
    print("\n" + "=" * 40)
    print("Test 2: Dynamic normalization (different ratios)")
    print("=" * 40)
    
    # Apply different normalization ratios
    norm_ratios = [0.5, 2.0, 1.5]  # Reduce first packet, boost second, slightly boost third
    vector_dynamic = create_test_vector(test_files, norm_ratios, "dynamic")
    
    # Test 3: Extreme normalization ratios
    print("\n" + "=" * 40)
    print("Test 3: Extreme normalization ratios")
    print("=" * 40)
    
    extreme_ratios = [0.1, 5.0, 0.8]  # Very low, very high, medium
    vector_extreme = create_test_vector(test_files, extreme_ratios, "extreme")
    
    # Analyze results
    print("\n" + "=" * 40)
    print("Results Analysis")
    print("=" * 40)
    
    analyze_vectors(vector_standard, vector_dynamic, vector_extreme)
    
    # Show spectrograms
    print("\n" + "=" * 40)
    print("Generating Spectrograms")
    print("=" * 40)
    
    show_comparison_spectrograms(vector_standard, vector_dynamic, vector_extreme)
    
    # Cleanup
    cleanup_test_files(test_files)
    
    print("\n" + "=" * 60)
    print("Dynamic Normalization Test Complete!")
    print("=" * 60)

def create_test_vector(packet_files, norm_ratios, test_name):
    """Create a vector from test packets with specified normalization ratios"""
    vector_length_ms = 200  # 200ms vector
    vector_length_s = vector_length_ms / 1000.0
    total_samples = int(vector_length_s * TARGET_SAMPLE_RATE)
    vector = np.zeros(total_samples, dtype=np.complex64)
    
    print(f"\nCreating {test_name} vector:")
    print(f"  Vector length: {vector_length_ms} ms ({total_samples:,} samples)")
    
    # Packet timing configuration
    packet_configs = [
        {'period': 0.080, 'start_time': 0.010},  # 80ms period, start at 10ms
        {'period': 0.060, 'start_time': 0.020},  # 60ms period, start at 20ms
        {'period': 0.100, 'start_time': 0.030},  # 100ms period, start at 30ms
    ]
    
    for i, (packet_file, norm_ratio) in enumerate(zip(packet_files, norm_ratios)):
        print(f"  Processing packet {i+1}: {packet_file}")
        print(f"    Normalization ratio: {norm_ratio:.1f}x")
        
        # Load packet
        packet = load_packet(packet_file)
        original_max = np.max(np.abs(packet))
        
        # Apply normalization ratio
        if norm_ratio != 1.0:
            packet = packet * norm_ratio
            new_max = np.max(np.abs(packet))
            print(f"    Original max amplitude: {original_max:.3f}")
            print(f"    New max amplitude: {new_max:.3f}")
            print(f"    Ratio applied: {new_max/original_max:.3f}")
        
        # Insert packet instances
        config = packet_configs[i]
        period_samples = int(config['period'] * TARGET_SAMPLE_RATE)
        start_offset = int(config['start_time'] * TARGET_SAMPLE_RATE)
        
        current_pos = start_offset
        instance_count = 0
        
        while current_pos + len(packet) <= total_samples:
            end_pos = current_pos + len(packet)
            vector[current_pos:end_pos] += packet
            instance_count += 1
            current_pos += period_samples
        
        print(f"    Inserted {instance_count} instances")
    
    # Save vector
    output_filename = f"test_vector_{test_name}.mat"
    save_vector(vector, output_filename)
    print(f"  Vector saved to: {output_filename}")
    
    return vector

def analyze_vectors(vector_standard, vector_dynamic, vector_extreme):
    """Analyze and compare the generated vectors"""
    print("\nVector Analysis:")
    
    # Power analysis
    power_standard = np.sum(np.abs(vector_standard)**2)
    power_dynamic = np.sum(np.abs(vector_dynamic)**2)
    power_extreme = np.sum(np.abs(vector_extreme)**2)
    
    print(f"  Standard vector power: {power_standard:.2e}")
    print(f"  Dynamic vector power: {power_dynamic:.2e}")
    print(f"  Extreme vector power: {power_extreme:.2e}")
    
    # Peak amplitude analysis
    peak_standard = np.max(np.abs(vector_standard))
    peak_dynamic = np.max(np.abs(vector_dynamic))
    peak_extreme = np.max(np.abs(vector_extreme))
    
    print(f"  Standard vector peak: {peak_standard:.3f}")
    print(f"  Dynamic vector peak: {peak_dynamic:.3f}")
    print(f"  Extreme vector peak: {peak_extreme:.3f}")
    
    # Verify normalization effects
    print(f"\nNormalization Effects:")
    print(f"  Dynamic/Standard power ratio: {power_dynamic/power_standard:.2f}")
    print(f"  Extreme/Standard power ratio: {power_extreme/power_standard:.2f}")
    print(f"  Dynamic/Standard peak ratio: {peak_dynamic/peak_standard:.2f}")
    print(f"  Extreme/Standard peak ratio: {peak_extreme/peak_standard:.2f}")

def show_comparison_spectrograms(vector_standard, vector_dynamic, vector_extreme):
    """Show spectrograms for comparison"""
    try:
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        vectors = [vector_standard, vector_dynamic, vector_extreme]
        titles = ['Standard Normalization (all 1.0x)', 
                 'Dynamic Normalization (0.5x, 2.0x, 1.5x)', 
                 'Extreme Normalization (0.1x, 5.0x, 0.8x)']
        
        for i, (vector, title) in enumerate(zip(vectors, titles)):
            # Create spectrogram
            f, t, Sxx = create_spectrogram(vector, TARGET_SAMPLE_RATE, time_resolution_us=5)
            
            # Plot spectrogram
            ax = axes[i]
            im = ax.pcolormesh(t, f/1e6, 10*np.log10(np.abs(Sxx)), shading='auto')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (MHz)')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Power (dB)')
        
        plt.tight_layout()
        plt.savefig('dynamic_normalization_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("  Comparison spectrograms saved to: dynamic_normalization_comparison.png")
        
    except Exception as e:
        print(f"  Error creating spectrograms: {e}")

def cleanup_test_files(test_files):
    """Clean up test files"""
    print("\nCleaning up test files:")
    
    # Remove test packet files
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"  Removed: {file}")
    
    # Remove test vector files
    vector_files = ['test_vector_standard.mat', 'test_vector_dynamic.mat', 'test_vector_extreme.mat']
    for file in vector_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"  Removed: {file}")

if __name__ == "__main__":
    test_dynamic_normalization()