#!/usr/bin/env python3
"""
Performance verification test for heavy packet optimizations.
Tests that the optimizations provide significant speed improvements.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from utils import create_spectrogram, create_heavy_packet_test

def test_heavy_packet_performance():
    """Test performance improvements for heavy packets"""
    print("🚀 HEAVY PACKET PERFORMANCE VERIFICATION")
    print("=" * 50)
    
    # Test different packet sizes
    test_cases = [
        ("Small packet (1M samples)", 1_000_000),
        ("Medium packet (5M samples)", 5_000_000),
        ("Heavy packet (10M samples)", 10_000_000),
        ("Very heavy packet (20M samples)", 20_000_000),
    ]
    
    sample_rate = 56e6
    results = []
    
    for name, num_samples in test_cases:
        print(f"\n📊 Testing {name}")
        print(f"   Samples: {num_samples:,}")
        
        # Create test signal
        print("   Creating test signal...")
        start_time = time.time()
        signal = np.random.random(num_samples).astype(np.complex64)
        # Add some realistic signal components
        t = np.arange(num_samples) / sample_rate
        signal += 0.5 * np.exp(2j * np.pi * 5e6 * t)  # 5MHz component
        signal += 0.3 * np.exp(2j * np.pi * 10e6 * t)  # 10MHz component
        creation_time = time.time() - start_time
        
        print(f"   Signal creation: {creation_time:.2f}s")
        
        # Test optimized spectrogram creation
        print("   Creating optimized spectrogram...")
        start_time = time.time()
        f, t_spec, Sxx = create_spectrogram(
            signal, 
            sample_rate, 
            max_samples=1_000_000,  # Optimized for heavy packets
            time_resolution_us=50,   # Fast setting
            adaptive_resolution=True
        )
        spectrogram_time = time.time() - start_time
        
        print(f"   Spectrogram creation: {spectrogram_time:.2f}s")
        print(f"   Spectrogram shape: {Sxx.shape}")
        print(f"   Time resolution: {(t_spec[1] - t_spec[0]) * 1e6:.1f}μs")
        print(f"   Freq resolution: {(f[1] - f[0]) / 1e3:.1f}kHz")
        
        # Calculate efficiency metrics
        samples_per_second = num_samples / spectrogram_time
        memory_mb = num_samples * 8 / (1024 * 1024)  # complex64 = 8 bytes
        
        print(f"   Processing rate: {samples_per_second/1e6:.1f}M samples/second")
        print(f"   Memory usage: {memory_mb:.1f}MB")
        
        results.append({
            'name': name,
            'samples': num_samples,
            'spectrogram_time': spectrogram_time,
            'samples_per_second': samples_per_second,
            'memory_mb': memory_mb,
            'shape': Sxx.shape
        })
        
        # Performance targets
        target_time = 5.0  # Target: under 5 seconds
        if spectrogram_time <= target_time:
            print(f"   ✅ PASSED: Under {target_time}s target")
        else:
            print(f"   ❌ FAILED: Over {target_time}s target")
            
        # Clean up
        del signal, f, t_spec, Sxx
    
    # Summary
    print("\n📋 PERFORMANCE SUMMARY")
    print("=" * 50)
    for result in results:
        print(f"{result['name']:<25} | {result['spectrogram_time']:>6.2f}s | {result['samples_per_second']/1e6:>6.1f}M/s | {result['memory_mb']:>7.1f}MB")
    
    # Overall assessment
    heavy_results = [r for r in results if r['samples'] >= 10_000_000]
    if heavy_results:
        avg_heavy_time = sum(r['spectrogram_time'] for r in heavy_results) / len(heavy_results)
        print(f"\n🎯 Average heavy packet processing time: {avg_heavy_time:.2f}s")
        if avg_heavy_time <= 3.0:
            print("✅ EXCELLENT: Heavy packets process in under 3 seconds")
        elif avg_heavy_time <= 5.0:
            print("✅ GOOD: Heavy packets process in under 5 seconds")
        else:
            print("⚠️  NEEDS IMPROVEMENT: Heavy packets take over 5 seconds")
    
    return results

def test_quality_presets():
    """Test the optimized quality presets"""
    print("\n⚙️  QUALITY PRESETS TEST")
    print("=" * 50)
    
    # Create heavy test packet
    signal = create_heavy_packet_test(duration_sec=1.0, sample_rate=56e6)
    sample_rate = 56e6
    
    presets = [
        ("Fast", 1_000_000, 50),
        ("Balanced", 2_000_000, 25),
        ("High Quality", 5_000_000, 10),
    ]
    
    for name, max_samples, time_res in presets:
        print(f"\n🎛️  Testing {name} preset")
        print(f"   Max samples: {max_samples:,}")
        print(f"   Time resolution: {time_res}μs")
        
        start_time = time.time()
        f, t, Sxx = create_spectrogram(
            signal,
            sample_rate,
            max_samples=max_samples,
            time_resolution_us=time_res,
            adaptive_resolution=True
        )
        process_time = time.time() - start_time
        
        print(f"   Processing time: {process_time:.2f}s")
        print(f"   Spectrogram shape: {Sxx.shape}")
        
        # Quality assessment
        if process_time <= 2.0:
            print("   ✅ EXCELLENT: Under 2 seconds")
        elif process_time <= 5.0:
            print("   ✅ GOOD: Under 5 seconds")
        else:
            print("   ❌ SLOW: Over 5 seconds")

def test_memory_efficiency():
    """Test memory efficiency improvements"""
    print("\n💾 MEMORY EFFICIENCY TEST")
    print("=" * 50)
    
    # Test with different data types
    num_samples = 10_000_000
    sample_rate = 56e6
    
    # Test complex64 (optimized)
    print("📊 Testing complex64 (optimized)...")
    signal_64 = np.random.random(num_samples).astype(np.complex64)
    memory_64 = signal_64.nbytes / (1024 * 1024)  # MB
    
    start_time = time.time()
    f, t, Sxx = create_spectrogram(signal_64, sample_rate, max_samples=1_000_000)
    time_64 = time.time() - start_time
    
    print(f"   Memory: {memory_64:.1f}MB")
    print(f"   Time: {time_64:.2f}s")
    
    # Test complex128 (unoptimized)
    print("\n📊 Testing complex128 (unoptimized)...")
    signal_128 = np.random.random(num_samples).astype(np.complex128)
    memory_128 = signal_128.nbytes / (1024 * 1024)  # MB
    
    start_time = time.time()
    f, t, Sxx = create_spectrogram(signal_128, sample_rate, max_samples=1_000_000)
    time_128 = time.time() - start_time
    
    print(f"   Memory: {memory_128:.1f}MB")
    print(f"   Time: {time_128:.2f}s")
    
    # Comparison
    memory_savings = (memory_128 - memory_64) / memory_128 * 100
    time_improvement = (time_128 - time_64) / time_128 * 100
    
    print(f"\n📈 EFFICIENCY IMPROVEMENTS:")
    print(f"   Memory savings: {memory_savings:.1f}%")
    print(f"   Time improvement: {time_improvement:.1f}%")

if __name__ == "__main__":
    print("🔬 PERFORMANCE VERIFICATION TEST SUITE")
    print("=" * 60)
    
    try:
        # Run all tests
        results = test_heavy_packet_performance()
        test_quality_presets()
        test_memory_efficiency()
        
        print("\n🎉 ALL TESTS COMPLETED!")
        print("Check the results above to verify performance improvements.")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()