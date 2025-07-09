#!/usr/bin/env python3
"""
Quick performance test for heavy packet optimizations
"""

import time
import numpy as np
from utils import create_spectrogram

def test_performance():
    print("🚀 QUICK PERFORMANCE TEST")
    print("=" * 40)
    
    # Create heavy packet (10M samples)
    print("Creating heavy packet (10M samples)...")
    signal = np.random.random(10_000_000).astype(np.complex64)
    sample_rate = 56e6
    
    # Test optimized settings
    print("\nTesting optimized settings for heavy packets...")
    start_time = time.time()
    
    f, t, Sxx = create_spectrogram(
        signal,
        sample_rate,
        max_samples=1_000_000,    # Optimized: 1M samples
        time_resolution_us=50,    # Optimized: 50μs
        adaptive_resolution=True
    )
    
    process_time = time.time() - start_time
    print(f"✅ Processing time: {process_time:.2f}s")
    print(f"✅ Spectrogram shape: {Sxx.shape}")
    print(f"✅ Time resolution: {(t[1] - t[0]) * 1e6:.1f}μs")
    
    # Performance assessment
    if process_time <= 2.0:
        print("🎉 EXCELLENT: Under 2 seconds!")
    elif process_time <= 5.0:
        print("✅ GOOD: Under 5 seconds")
    else:
        print("⚠️ NEEDS IMPROVEMENT: Over 5 seconds")
    
    print(f"\nProcessing rate: {len(signal)/process_time/1e6:.1f}M samples/second")
    
    return process_time

if __name__ == "__main__":
    test_performance()