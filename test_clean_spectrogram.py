#!/usr/bin/env python3
"""
Quick Spectrogram Cleanliness Test
=================================

Quick test to verify spectrogram cleanliness with visual output.
This can be run easily to check that spectrograms remain clean and noise-free.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from utils import create_spectrogram, plot_spectrogram, normalize_spectrogram

def test_clean_spectrogram():
    print("🧹 QUICK SPECTROGRAM CLEANLINESS TEST")
    print("=" * 50)
    
    # Create test signals similar to real packet data
    sr = 1000000  # 1 MHz sample rate
    
    # Test 1: Tone burst - should appear as clean horizontal line
    print("\n1. Testing tone burst (should be clean horizontal line)...")
    duration = 0.0005  # 500 μs
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq = 100000  # 100 kHz tone
    
    # Create windowed tone burst
    window = np.hanning(len(t))
    signal1 = window * np.exp(2j * np.pi * freq * t)
    
    # Test 2: Frequency sweep - should appear as clean diagonal line
    print("2. Testing frequency sweep (should be clean diagonal line)...")
    f_start = 50000   # 50 kHz
    f_end = 200000    # 200 kHz
    signal2 = window * np.exp(2j * np.pi * (f_start * t + (f_end - f_start) * t**2 / (2 * duration)))
    
    # Test 3: Multi-tone burst - should appear as multiple clean horizontal lines
    print("3. Testing multi-tone burst (should be multiple clean lines)...")
    signal3 = window * (
        0.7 * np.exp(2j * np.pi * 80000 * t) +   # 80 kHz
        0.5 * np.exp(2j * np.pi * 120000 * t) +  # 120 kHz
        0.3 * np.exp(2j * np.pi * 160000 * t)    # 160 kHz
    )
    
    # Create spectrograms with new clean functions
    print("\nCreating clean spectrograms...")
    f1, t1, Sxx1 = create_spectrogram(signal1, sr, time_resolution_us=5)
    f2, t2, Sxx2 = create_spectrogram(signal2, sr, time_resolution_us=5) 
    f3, t3, Sxx3 = create_spectrogram(signal3, sr, time_resolution_us=5)
    
    # Create comparison plots
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # Tone burst
    Sxx1_db, vmin1, vmax1 = normalize_spectrogram(Sxx1)
    im1 = axes[0,0].pcolormesh(t1 * 1000, f1 / 1000, Sxx1_db, 
                               shading='nearest', cmap='turbo', vmin=vmin1, vmax=vmax1)
    axes[0,0].set_title('Cleaned Tone Burst')
    axes[0,0].set_ylabel('Frequency [kHz]')
    plt.colorbar(im1, ax=axes[0,0], label='Power [dB]')
    
    # Time domain
    axes[0,1].plot(t * 1000, np.abs(signal1))
    axes[0,1].set_title('Time Domain - Tone Burst')
    axes[0,1].set_ylabel('Amplitude')
    axes[0,1].grid(True, alpha=0.3)
    
    # Frequency sweep
    Sxx2_db, vmin2, vmax2 = normalize_spectrogram(Sxx2)
    im2 = axes[1,0].pcolormesh(t2 * 1000, f2 / 1000, Sxx2_db, 
                               shading='nearest', cmap='turbo', vmin=vmin2, vmax=vmax2)
    axes[1,0].set_title('Cleaned Frequency Sweep')
    axes[1,0].set_ylabel('Frequency [kHz]')
    plt.colorbar(im2, ax=axes[1,0], label='Power [dB]')
    
    # Time domain
    axes[1,1].plot(t * 1000, np.abs(signal2))
    axes[1,1].set_title('Time Domain - Frequency Sweep')
    axes[1,1].set_ylabel('Amplitude')
    axes[1,1].grid(True, alpha=0.3)
    
    # Multi-tone
    Sxx3_db, vmin3, vmax3 = normalize_spectrogram(Sxx3)
    im3 = axes[2,0].pcolormesh(t3 * 1000, f3 / 1000, Sxx3_db, 
                               shading='nearest', cmap='turbo', vmin=vmin3, vmax=vmax3)
    axes[2,0].set_title('Cleaned Multi-tone Burst')
    axes[2,0].set_ylabel('Frequency [kHz]')
    axes[2,0].set_xlabel('Time [ms]')
    plt.colorbar(im3, ax=axes[2,0], label='Power [dB]')
    
    # Time domain
    axes[2,1].plot(t * 1000, np.abs(signal3))
    axes[2,1].set_title('Time Domain - Multi-tone Burst')
    axes[2,1].set_ylabel('Amplitude')
    axes[2,1].set_xlabel('Time [ms]')
    axes[2,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('clean_spectrogram_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Clean spectrogram test saved as 'clean_spectrogram_test.png'")
    
    # Print analysis
    print(f"\nCleaned Implementation Results:")
    print(f"Tone Burst     - Shape: {Sxx1.shape}, Dynamic range: {vmax1-vmin1:.1f} dB")
    print(f"Frequency Sweep - Shape: {Sxx2.shape}, Dynamic range: {vmax2-vmin2:.1f} dB")
    print(f"Multi-tone     - Shape: {Sxx3.shape}, Dynamic range: {vmax3-vmin3:.1f} dB")
    
    # Quick validation
    all_clean = True
    if vmax1 - vmin1 > 35:
        print(f"⚠️  WARNING: Tone burst dynamic range too high: {vmax1-vmin1:.1f} dB")
        all_clean = False
    if vmax2 - vmin2 > 35:
        print(f"⚠️  WARNING: Frequency sweep dynamic range too high: {vmax2-vmin2:.1f} dB")
        all_clean = False
    if vmax3 - vmin3 > 35:
        print(f"⚠️  WARNING: Multi-tone dynamic range too high: {vmax3-vmin3:.1f} dB")
        all_clean = False
    
    if all_clean:
        print(f"\n🎯 ✅ QUICK CLEANLINESS CHECK PASSED")
        print(f"Spectrograms are clean and noise-free!")
    else:
        print(f"\n⚠️  CLEANLINESS ISSUES DETECTED")
        print(f"Review spectrogram parameters!")
    
    print(f"\n🎯 QUICK TEST COMPLETE")
    print(f"Check 'clean_spectrogram_test.png' for visual verification")

if __name__ == "__main__":
    test_clean_spectrogram()