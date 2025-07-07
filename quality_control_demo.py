#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quality Control Demo for Packet Extractor
Demonstrates the new quality control features for handling large audio files
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import os

# Import the utilities
from utils import create_spectrogram, normalize_spectrogram, plot_spectrogram
from unified_gui import UnifiedVectorApp

def create_demo_files():
    """Create demo files of different sizes to test quality control"""
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Demo file parameters
    sample_rate = 56e6
    
    # Create small file (fast to load)
    print("Creating small demo file...")
    duration_small = 0.001  # 1ms
    t_small = np.linspace(0, duration_small, int(sample_rate * duration_small), endpoint=False)
    signal_small = np.exp(2j * np.pi * 10e6 * t_small) + 0.1 * np.random.randn(len(t_small))
    sio.savemat("data/demo_small.mat", {"Y": signal_small.astype(np.complex64)})
    print(f"  Small file: {len(signal_small):,} samples ({duration_small*1000:.1f}ms)")
    
    # Create medium file 
    print("Creating medium demo file...")
    duration_medium = 0.01  # 10ms
    t_medium = np.linspace(0, duration_medium, int(sample_rate * duration_medium), endpoint=False)
    signal_medium = np.exp(2j * np.pi * 10e6 * t_medium) + 0.1 * np.random.randn(len(t_medium))
    sio.savemat("data/demo_medium.mat", {"Y": signal_medium.astype(np.complex64)})
    print(f"  Medium file: {len(signal_medium):,} samples ({duration_medium*1000:.1f}ms)")
    
    # Create large file (slow to load with high quality)
    print("Creating large demo file...")
    duration_large = 0.1  # 100ms
    t_large = np.linspace(0, duration_large, int(sample_rate * duration_large), endpoint=False)
    signal_large = np.exp(2j * np.pi * 10e6 * t_large) + 0.1 * np.random.randn(len(t_large))
    sio.savemat("data/demo_large.mat", {"Y": signal_large.astype(np.complex64)})
    print(f"  Large file: {len(signal_large):,} samples ({duration_large*1000:.1f}ms)")
    
    print("\nDemo files created successfully!")
    return signal_small, signal_medium, signal_large

def demo_quality_settings():
    """Demonstrate different quality settings and their performance"""
    
    print("\n" + "="*60)
    print("QUALITY CONTROL DEMONSTRATION")
    print("="*60)
    
    # Create demo files
    signal_small, signal_medium, signal_large = create_demo_files()
    
    # Quality settings to test
    quality_settings = [
        {
            "name": "Fast",
            "max_samples": 500_000,
            "time_resolution_us": 10,
            "adaptive_resolution": True,
            "description": "⚡ Fast: Quick loading for large files"
        },
        {
            "name": "Balanced", 
            "max_samples": 1_000_000,
            "time_resolution_us": 5,
            "adaptive_resolution": True,
            "description": "⚖️ Balanced: Good for most files"
        },
        {
            "name": "High Quality",
            "max_samples": 2_000_000,
            "time_resolution_us": 1,
            "adaptive_resolution": False,
            "description": "🎯 High Quality: Precise resolution"
        }
    ]
    
    # Test each quality setting
    sample_rate = 56e6
    
    for i, settings in enumerate(quality_settings):
        print(f"\n{i+1}. Testing {settings['name']} Quality Settings:")
        print(f"   {settings['description']}")
        print(f"   Max Samples: {settings['max_samples']:,}")
        print(f"   Time Resolution: {settings['time_resolution_us']} μs")
        print(f"   Adaptive Resolution: {settings['adaptive_resolution']}")
        
        # Test on large signal
        print(f"\n   Testing on large signal ({len(signal_large):,} samples)...")
        
        start_time = time.time()
        try:
            f, t, Sxx = create_spectrogram(
                signal_large,
                sample_rate,
                max_samples=settings['max_samples'],
                time_resolution_us=settings['time_resolution_us'],
                adaptive_resolution=settings['adaptive_resolution']
            )
            
            load_time = time.time() - start_time
            print(f"   ✅ Success! Processing time: {load_time:.3f} seconds")
            print(f"   Spectrogram shape: {Sxx.shape}")
            print(f"   Frequency bins: {len(f)}")
            print(f"   Time bins: {len(t)}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n" + "="*60)
    print("QUALITY CONTROL COMPARISON COMPLETE")
    print("="*60)
    
    print("\nKey Benefits of Quality Control:")
    print("• Fast mode: Quick loading of large files for initial inspection")
    print("• Balanced mode: Good compromise between speed and quality")
    print("• High Quality mode: Maximum precision for detailed analysis")
    print("• Adaptive resolution: Automatically adjusts based on signal characteristics")
    print("\nRecommended Usage:")
    print("1. Start with 'Fast' mode for large files to get quick overview")
    print("2. Switch to 'Balanced' for most analysis tasks")
    print("3. Use 'High Quality' only when precise analysis is needed")

def demo_comparison():
    """Create a visual comparison of different quality settings"""
    
    print("\n" + "="*60)
    print("VISUAL QUALITY COMPARISON")
    print("="*60)
    
    # Load the medium demo file
    data = sio.loadmat("data/demo_medium.mat")
    signal = data["Y"].flatten()
    sample_rate = 56e6
    
    # Create comparison plot
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    quality_settings = [
        ("Fast", 500_000, 10, True),
        ("Balanced", 1_000_000, 5, True),
        ("High Quality", 2_000_000, 1, False)
    ]
    
    for i, (name, max_samples, time_res, adaptive) in enumerate(quality_settings):
        print(f"Processing {name} quality...")
        
        start_time = time.time()
        f, t, Sxx = create_spectrogram(
            signal,
            sample_rate,
            max_samples=max_samples,
            time_resolution_us=time_res,
            adaptive_resolution=adaptive
        )
        process_time = time.time() - start_time
        
        # Normalize for display
        Sxx_db, vmin, vmax = normalize_spectrogram(Sxx)
        
        # Plot
        im = axes[i].pcolormesh(t * 1000, f / 1e6, Sxx_db, shading='nearest', cmap='viridis')
        axes[i].set_title(f"{name} Quality (Processing: {process_time:.3f}s, Shape: {Sxx.shape})")
        axes[i].set_xlabel("Time [ms]")
        axes[i].set_ylabel("Frequency [MHz]")
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i], label='Power [dB]')
    
    plt.tight_layout()
    plt.savefig("quality_comparison.png", dpi=150, bbox_inches='tight')
    print("Visual comparison saved as 'quality_comparison.png'")
    plt.show()

def main():
    """Main demonstration function"""
    
    print("🎛️ PACKET EXTRACTOR QUALITY CONTROL DEMO")
    print("=" * 60)
    
    try:
        # Run the demonstrations
        demo_quality_settings()
        demo_comparison()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE!")
        print("="*60)
        
        print("\nTo use the quality control features:")
        print("1. Run the unified GUI: python unified_gui.py")
        print("2. In the packet extractor tab, you'll see quality control options")
        print("3. Choose 'Fast' for quick loading of large files")
        print("4. Choose 'Balanced' for most analysis tasks")
        print("5. Choose 'High Quality' for detailed analysis")
        print("\nThe quality controls affect:")
        print("• Loading speed of large files")
        print("• Spectrogram resolution")
        print("• Memory usage")
        print("• Analysis precision")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()