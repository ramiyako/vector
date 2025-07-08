#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for automatic quality selection feature
Demonstrates how the application automatically selects quality based on file size and analysis time
"""

import numpy as np
import scipy.io as sio
import os
import time

def create_test_files():
    """Create test files of different sizes to demonstrate auto quality selection"""
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    sample_rate = 56e6
    test_files = []
    
    # Small file - should auto-select High Quality
    print("Creating small test file...")
    duration_small = 0.005  # 5ms
    t_small = np.linspace(0, duration_small, int(sample_rate * duration_small), endpoint=False)
    signal_small = np.exp(2j * np.pi * 10e6 * t_small) + 0.1 * np.random.randn(len(t_small))
    file_small = "data/test_small_auto.mat"
    sio.savemat(file_small, {"Y": signal_small.astype(np.complex64)})
    test_files.append((file_small, "Small", len(signal_small)))
    
    # Medium file - should auto-select Balanced
    print("Creating medium test file...")
    duration_medium = 0.02  # 20ms
    t_medium = np.linspace(0, duration_medium, int(sample_rate * duration_medium), endpoint=False)
    signal_medium = np.exp(2j * np.pi * 10e6 * t_medium) + 0.1 * np.random.randn(len(t_medium))
    file_medium = "data/test_medium_auto.mat"
    sio.savemat(file_medium, {"Y": signal_medium.astype(np.complex64)})
    test_files.append((file_medium, "Medium", len(signal_medium)))
    
    # Large file - should auto-select Fast
    print("Creating large test file...")
    duration_large = 0.1  # 100ms
    t_large = np.linspace(0, duration_large, int(sample_rate * duration_large), endpoint=False)
    signal_large = np.exp(2j * np.pi * 10e6 * t_large) + 0.1 * np.random.randn(len(t_large))
    file_large = "data/test_large_auto.mat"
    sio.savemat(file_large, {"Y": signal_large.astype(np.complex64)})
    test_files.append((file_large, "Large", len(signal_large)))
    
    return test_files

def test_auto_quality_decision():
    """Test the automatic quality decision logic"""
    
    # Import the automatic quality decision function
    from unified_gui import ModernPacketExtractor
    
    print("\n" + "="*60)
    print("AUTOMATIC QUALITY SELECTION TEST")
    print("="*60)
    
    # Create test files
    test_files = create_test_files()
    
    # Create a dummy packet extractor to test the auto quality function
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    frame = tk.Frame(root)
    extractor = ModernPacketExtractor(frame)
    
    # Test each file
    for file_path, size_category, signal_length in test_files:
        print(f"\n📁 Testing {size_category} file: {file_path}")
        
        # Get file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        # Test auto quality decision
        recommended_quality, reason, time_estimates = extractor.auto_determine_quality(file_size_mb, signal_length)
        
        print(f"   File size: {file_size_mb:.2f} MB")
        print(f"   Signal length: {signal_length:,} samples")
        print(f"   🤖 Auto-selected quality: {recommended_quality}")
        print(f"   📝 Reason: {reason}")
        print(f"   ⏱️ Estimated analysis times:")
        print(f"      • Fast: {time_estimates['estimated_time_fast']:.2f}s")
        print(f"      • Balanced: {time_estimates['estimated_time_balanced']:.2f}s")
        print(f"      • High Quality: {time_estimates['estimated_time_high']:.2f}s")
        
        # Verify the selection makes sense
        if size_category == "Small" and recommended_quality == "High Quality":
            print("   ✅ Correct selection for small file")
        elif size_category == "Medium" and recommended_quality == "Balanced":
            print("   ✅ Correct selection for medium file")
        elif size_category == "Large" and recommended_quality == "Fast":
            print("   ✅ Correct selection for large file")
        else:
            print(f"   ⚠️ Unexpected selection: {recommended_quality} for {size_category} file")
    
    root.destroy()
    
    print(f"\n" + "="*60)
    print("AUTOMATIC QUALITY SELECTION TEST COMPLETE")
    print("="*60)
    
    print("\n📋 Summary:")
    print("• Small files (< 50MB, < 10s analysis) → High Quality")
    print("• Medium files (50-200MB, 10-30s analysis) → Balanced")
    print("• Large files (> 200MB, > 30s analysis) → Fast")
    print("\n🎯 Benefits:")
    print("• Optimal performance for different file sizes")
    print("• Automatic decision saves user time")
    print("• User can still override if needed")
    print("• Provides clear reasoning for the selection")

def main():
    """Main test function"""
    print("🤖 AUTOMATIC QUALITY SELECTION TEST")
    print("=" * 60)
    
    try:
        test_auto_quality_decision()
        
        print("\n✅ Test completed successfully!")
        print("\nTo test the feature in the GUI:")
        print("1. Run: python unified_gui.py")
        print("2. Load one of the test files created in the data/ directory")
        print("3. The application will automatically select the quality")
        print("4. You can toggle 'Auto Quality Selection' to disable/enable")
        print("5. You can manually override the selection if needed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()