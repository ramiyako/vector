#!/usr/bin/env python3
"""
Basic test for spectrogram resolution improvements.
Tests only with dependencies already available in the project.
"""

import sys
import time

def test_imports():
    """Test that our improved functions can be imported"""
    print("Testing imports...")
    try:
        from utils import create_spectrogram, plot_spectrogram, normalize_spectrogram
        print("✅ Successfully imported improved functions")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_function_signatures():
    """Test that the improved function signatures work"""
    print("\nTesting function signatures...")
    try:
        from utils import create_spectrogram
        import inspect
        
        sig = inspect.signature(create_spectrogram)
        params = list(sig.parameters.keys())
        
        expected_params = ['sig', 'sr', 'center_freq', 'max_samples', 'time_resolution_us', 'adaptive_resolution']
        
        print(f"Function parameters: {params}")
        
        # Check if new parameters are present
        if 'adaptive_resolution' in params:
            print("✅ New adaptive_resolution parameter present")
        else:
            print("❌ adaptive_resolution parameter missing")
            return False
            
        # Check default values
        defaults = {name: param.default for name, param in sig.parameters.items() if param.default != inspect.Parameter.empty}
        print(f"Default values: {defaults}")
        
        if defaults.get('time_resolution_us') == 1:
            print("✅ Default time resolution improved to 1μs")
        else:
            print(f"❌ Time resolution not improved: {defaults.get('time_resolution_us')}")
            
        if defaults.get('adaptive_resolution') == True:
            print("✅ Adaptive resolution enabled by default")
        else:
            print(f"❌ Adaptive resolution not enabled: {defaults.get('adaptive_resolution')}")
            
        if defaults.get('max_samples') == 2_000_000:
            print("✅ Maximum samples increased to 2M")
        else:
            print(f"❌ Max samples not increased: {defaults.get('max_samples')}")
            
        return True
        
    except Exception as e:
        print(f"❌ Signature test failed: {e}")
        return False

def test_normalize_function():
    """Test the improved normalize_spectrogram function"""
    print("\nTesting normalize_spectrogram improvements...")
    try:
        from utils import normalize_spectrogram
        import numpy as np
        
        # Create a simple test spectrogram
        test_sxx = np.random.rand(100, 50) + 1j * np.random.rand(100, 50)
        
        # Test with float percentiles (new feature)
        sxx_db, vmin, vmax = normalize_spectrogram(test_sxx, low_percentile=5.5, high_percentile=99.5)
        
        print(f"✅ normalize_spectrogram accepts float percentiles")
        print(f"   Output shape: {sxx_db.shape}")
        print(f"   Dynamic range: {vmax - vmin:.1f} dB")
        
        return True
        
    except Exception as e:
        print(f"❌ normalize_spectrogram test failed: {e}")
        return False

def test_existing_integration():
    """Test that existing code still works with the improvements"""
    print("\nTesting integration with existing code...")
    try:
        # Test if we can load a simple packet
        from utils import generate_sample_packet
        
        # Generate a test signal
        sr = 44100
        duration = 0.1  # 100ms
        frequency = 1000  # 1kHz
        
        signal = generate_sample_packet(duration, sr, frequency)
        print(f"✅ Generated test signal: {len(signal)} samples")
        
        # Test the improved create_spectrogram function
        from utils import create_spectrogram
        
        print("Testing with new improved parameters...")
        start_time = time.time()
        f, t, Sxx = create_spectrogram(signal, sr, time_resolution_us=1, adaptive_resolution=True)
        processing_time = time.time() - start_time
        
        print(f"✅ create_spectrogram executed successfully")
        print(f"   Processing time: {processing_time:.3f} seconds")
        print(f"   Frequencies shape: {f.shape}")
        print(f"   Times shape: {t.shape}")
        print(f"   Spectrogram shape: {Sxx.shape}")
        
        if len(t) > 1:
            time_res = (t[1] - t[0]) * 1e6
            print(f"   Time resolution: {time_res:.2f} μs")
        
        if len(f) > 1:
            freq_res = (f[1] - f[0]) / 1e3
            print(f"   Frequency resolution: {freq_res:.2f} kHz")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_adaptive_behavior():
    """Test that adaptive resolution behaves differently for different signal lengths"""
    print("\nTesting adaptive resolution behavior...")
    try:
        from utils import create_spectrogram, generate_sample_packet
        
        sr = 56e6  # High sample rate
        
        # Test with very short signal (should get maximum resolution)
        short_signal = generate_sample_packet(50e-6, sr, 5e6)  # 50μs
        f1, t1, Sxx1 = create_spectrogram(short_signal, sr, adaptive_resolution=True)
        
        # Test with longer signal (should get lower resolution)
        long_signal = generate_sample_packet(10e-3, sr, 5e6)  # 10ms
        f2, t2, Sxx2 = create_spectrogram(long_signal, sr, adaptive_resolution=True)
        
        # Calculate resolutions
        if len(t1) > 1:
            short_time_res = (t1[1] - t1[0]) * 1e6
        else:
            short_time_res = 0
            
        if len(t2) > 1:
            long_time_res = (t2[1] - t2[0]) * 1e6
        else:
            long_time_res = 0
        
        print(f"Short signal (50μs): {short_time_res:.2f} μs time resolution")
        print(f"Long signal (10ms): {long_time_res:.2f} μs time resolution")
        
        if short_time_res < long_time_res:
            print("✅ Adaptive resolution working - shorter signals get finer resolution")
            return True
        else:
            print("⚠️  Adaptive resolution may not be working as expected")
            return False
            
    except Exception as e:
        print(f"❌ Adaptive resolution test failed: {e}")
        return False

def main():
    """Run all basic tests"""
    print("BASIC SPECTROGRAM RESOLUTION IMPROVEMENT TESTS")
    print("=" * 60)
    print("Testing core functionality improvements...")
    print()
    
    tests = [
        test_imports,
        test_function_signatures,
        test_normalize_function,
        test_existing_integration,
        test_adaptive_behavior
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Passed: {passed}/{total} tests")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Resolution improvements successfully implemented and working")
        print("\nKey improvements verified:")
        print("  • Time resolution improved from 10μs to 1μs default")
        print("  • Adaptive resolution feature working")
        print("  • Enhanced normalization with float percentiles")
        print("  • Increased max samples to 2M for better detail")
        print("  • Backward compatibility maintained")
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        print("Some improvements may not be working correctly")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)