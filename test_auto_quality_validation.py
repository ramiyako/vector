#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal test for automatic quality selection logic
Tests only the decision logic without requiring any GUI dependencies
"""

def auto_determine_quality(file_size_mb, signal_length):
    """Automatically determine quality based on file size and estimated analysis time"""
    
    # Estimate analysis time based on signal length
    # These are rough estimates based on typical performance
    estimated_time_fast = signal_length / 10_000_000  # Fast mode: ~10M samples per second
    estimated_time_balanced = signal_length / 5_000_000  # Balanced: ~5M samples per second
    estimated_time_high = signal_length / 2_000_000  # High quality: ~2M samples per second
    
    # Decision logic - prioritize analysis time over file size
    if estimated_time_high >= 30 or file_size_mb > 200:
        # Very long analysis time or very large files - use Fast mode
        if estimated_time_high >= 30:
            reason = f"Long analysis time ({estimated_time_high:.1f}s) requires fast mode"
        else:
            reason = f"Large file ({file_size_mb:.1f}MB) requires fast mode"
        recommended_quality = "Fast"
    elif estimated_time_high > 10 or file_size_mb > 50:
        # Moderate analysis time or medium files - use Balanced mode
        if estimated_time_high > 10:
            reason = f"Moderate analysis time ({estimated_time_high:.1f}s) suggests balanced mode"
        else:
            reason = f"Medium file ({file_size_mb:.1f}MB) suggests balanced mode"
        recommended_quality = "Balanced"
    else:
        # Fast analysis time and small files - use High Quality mode
        recommended_quality = "High Quality"
        reason = f"Small file ({file_size_mb:.1f}MB) and fast analysis ({estimated_time_high:.1f}s) allow high quality"
    
    return recommended_quality, reason, {
        "estimated_time_fast": estimated_time_fast,
        "estimated_time_balanced": estimated_time_balanced,
        "estimated_time_high": estimated_time_high
    }

def test_auto_quality_logic():
    """Test the auto quality determination logic"""
    
    print("🤖 Testing Auto Quality Selection Logic")
    print("=" * 50)
    
    # Test cases: (file_size_mb, signal_length, expected_quality)
    test_cases = [
        (10, 280_000, "High Quality"),      # Small file, short signal (0.1s)
        (30, 1_120_000, "High Quality"),   # Medium-small file, medium signal (0.6s)
        (75, 5_600_000, "Balanced"),       # Medium file, medium signal (2.8s)
        (150, 11_200_000, "Balanced"),     # Large file, long signal (5.6s)
        (250, 22_400_000, "Fast"),         # Very large file, very long signal (11.2s)
        (40, 22_400_000, "Balanced"),      # Small file but long signal (11.2s > 10s but < 30s)
    ]
    
    print("Test Results:")
    print("-" * 50)
    
    all_passed = True
    
    for i, (file_size_mb, signal_length, expected_quality) in enumerate(test_cases):
        try:
            recommended_quality, reason, time_estimates = auto_determine_quality(file_size_mb, signal_length)
            
            # Check if the result matches expected
            passed = recommended_quality == expected_quality
            status = "✅ PASS" if passed else "❌ FAIL"
            
            if not passed:
                all_passed = False
            
            print(f"Test {i+1}: {status}")
            print(f"  File Size: {file_size_mb:.1f}MB")
            print(f"  Signal Length: {signal_length:,} samples")
            print(f"  Expected: {expected_quality}")
            print(f"  Got: {recommended_quality}")
            print(f"  Reason: {reason}")
            print(f"  Times - Fast: {time_estimates['estimated_time_fast']:.1f}s, "
                  f"Balanced: {time_estimates['estimated_time_balanced']:.1f}s, "
                  f"High: {time_estimates['estimated_time_high']:.1f}s")
            print()
            
        except Exception as e:
            print(f"Test {i+1}: ❌ ERROR - {e}")
            all_passed = False
            print()
    
    print("=" * 50)
    if all_passed:
        print("🎉 All tests PASSED!")
        print("The auto-quality selection logic is working correctly.")
    else:
        print("⚠️  Some tests FAILED!")
        print("The auto-quality selection logic needs review.")
    
    return all_passed

def test_edge_cases():
    """Test edge cases for the auto-quality logic"""
    
    print("\n🧪 Testing Edge Cases")
    print("=" * 50)
    
    edge_cases = [
        # Test exact boundary conditions
        (50, 10_000_000, "High Quality"),   # Exactly at file size boundary (5.0s)
        (50.1, 10_000_000, "Balanced"),     # Just over file size boundary (5.0s)
        (200, 60_000_000, "Fast"),          # Exactly at file size boundary (30.0s)
        (200.1, 60_000_000, "Fast"),        # Just over file size boundary (30.0s)
        
        # Test time-based decisions
        (10, 20_000_000, "High Quality"),   # Small file, analysis time exactly 10.0s
        (10, 60_000_000, "Fast"),           # Small file, very long analysis time (30.0s)
        
        # Test extreme values
        (1, 1000, "High Quality"),          # Very small file
        (1000, 100_000_000, "Fast"),        # Very large file
    ]
    
    print("Edge Case Results:")
    print("-" * 50)
    
    for i, (file_size_mb, signal_length, expected_quality) in enumerate(edge_cases):
        try:
            recommended_quality, reason, time_estimates = auto_determine_quality(file_size_mb, signal_length)
            
            # Check if the result matches expected
            passed = recommended_quality == expected_quality
            status = "✅ PASS" if passed else "❌ FAIL"
            
            print(f"Edge Case {i+1}: {status}")
            print(f"  File Size: {file_size_mb:.1f}MB")
            print(f"  Signal Length: {signal_length:,} samples")
            print(f"  Expected: {expected_quality}")
            print(f"  Got: {recommended_quality}")
            print(f"  Reason: {reason}")
            print()
            
        except Exception as e:
            print(f"Edge Case {i+1}: ❌ ERROR - {e}")
            print()

def main():
    """Run all tests"""
    
    print("🧪 AUTO QUALITY LOGIC TESTS")
    print("=" * 60)
    
    try:
        # Test 1: Basic logic
        logic_passed = test_auto_quality_logic()
        
        # Test 2: Edge cases
        test_edge_cases()
        
        print("\n" + "=" * 60)
        print("📊 SUMMARY")
        print("=" * 60)
        
        if logic_passed:
            print("🎉 CORE LOGIC TESTS PASSED!")
            print("The auto-quality selection logic is working correctly.")
            print("\nThe algorithm uses this decision matrix:")
            print("• Small files (< 50MB) + Fast analysis (< 10s) → High Quality")
            print("• Medium files (50-200MB) + Medium analysis (10-30s) → Balanced")
            print("• Large files (> 200MB) + Long analysis (> 30s) → Fast")
            print("\nNote: The algorithm prioritizes analysis time over file size")
            print("      (if analysis time is long, it will use Fast mode regardless of file size)")
        else:
            print("⚠️  CORE LOGIC TESTS FAILED!")
            print("Please review the implementation.")
            
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()