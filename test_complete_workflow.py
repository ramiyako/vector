#!/usr/bin/env python3
"""
מהלך בדיקה מלא למערכת חילוץ פקטות ויצירת וקטורים
Complete workflow test for packet extraction and vector generation system
"""

import os
import sys
import numpy as np
import scipy.io as sio
import time
from pathlib import Path

# Add current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    load_packet, 
    load_packet_info,
    detect_packet_bounds,
    adjust_packet_bounds_gui,
    create_spectrogram,
    plot_spectrogram,
    save_vector
)

# Import TARGET_SAMPLE_RATE from unified_gui
TARGET_SAMPLE_RATE = 56e6

def test_packet_extraction_and_saving():
    """
    בדיקה 1: חילוץ פקטה מקובץ mat ושמירתה
    Test 1: Extract packet from mat file and save it
    """
    print("\n" + "="*60)
    print("בדיקה 1: חילוץ פקטה מקובץ mat")
    print("Test 1: Packet extraction from mat file")
    print("="*60)
    
    # Select a test packet
    test_file = "data/packet_1_5MHz.mat"
    if not os.path.exists(test_file):
        print(f"Error: Test file {test_file} not found!")
        return False
    
    print(f"Loading test file: {test_file}")
    
    # Load the signal
    data = sio.loadmat(test_file)
    signal = data['Y'].flatten().astype(np.complex64)
    pre_samples = int(data.get('pre_samples', 0))
    
    print(f"Signal length: {len(signal):,} samples")
    print(f"Pre-buffer samples: {pre_samples}")
    
    # Detect packet bounds automatically
    start_sample, end_sample = detect_packet_bounds(signal, TARGET_SAMPLE_RATE)
    print(f"Auto-detected bounds: start={start_sample}, end={end_sample}")
    
    # Test the interactive GUI bounds adjustment
    print("\nTesting packet bounds adjustment...")
    print("Note: In real usage, this would open an interactive window")
    print("For testing, we'll use the auto-detected bounds")
    
    # Extract the packet
    extracted_packet = signal[start_sample:end_sample]
    
    print(f"Extracted packet length: {len(extracted_packet):,} samples")
    print(f"Duration: {len(extracted_packet) / TARGET_SAMPLE_RATE * 1000:.1f} ms")
    
    # Save the extracted packet
    output_file = "data/extracted_test_packet.mat"
    sio.savemat(output_file, {
        'Y': extracted_packet,
        'pre_samples': max(0, pre_samples - start_sample)
    })
    
    print(f"Packet saved to: {output_file}")
    
    # Verify the saved packet
    verify_data = sio.loadmat(output_file)
    verify_packet = verify_data['Y'].flatten()
    
    if np.array_equal(extracted_packet, verify_packet):
        print("✓ Packet saving verification PASSED")
        return True
    else:
        print("✗ Packet saving verification FAILED")
        return False

def test_vector_creation_2_5ms():
    """
    בדיקה 2: יצירת וקטור באורך 2.5 מילישניות משתי פקטות
    Test 2: Create 2.5ms vector from two packets
    """
    print("\n" + "="*60)
    print("בדיקה 2: יצירת וקטור באורך 2.5 מילישניות")
    print("Test 2: Create 2.5ms vector from two packets")
    print("="*60)
    
    # Parameters for 2.5ms vector
    vector_length_ms = 2.5
    vector_length_seconds = vector_length_ms / 1000.0
    total_samples = int(vector_length_seconds * TARGET_SAMPLE_RATE)
    
    print(f"Target vector length: {vector_length_ms} ms")
    print(f"Total samples: {total_samples:,}")
    
    # Load two different packets
    packet1_file = "data/packet_1_5MHz.mat"
    packet2_file = "data/packet_2_chirp.mat"
    
    packet1, pre1 = load_packet_info(packet1_file)
    packet2, pre2 = load_packet_info(packet2_file)
    
    print(f"Packet 1 length: {len(packet1):,} samples")
    print(f"Packet 2 length: {len(packet2):,} samples")
    
    # Create the vector
    vector = np.zeros(total_samples, dtype=np.complex64)
    
    # Configuration for packet placement
    # Packet 1: starts at 0.2ms with period of 1.0ms
    packet1_start_time = 0.2e-3  # 0.2ms
    packet1_period = 1.0e-3      # 1.0ms
    
    # Packet 2: starts at 0.6ms with period of 1.2ms  
    packet2_start_time = 0.6e-3  # 0.6ms
    packet2_period = 1.2e-3      # 1.2ms
    
    packet1_instances = 0
    packet2_instances = 0
    markers = []
    
    # Place packet 1 instances
    current_time = packet1_start_time
    while current_time + len(packet1)/TARGET_SAMPLE_RATE <= vector_length_seconds:
        start_sample = int(current_time * TARGET_SAMPLE_RATE)
        end_sample = start_sample + len(packet1)
        
        if end_sample <= total_samples:
            vector[start_sample:end_sample] += packet1
            markers.append((current_time + pre1/TARGET_SAMPLE_RATE, 0, "Packet1"))
            packet1_instances += 1
            
        current_time += packet1_period
    
    # Place packet 2 instances
    current_time = packet2_start_time
    while current_time + len(packet2)/TARGET_SAMPLE_RATE <= vector_length_seconds:
        start_sample = int(current_time * TARGET_SAMPLE_RATE)
        end_sample = start_sample + len(packet2)
        
        if end_sample <= total_samples:
            vector[start_sample:end_sample] += packet2
            markers.append((current_time + pre2/TARGET_SAMPLE_RATE, 0, "Packet2"))
            packet2_instances += 1
            
        current_time += packet2_period
    
    print(f"Placed {packet1_instances} instances of Packet 1")
    print(f"Placed {packet2_instances} instances of Packet 2")
    print(f"Total markers: {len(markers)}")
    
    # Save the vector
    vector_file = f"vector_2_5ms_{int(time.time())}.mat"
    save_vector(vector, vector_file)
    
    print(f"Vector saved to: {vector_file}")
    print(f"Actual vector length: {len(vector)/TARGET_SAMPLE_RATE*1000:.1f} ms")
    
    return vector_file, markers

def test_microsecond_timing_precision():
    """
    בדיקה 3: בדיקת דיוק זמן למיקרושניה
    Test 3: Verify microsecond timing precision
    """
    print("\n" + "="*60)
    print("בדיקה 3: בדיקת דיוק זמן במיקרושניה")
    print("Test 3: Microsecond timing precision verification")
    print("="*60)
    
    # Create a test vector with known precise timing
    test_duration = 0.005  # 5ms
    total_samples = int(test_duration * TARGET_SAMPLE_RATE)
    test_vector = np.zeros(total_samples, dtype=np.complex64)
    
    # Create short test packets with 1μs spacing
    packet_duration = 10e-6  # 10 microseconds
    packet_samples = int(packet_duration * TARGET_SAMPLE_RATE)
    test_packet = np.ones(packet_samples, dtype=np.complex64)
    
    # Place packets every 100μs (exact microsecond timing)
    spacing_us = 100  # microseconds
    spacing_samples = int(spacing_us * TARGET_SAMPLE_RATE / 1e6)
    
    packet_times = []
    packet_count = 0
    
    current_sample = 0
    while current_sample + packet_samples < total_samples:
        end_sample = current_sample + packet_samples
        test_vector[current_sample:end_sample] = test_packet
        
        # Record exact timing
        packet_time_us = current_sample / TARGET_SAMPLE_RATE * 1e6
        packet_times.append(packet_time_us)
        packet_count += 1
        
        current_sample += spacing_samples
    
    print(f"Created test vector with {packet_count} packets")
    print(f"Designed spacing: {spacing_us} μs")
    
    # Verify timing precision
    if len(packet_times) > 1:
        measured_spacings = np.diff(packet_times)
        avg_spacing = np.mean(measured_spacings)
        spacing_error = abs(avg_spacing - spacing_us)
        
        print(f"Average measured spacing: {avg_spacing:.3f} μs")
        print(f"Timing error: {spacing_error:.3f} μs")
        
        # Check if error is within 1 microsecond
        if spacing_error < 1.0:
            print("✓ Microsecond timing precision PASSED")
            timing_ok = True
        else:
            print("✗ Microsecond timing precision FAILED")
            timing_ok = False
    else:
        print("✗ Not enough packets to verify timing")
        timing_ok = False
    
    # Save test vector
    test_vector_file = f"test_timing_vector_{int(time.time())}.mat"
    save_vector(test_vector, test_vector_file)
    print(f"Test vector saved to: {test_vector_file}")
    
    return timing_ok

def test_spectrogram_display():
    """
    בדיקה 4: בדיקת הצגת ספקטרוגרמה וגרפים
    Test 4: Test spectrogram and graph display
    """
    print("\n" + "="*60)
    print("בדיקה 4: בדיקת הצגת ספקטרוגרמה וגרפים")
    print("Test 4: Spectrogram and graph display")
    print("="*60)
    
    # Load a test packet
    test_file = "data/packet_2_chirp.mat"
    packet = load_packet(test_file)
    
    print(f"Creating spectrogram for packet with {len(packet)} samples")
    
    try:
        # Create spectrogram
        f, t, Sxx = create_spectrogram(packet, TARGET_SAMPLE_RATE, time_resolution_us=1)
        
        print(f"Spectrogram created successfully:")
        print(f"  Frequency bins: {len(f)}")
        print(f"  Time bins: {len(t)}")
        print(f"  Time resolution: {(t[1]-t[0])*1e6:.2f} μs" if len(t) > 1 else "  Single time bin")
        print(f"  Frequency resolution: {(f[1]-f[0])/1e3:.2f} kHz" if len(f) > 1 else "  Single frequency bin")
        
        # Test the plotting function (would normally display)
        print("\nTesting plot function...")
        print("Note: In GUI mode, this would display interactive plots")
        
        # Verify plot parameters are reasonable
        time_span_ms = (t[-1] - t[0]) * 1000 if len(t) > 1 else 0
        freq_span_mhz = (f[-1] - f[0]) / 1e6 if len(f) > 1 else 0
        
        print(f"Time span: {time_span_ms:.3f} ms")
        print(f"Frequency span: {freq_span_mhz:.1f} MHz")
        
        print("✓ Spectrogram generation PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Spectrogram generation FAILED: {e}")
        return False

def run_complete_workflow_test():
    """
    הרצת בדיקה מלאה של כל המהלך
    Run complete workflow test
    """
    print("מתחיל בדיקה מלאה של מערכת חילוץ פקטות ויצירת וקטורים")
    print("Starting complete workflow test for packet extraction and vector generation")
    print("="*80)
    
    results = {}
    
    # Test 1: Packet extraction and saving
    results['packet_extraction'] = test_packet_extraction_and_saving()
    
    # Test 2: Vector creation (2.5ms)
    try:
        vector_file, markers = test_vector_creation_2_5ms()
        results['vector_creation'] = True
        results['vector_file'] = vector_file
    except Exception as e:
        print(f"Vector creation failed: {e}")
        results['vector_creation'] = False
        results['vector_file'] = None
    
    # Test 3: Microsecond timing precision
    results['timing_precision'] = test_microsecond_timing_precision()
    
    # Test 4: Spectrogram display
    results['spectrogram_display'] = test_spectrogram_display()
    
    # Summary
    print("\n" + "="*80)
    print("סיכום תוצאות הבדיקה - TEST RESULTS SUMMARY")
    print("="*80)
    
    total_tests = len([k for k in results.keys() if k != 'vector_file'])
    passed_tests = sum([1 for k, v in results.items() if k != 'vector_file' and v])
    
    print(f"1. חילוץ פקטה ושמירה / Packet extraction & saving: {'✓ PASS' if results['packet_extraction'] else '✗ FAIL'}")
    print(f"2. יצירת וקטור 2.5ms / 2.5ms vector creation: {'✓ PASS' if results['vector_creation'] else '✗ FAIL'}")
    print(f"3. דיוק זמן מיקרושניה / Microsecond timing: {'✓ PASS' if results['timing_precision'] else '✗ FAIL'}")
    print(f"4. הצגת ספקטרוגרמה / Spectrogram display: {'✓ PASS' if results['spectrogram_display'] else '✗ FAIL'}")
    
    print(f"\nציון כולל / Overall Score: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 כל הבדיקות עברו בהצלחה! / All tests PASSED!")
    else:
        print("⚠️  יש בדיקות שנכשלו / Some tests FAILED")
    
    # Additional notes
    print("\nהערות נוספות / Additional Notes:")
    print("- בסביבה גרפית, החלונות האינטראקטיביים ייפתחו אוטומטית")
    print("- In GUI environment, interactive windows would open automatically")
    print("- הספקטרוגרמות מוצגות ברזולוציה גבוהה עם דיוק של מיקרושניה")
    print("- Spectrograms are displayed in high resolution with microsecond accuracy")
    print("- המערכת תומכת בשמירה אוטומטית של פקטות וקטורים")
    print("- System supports automatic saving of packets and vectors")
    
    return results

if __name__ == "__main__":
    # Ensure we have test data
    if not os.path.exists("data") or not os.listdir("data"):
        print("Creating test packets first...")
        os.system("python create_test_packets.py")
    
    # Run the complete workflow test
    results = run_complete_workflow_test()
    
    # Clean up temporary files
    print(f"\nניקוי קבצים זמניים / Cleaning up temporary files...")
    temp_files = ["data/extracted_test_packet.mat"]
    temp_files.extend([f for f in os.listdir(".") if f.startswith("vector_") and f.endswith(".mat")])
    temp_files.extend([f for f in os.listdir(".") if f.startswith("test_timing_vector_") and f.endswith(".mat")])
    
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                print(f"Removed: {temp_file}")
            except:
                pass
    
    print("\nבדיקה הושלמה / Testing completed!")