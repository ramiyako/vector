#!/usr/bin/env python3
"""
Test script to demonstrate the improved packet transplant functionality
with power normalization and updated validation criteria.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import (
    generate_sample_packet,
    extract_reference_segment,
    find_packet_location_in_vector,
    transplant_packet_in_vector,
    validate_transplant_quality,
    save_vector
)

def create_test_scenario():
    """Create a test scenario with realistic signal characteristics"""
    sample_rate = 56e6
    
    # Create a longer vector with background noise
    vector_length = 50000
    vector = 0.1 * (np.random.randn(vector_length) + 1j * np.random.randn(vector_length))
    
    # Create a clean packet with higher power
    packet_length = 5000
    t = np.arange(packet_length) / sample_rate
    clean_packet = 2.0 * np.exp(1j * 2 * np.pi * 5e6 * t)  # 5 MHz tone, higher amplitude
    
    # Create a low-power packet to simulate the original problem
    low_power_packet = 0.1 * clean_packet  # 20 dB lower power
    
    # Embed the clean packet in the vector at a known location
    embed_location = 15000
    vector[embed_location:embed_location + packet_length] = clean_packet
    
    return {
        'vector': vector,
        'clean_packet': clean_packet,
        'low_power_packet': low_power_packet,
        'embed_location': embed_location,
        'sample_rate': sample_rate
    }

def test_without_power_normalization():
    """Test transplant without power normalization (original behavior)"""
    print("=== Testing WITHOUT Power Normalization ===")
    
    scenario = create_test_scenario()
    vector = scenario['vector']
    clean_packet = scenario['clean_packet']
    low_power_packet = scenario['low_power_packet']
    embed_location = scenario['embed_location']
    sample_rate = scenario['sample_rate']
    
    # Extract reference segment
    reference_segment = extract_reference_segment(clean_packet, 0, 500)
    
    # Find packet location
    vector_location, packet_location, confidence = find_packet_location_in_vector(
        vector, clean_packet, reference_segment
    )
    
    print(f"Found packet at location: {vector_location} (expected: {embed_location})")
    print(f"Correlation confidence: {confidence:.3f}")
    
    # Perform transplant WITHOUT power normalization using low-power packet
    transplanted_vector = transplant_packet_in_vector(
        vector, 
        low_power_packet, 
        vector_location, 
        packet_location,
        normalize_power=False  # Disable power normalization
    )
    
    # Validate results
    validation_results = validate_transplant_quality(
        vector, transplanted_vector, low_power_packet, vector_location, 
        reference_segment, sample_rate
    )
    
    print(f"Validation Results (without normalization):")
    print(f"  Reference correlation: {validation_results['reference_correlation']:.3f}")
    print(f"  Reference confidence: {validation_results['reference_confidence']:.3f}")
    print(f"  Power ratio: {validation_results['power_ratio']:.6f}")
    print(f"  SNR improvement: {validation_results['snr_improvement_db']:.1f} dB")
    print(f"  Success: {validation_results['success']}")
    
    criteria = validation_results['criteria']
    print(f"  Criteria breakdown:")
    print(f"    Confidence OK: {criteria['confidence_ok']} ({validation_results['reference_confidence']:.3f} > {criteria['confidence_threshold']:.1f})")
    print(f"    Power OK: {criteria['power_ok']} ({validation_results['power_ratio']:.6f} > {criteria['power_ratio_threshold']:.3f})")
    print(f"    SNR OK: {criteria['snr_ok']} ({validation_results['snr_improvement_db']:.1f} > {criteria['min_snr_threshold']:.0f})")
    
    return validation_results

def test_with_power_normalization():
    """Test transplant with power normalization (improved behavior)"""
    print("\n=== Testing WITH Power Normalization ===")
    
    scenario = create_test_scenario()
    vector = scenario['vector']
    clean_packet = scenario['clean_packet']
    low_power_packet = scenario['low_power_packet']
    embed_location = scenario['embed_location']
    sample_rate = scenario['sample_rate']
    
    # Extract reference segment
    reference_segment = extract_reference_segment(clean_packet, 0, 500)
    
    # Find packet location
    vector_location, packet_location, confidence = find_packet_location_in_vector(
        vector, clean_packet, reference_segment
    )
    
    print(f"Found packet at location: {vector_location} (expected: {embed_location})")
    print(f"Correlation confidence: {confidence:.3f}")
    
    # Perform transplant WITH power normalization using low-power packet
    transplanted_vector = transplant_packet_in_vector(
        vector, 
        low_power_packet, 
        vector_location, 
        packet_location,
        normalize_power=True  # Enable power normalization
    )
    
    # Validate results
    validation_results = validate_transplant_quality(
        vector, transplanted_vector, low_power_packet, vector_location, 
        reference_segment, sample_rate
    )
    
    print(f"Validation Results (with normalization):")
    print(f"  Reference correlation: {validation_results['reference_correlation']:.3f}")
    print(f"  Reference confidence: {validation_results['reference_confidence']:.3f}")
    print(f"  Power ratio: {validation_results['power_ratio']:.6f}")
    print(f"  SNR improvement: {validation_results['snr_improvement_db']:.1f} dB")
    print(f"  Success: {validation_results['success']}")
    
    criteria = validation_results['criteria']
    print(f"  Criteria breakdown:")
    print(f"    Confidence OK: {criteria['confidence_ok']} ({validation_results['reference_confidence']:.3f} > {criteria['confidence_threshold']:.1f})")
    print(f"    Power OK: {criteria['power_ok']} ({validation_results['power_ratio']:.6f} > {criteria['power_ratio_threshold']:.3f})")
    print(f"    SNR OK: {criteria['snr_ok']} ({validation_results['snr_improvement_db']:.1f} > {criteria['min_snr_threshold']:.0f})")
    
    return validation_results

def demonstrate_improvements():
    """Demonstrate the improvements made to the transplant system"""
    print("Packet Transplant Improvements Demonstration")
    print("=" * 50)
    
    # Test both scenarios
    results_without = test_without_power_normalization()
    results_with = test_with_power_normalization()
    
    print("\n=== IMPROVEMENT SUMMARY ===")
    print(f"Power ratio improvement: {results_without['power_ratio']:.6f} → {results_with['power_ratio']:.6f}")
    print(f"SNR improvement: {results_without['snr_improvement_db']:.1f} → {results_with['snr_improvement_db']:.1f} dB")
    print(f"Validation success: {results_without['success']} → {results_with['success']}")
    
    # Calculate improvements
    power_improvement = results_with['power_ratio'] / results_without['power_ratio'] if results_without['power_ratio'] > 0 else float('inf')
    snr_improvement = results_with['snr_improvement_db'] - results_without['snr_improvement_db']
    
    print(f"\nQuantitative improvements:")
    print(f"  Power ratio improved by: {power_improvement:.1f}x")
    print(f"  SNR improved by: {snr_improvement:.1f} dB")
    print(f"  Validation criteria updated to be more realistic")
    
    # Show the updated thresholds
    criteria = results_with['criteria']
    print(f"\nUpdated validation thresholds:")
    print(f"  Confidence threshold: {criteria['confidence_threshold']:.1f} (was 0.5)")
    print(f"  Power ratio threshold: {criteria['power_ratio_threshold']:.3f} (was 0.1)")
    print(f"  Minimum SNR threshold: {criteria['min_snr_threshold']:.0f} dB (was unlimited)")

def create_sample_files():
    """Create sample files to demonstrate the functionality"""
    print("\n=== Creating Sample Files ===")
    
    scenario = create_test_scenario()
    
    # Save the test vector
    save_vector(scenario['vector'], 'sample_vector.mat')
    print("Created: sample_vector.mat")
    
    # Save the clean packet
    save_vector(scenario['clean_packet'], 'sample_clean_packet.mat')
    print("Created: sample_clean_packet.mat")
    
    # Save the low-power packet
    save_vector(scenario['low_power_packet'], 'sample_low_power_packet.mat')
    print("Created: sample_low_power_packet.mat")
    
    print("\nYou can now use these files in the GUI to test the improved transplant functionality!")

if __name__ == "__main__":
    # Run the demonstration
    demonstrate_improvements()
    
    # Create sample files for manual testing
    create_sample_files()
    
    print("\n" + "=" * 50)
    print("SUMMARY OF IMPROVEMENTS:")
    print("1. ✅ Added power normalization to transplant function")
    print("2. ✅ Updated validation criteria to be more realistic")
    print("3. ✅ Added detailed criteria breakdown in validation")
    print("4. ✅ Improved GUI feedback with power normalization messages")
    print("5. ✅ Updated all test cases to use new functionality")
    print("\nThe transplant system should now pass validation with proper power matching!")