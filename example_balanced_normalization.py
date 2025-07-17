#!/usr/bin/env python3
"""
Example usage of balanced vector normalization for preserving weak frequency components.

This example demonstrates how the new balanced normalization ensures that weak frequency
components remain visible in the final vector, addressing the original issue where
weak components would disappear after standard normalization.
"""

import numpy as np
from utils import balanced_vector_normalization, generate_sample_packet

def create_example_vector():
    """Create an example vector with strong and weak frequency components"""
    print("Creating example vector with mixed frequency components...")
    
    sample_rate = 65.536e6  # 65.536 MHz
    duration = 0.001  # 1 ms
    
    # Strong frequency component (WiFi-like signal)
    strong_packet = generate_sample_packet(duration, sample_rate, 5e6, 1.0)
    
    # Weak frequency component (sensor signal)  
    weak_packet = generate_sample_packet(duration, sample_rate, 15e6, 0.05)
    
    # Combine components
    combined_vector = strong_packet + weak_packet
    
    print(f"Strong component: 5 MHz, amplitude: 1.0")
    print(f"Weak component: 15 MHz, amplitude: 0.05 (20x weaker)")
    print(f"Combined vector length: {len(combined_vector)} samples")
    
    return combined_vector

def demonstrate_normalization_comparison():
    """Demonstrate the difference between standard and balanced normalization"""
    print("\n" + "="*60)
    print("🔬 NORMALIZATION COMPARISON DEMONSTRATION")
    print("="*60)
    
    # Create test vector
    vector = create_example_vector()
    
    # Standard normalization (old approach)
    print(f"\n📊 STANDARD NORMALIZATION:")
    max_amplitude = np.max(np.abs(vector))
    standard_normalized = vector / max_amplitude
    
    print(f"Max amplitude before: {max_amplitude:.3f}")
    print(f"Max amplitude after: {np.max(np.abs(standard_normalized)):.3f}")
    
    # Analyze frequency content
    fft_std = np.fft.fft(standard_normalized)
    fft_magnitude_std = np.abs(fft_std)
    
    # Find the two main peaks (strong and weak components)
    peaks_std = np.sort(fft_magnitude_std)[-2:]  # Two largest peaks
    weak_strong_ratio_std = peaks_std[0] / peaks_std[1]
    
    print(f"Weak/Strong component ratio: {weak_strong_ratio_std:.3f} ({weak_strong_ratio_std*100:.1f}%)")
    
    # Balanced normalization (new approach)
    print(f"\n🎯 BALANCED NORMALIZATION:")
    balanced_normalized, scale_factor = balanced_vector_normalization(vector)
    
    print(f"Max amplitude after: {np.max(np.abs(balanced_normalized)):.3f}")
    print(f"Scale factor applied: {scale_factor:.3f}")
    
    # Analyze frequency content
    fft_bal = np.fft.fft(balanced_normalized)
    fft_magnitude_bal = np.abs(fft_bal)
    
    peaks_bal = np.sort(fft_magnitude_bal)[-2:]  # Two largest peaks
    weak_strong_ratio_bal = peaks_bal[0] / peaks_bal[1]
    
    print(f"Weak/Strong component ratio: {weak_strong_ratio_bal:.3f} ({weak_strong_ratio_bal*100:.1f}%)")
    
    # Calculate improvement
    improvement = weak_strong_ratio_bal / weak_strong_ratio_std
    print(f"\n✨ IMPROVEMENT: {improvement:.2f}x better weak component visibility!")
    
    return improvement

def demonstrate_usage_in_gui():
    """Show how balanced normalization integrates with the existing GUI"""
    print(f"\n" + "="*60)
    print("🖥️  GUI INTEGRATION EXAMPLE")
    print("="*60)
    
    print("""
In the vector generation GUI, balanced normalization is now automatically applied:

1. User creates vector with multiple frequency components
2. Some components may be much weaker than others
3. When 'Normalize Final Vector' is checked:
   
   OLD BEHAVIOR:
   - Standard normalization: vector = vector / max(abs(vector))
   - Weak components become nearly invisible
   
   NEW BEHAVIOR:
   - Balanced normalization automatically detects amplitude imbalance
   - Frequency-domain selective boosting preserves weak components
   - All components remain visible in spectrograms
   
4. Manual X2 boost still available for additional control
5. Enhanced packet transplant also uses improved power normalization
""")

def main():
    """Run the demonstration"""
    print("🚀 BALANCED VECTOR NORMALIZATION DEMONSTRATION")
    print("=" * 60)
    print("Addressing the issue where weak frequency components disappear after normalization")
    
    # Demonstrate the improvement
    improvement = demonstrate_normalization_comparison()
    
    # Show GUI integration
    demonstrate_usage_in_gui()
    
    # Summary
    print(f"\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    
    if improvement > 1.5:
        print(f"✅ SUCCESS: {improvement:.1f}x improvement in weak component visibility")
        print("✅ Weak frequency components now remain visible after normalization")
        print("✅ Existing X2 boost mechanism preserved for manual control")
        print("✅ Enhanced packet transplant prevents over-attenuation")
    else:
        print("❌ Improvement less than expected - may need parameter adjustment")
    
    print(f"\n🎯 The solution ensures that weak frequency components (פקטות חלשות)")
    print(f"   maintain appropriate amplitude levels in the normalized vector!")

if __name__ == "__main__":
    main()