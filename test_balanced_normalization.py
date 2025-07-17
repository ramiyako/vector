#!/usr/bin/env python3
"""
Test script for balanced vector normalization functionality.
Validates that weak frequency components maintain appropriate amplitude levels.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import (
    balanced_vector_normalization, 
    create_spectrogram, 
    plot_spectrogram,
    generate_sample_packet
)

def create_test_vector_with_weak_components():
    """Create a test vector with strong and weak frequency components"""
    sample_rate = 65.536e6  # 65.536 MHz
    duration = 0.001  # 1 ms
    
    # Create time vector
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Strong frequency component (high amplitude)
    strong_freq = 5e6  # 5 MHz
    strong_amplitude = 1.0
    strong_component = strong_amplitude * np.exp(2j * np.pi * strong_freq * t)
    
    # Weak frequency component (very low amplitude)
    weak_freq = 15e6  # 15 MHz  
    weak_amplitude = 0.05  # 20x weaker than strong component
    weak_component = weak_amplitude * np.exp(2j * np.pi * weak_freq * t)
    
    # Combine components
    test_vector = strong_component + weak_component
    
    # Add some noise
    noise_level = 0.01
    noise = noise_level * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
    test_vector += noise
    
    return test_vector, sample_rate, {
        'strong_freq': strong_freq,
        'weak_freq': weak_freq,
        'strong_amp': strong_amplitude,
        'weak_amp': weak_amplitude,
        'amplitude_ratio': weak_amplitude / strong_amplitude
    }

def test_standard_vs_balanced_normalization():
    """Compare standard and balanced normalization approaches"""
    print("=== Testing Standard vs Balanced Normalization ===")
    
    # Create test vector with weak components
    vector, sample_rate, params = create_test_vector_with_weak_components()
    
    print(f"Original vector parameters:")
    print(f"  Strong component: {params['strong_freq']/1e6:.1f} MHz, amplitude: {params['strong_amp']:.3f}")
    print(f"  Weak component: {params['weak_freq']/1e6:.1f} MHz, amplitude: {params['weak_amp']:.3f}")
    print(f"  Amplitude ratio: {params['amplitude_ratio']:.3f} ({params['amplitude_ratio']*100:.1f}%)")
    
    # Standard normalization (old method)
    max_val = np.max(np.abs(vector))
    standard_normalized = vector / max_val if max_val > 0 else vector
    
    # Balanced normalization (new method)
    balanced_normalized, scale_factor = balanced_vector_normalization(vector)
    
    # Analyze results
    print(f"\nNormalization Results:")
    print(f"  Standard normalization - Max amplitude: {np.max(np.abs(standard_normalized)):.3f}")
    print(f"  Balanced normalization - Max amplitude: {np.max(np.abs(balanced_normalized)):.3f}")
    print(f"  Balanced scale factor: {scale_factor:.3f}")
    
    # Calculate amplitude ratios for weak vs strong components
    def analyze_frequency_content(signal, title):
        """Analyze frequency content to identify component amplitudes"""
        fft_signal = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
        
        # Find peaks around expected frequencies
        strong_idx = np.argmin(np.abs(freqs - params['strong_freq']))
        weak_idx = np.argmin(np.abs(freqs - params['weak_freq']))
        
        strong_amp = np.abs(fft_signal[strong_idx]) / len(signal)
        weak_amp = np.abs(fft_signal[weak_idx]) / len(signal)
        ratio = weak_amp / strong_amp if strong_amp > 0 else 0
        
        print(f"  {title}:")
        print(f"    Strong component amplitude: {strong_amp:.6f}")
        print(f"    Weak component amplitude: {weak_amp:.6f}")
        print(f"    Weak/Strong ratio: {ratio:.4f} ({ratio*100:.2f}%)")
        
        return strong_amp, weak_amp, ratio
    
    print(f"\nFrequency Analysis:")
    orig_strong, orig_weak, orig_ratio = analyze_frequency_content(vector, "Original")
    std_strong, std_weak, std_ratio = analyze_frequency_content(standard_normalized, "Standard Normalized")
    bal_strong, bal_weak, bal_ratio = analyze_frequency_content(balanced_normalized, "Balanced Normalized")
    
    # Create comparison spectrograms
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original
    f1, t1, Sxx1 = create_spectrogram(vector, sample_rate, time_resolution_us=1)
    im1 = axes[0, 0].pcolormesh(t1 * 1000, f1 / 1e6, 10 * np.log10(np.abs(Sxx1) + 1e-12), 
                                shading='nearest', cmap='turbo')
    axes[0, 0].set_title('Original Vector')
    axes[0, 0].set_xlabel('Time [ms]')
    axes[0, 0].set_ylabel('Frequency [MHz]')
    
    # Standard normalized
    f2, t2, Sxx2 = create_spectrogram(standard_normalized, sample_rate, time_resolution_us=1)
    im2 = axes[0, 1].pcolormesh(t2 * 1000, f2 / 1e6, 10 * np.log10(np.abs(Sxx2) + 1e-12), 
                                shading='nearest', cmap='turbo')
    axes[0, 1].set_title('Standard Normalization')
    axes[0, 1].set_xlabel('Time [ms]')
    axes[0, 1].set_ylabel('Frequency [MHz]')
    
    # Balanced normalized
    f3, t3, Sxx3 = create_spectrogram(balanced_normalized, sample_rate, time_resolution_us=1)
    im3 = axes[0, 2].pcolormesh(t3 * 1000, f3 / 1e6, 10 * np.log10(np.abs(Sxx3) + 1e-12), 
                                shading='nearest', cmap='turbo')
    axes[0, 2].set_title('Balanced Normalization')
    axes[0, 2].set_xlabel('Time [ms]')
    axes[0, 2].set_ylabel('Frequency [MHz]')
    
    # Add colorbars
    for i, im in enumerate([im1, im2, im3]):
        plt.colorbar(im, ax=axes[0, i], label='Power [dB]')
    
    # Time domain comparison
    t_plot = np.linspace(0, len(vector)/sample_rate * 1000, len(vector))
    
    axes[1, 0].plot(t_plot, np.abs(vector), 'b-', alpha=0.7, label='Magnitude')
    axes[1, 0].set_title('Original - Time Domain')
    axes[1, 0].set_xlabel('Time [ms]')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(t_plot, np.abs(standard_normalized), 'g-', alpha=0.7, label='Magnitude')
    axes[1, 1].set_title('Standard Normalized - Time Domain')
    axes[1, 1].set_xlabel('Time [ms]')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].plot(t_plot, np.abs(balanced_normalized), 'r-', alpha=0.7, label='Magnitude')
    axes[1, 2].set_title('Balanced Normalized - Time Domain')
    axes[1, 2].set_xlabel('Time [ms]')
    axes[1, 2].set_ylabel('Amplitude')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('balanced_normalization_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Validation
    print(f"\n=== Validation Results ===")
    improvement_factor = bal_ratio / std_ratio if std_ratio > 0 else float('inf')
    print(f"Weak component visibility improvement: {improvement_factor:.2f}x")
    
    # Check if balanced normalization preserved better visibility
    success = bal_ratio > std_ratio * 1.1  # At least 10% improvement
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"Balanced normalization test: {status}")
    
    if success:
        print("✅ Balanced normalization successfully preserved weak component visibility")
    else:
        print("❌ Balanced normalization did not significantly improve weak component visibility")
    
    return success

def test_packet_transplant_normalization():
    """Test enhanced packet transplant normalization"""
    print("\n=== Testing Enhanced Packet Transplant Normalization ===")
    
    from utils import transplant_packet_in_vector
    
    # Create base vector
    sample_rate = 65.536e6
    vector_duration = 0.002  # 2 ms
    vector = generate_sample_packet(vector_duration, sample_rate, 5e6, 0.8)
    
    # Create weak packet to transplant
    packet_duration = 0.0005  # 0.5 ms
    weak_packet = generate_sample_packet(packet_duration, sample_rate, 15e6, 0.1)  # Much weaker
    
    print(f"Vector amplitude: {np.max(np.abs(vector)):.3f}")
    print(f"Weak packet amplitude: {np.max(np.abs(weak_packet)):.3f}")
    print(f"Amplitude ratio: {np.max(np.abs(weak_packet))/np.max(np.abs(vector)):.3f}")
    
    # Transplant location (middle of vector)
    transplant_location = len(vector) // 2
    
    # Test enhanced transplant
    transplanted_vector = transplant_packet_in_vector(
        vector, weak_packet, transplant_location, normalize_power=True
    )
    
    # Create comparison spectrograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original vector
    f1, t1, Sxx1 = create_spectrogram(vector, sample_rate, time_resolution_us=1)
    im1 = ax1.pcolormesh(t1 * 1000, f1 / 1e6, 10 * np.log10(np.abs(Sxx1) + 1e-12), 
                         shading='nearest', cmap='turbo')
    ax1.set_title('Original Vector')
    ax1.set_xlabel('Time [ms]')
    ax1.set_ylabel('Frequency [MHz]')
    plt.colorbar(im1, ax=ax1, label='Power [dB]')
    
    # Transplanted vector
    f2, t2, Sxx2 = create_spectrogram(transplanted_vector, sample_rate, time_resolution_us=1)
    im2 = ax2.pcolormesh(t2 * 1000, f2 / 1e6, 10 * np.log10(np.abs(Sxx2) + 1e-12), 
                         shading='nearest', cmap='turbo')
    ax2.set_title('After Enhanced Transplant')
    ax2.set_xlabel('Time [ms]')
    ax2.set_ylabel('Frequency [MHz]')
    plt.colorbar(im2, ax=ax2, label='Power [dB]')
    
    # Mark transplant location
    transplant_time_ms = transplant_location / sample_rate * 1000
    ax2.axvline(x=transplant_time_ms, color='white', linestyle='--', linewidth=2, alpha=0.8)
    ax2.text(transplant_time_ms + 0.1, 20, 'Transplant', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('enhanced_transplant_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Validate transplant visibility
    transplant_region = transplanted_vector[transplant_location:transplant_location + len(weak_packet)]
    transplant_amplitude = np.max(np.abs(transplant_region))
    vector_amplitude = np.max(np.abs(vector))
    
    visibility_ratio = transplant_amplitude / vector_amplitude
    print(f"\nTransplant Results:")
    print(f"  Transplanted region amplitude: {transplant_amplitude:.3f}")
    print(f"  Vector amplitude: {vector_amplitude:.3f}")
    print(f"  Visibility ratio: {visibility_ratio:.3f}")
    
    # Success if transplanted packet is at least 10% of vector amplitude
    success = visibility_ratio >= 0.1
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"Enhanced transplant visibility test: {status}")
    
    return success

def main():
    """Run all balanced normalization tests"""
    print("🧪 Balanced Vector Normalization Test Suite")
    print("=" * 50)
    
    test_results = []
    
    # Test 1: Standard vs Balanced normalization
    test_results.append(test_standard_vs_balanced_normalization())
    
    # Test 2: Enhanced packet transplant
    test_results.append(test_packet_transplant_normalization())
    
    # Summary
    print(f"\n{'='*50}")
    print(f"📊 Test Summary:")
    print(f"{'='*50}")
    
    passed = sum(test_results)
    total = len(test_results)
    
    test_names = [
        "Standard vs Balanced Normalization",
        "Enhanced Packet Transplant Normalization"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Balanced normalization is working correctly.")
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
    
    return passed == total

if __name__ == "__main__":
    main()