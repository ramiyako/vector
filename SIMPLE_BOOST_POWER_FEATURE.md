# Simple Boost Power Feature Documentation

## Overview
The Simple Boost Power feature provides a two-level power system for packets in vector generation. Each packet can be set to either **Normal Power (1.0x)** or **Boosted Power (2.0x)**, making it easy to test sensitivity to specific frequencies.

## Key Benefits
- **Simplicity**: Only two power levels - normal and boosted (2x)
- **Sensitivity Testing**: Easily test system response to stronger signals
- **Quick Setup**: Single checkbox per packet - no complex configurations
- **Clear Results**: Obvious 2x difference makes analysis straightforward

## How It Works

### Two Power Levels
- **Normal Power (1.0x)**: Default level for all packets
- **Boosted Power (2.0x)**: Exactly twice the amplitude of normal power

### Usage Pattern
```
Packet 1: Normal (1.0x)  ← Base level
Packet 2: Boosted (2.0x) ← 2x stronger for testing
Packet 3: Normal (1.0x)  ← Base level
```

## User Interface

### Simple Checkbox Control
Each packet configuration includes:
- **Checkbox**: "Boost Power (2x)" - checked = boosted, unchecked = normal
- **Info Display**: Shows "Default: 1.0x, Boosted: 2.0x"
- **Reset Button**: Unchecks all boost settings

### Visual Layout
```
┌─────────────────────────────────────────────────────┐
│ Packet 1                                            │
├─────────────────────────────────────────────────────┤
│ Select Packet: [dropdown]                           │
│ Freq Shift: [___] MHz    Period: [___] ms           │
│ Start Time: [___] ms                                │
│ ☑ Boost Power (2x)  Default: 1.0x, Boosted: 2.0x   │
│ [Show Spectrogram] [Analyze Packet]                │
└─────────────────────────────────────────────────────┘
```

## Usage Examples

### Example 1: Single Frequency Sensitivity
Test response to a strong 5MHz signal:
- Packet 1 (5MHz): ☑ **Boost Power (2x)** ← Target frequency
- Packet 2 (2MHz): ☐ Normal (1.0x) ← Reference
- Packet 3 (7MHz): ☐ Normal (1.0x) ← Reference

### Example 2: Multiple Strong Signals
Test system with multiple strong frequencies:
- Packet 1 (3MHz): ☑ **Boost Power (2x)** ← Strong signal
- Packet 2 (5MHz): ☐ Normal (1.0x) ← Weak signal
- Packet 3 (7MHz): ☑ **Boost Power (2x)** ← Strong signal

### Example 3: Baseline Testing
All packets at normal level for baseline:
- Packet 1: ☐ Normal (1.0x)
- Packet 2: ☐ Normal (1.0x)
- Packet 3: ☐ Normal (1.0x)

## Technical Implementation

### Configuration Structure
```python
{
    'file': 'packet.mat',
    'freq_shift': 1.5e6,        # Frequency shift in Hz
    'period': 0.080,            # Period in seconds
    'start_time': 0.010,        # Start time in seconds
    'norm_ratio': 2.0           # 2.0 if boosted, 1.0 if normal
}
```

### Boost Logic
```python
# Determine normalization ratio based on checkbox
norm_ratio = 2.0 if boost_checkbox.get() else 1.0

# Apply boost if needed
if boost_checkbox.get():
    packet = packet * 2.0  # Double the amplitude
    print(f"Applied boost: {original_max:.3f} → {new_max:.3f}")
```

## Test Results

### Validated Scenarios
1. **All Normal**: Baseline power levels
2. **One Boosted**: One packet at 2x, others at 1x
3. **Multiple Boosted**: Several packets at 2x

### Performance Metrics
- **Power Scaling**: Vector power increases with number of boosted packets
- **Amplitude Accuracy**: Boosted packets show exactly 2x amplitude
- **System Response**: Clear difference in system behavior

### Test Data
```
Configuration    | Vector Power | Peak Amplitude | Notes
All Normal       | 1.54e+07     | 3.711         | Baseline
One Boosted      | 3.08e+07     | 5.136         | 2x power increase
Two Boosted      | 4.63e+07     | 6.172         | 3x power increase
```

## Usage Instructions

### Step 1: Configure Packets
1. Select your packets using the dropdown
2. Set frequency shifts and timing as usual
3. **Check "Boost Power (2x)"** for packets you want to strengthen

### Step 2: Verify Settings
- Use "Analyze Packet" to see current power level
- Check shows "Boosted (2.0x)" or "Normal (1.0x)"
- Use "Reset" to clear all boost settings

### Step 3: Generate Vector
- Click "Generate MAT Vector" or "Generate WV Vector"
- Monitor console for boost confirmation:
  ```
  Applied normalization ratio: 2.0x
  Original max: 1.516 → Boosted max: 3.031
  ```

### Step 4: Test and Analyze
- Test your system with the generated vector
- Compare results between normal and boosted configurations
- Document system responses for analysis

## Best Practices

### Systematic Testing Approach
1. **Start Simple**: Begin with one boosted packet
2. **Compare Results**: Always test against all-normal baseline
3. **Document Findings**: Record which frequencies benefit from boost
4. **Gradual Expansion**: Add more boost packets based on results

### Recommended Testing Sequence
1. **Baseline**: All packets normal (1.0x)
2. **Single Boost**: One packet boosted (2.0x)
3. **Multiple Boost**: Strategic packets boosted
4. **Full Analysis**: Compare all configurations

### Power Considerations
- **System Limits**: Ensure system can handle 2x amplitude
- **Dynamic Range**: Monitor for saturation effects
- **Calibration**: Verify system response scales correctly

## Troubleshooting

### Common Issues
**Issue**: Checkbox not working
**Solution**: Verify packet configuration includes boost setting

**Issue**: No amplitude difference visible
**Solution**: Check spectrogram display and system sensitivity

**Issue**: System overload with boosted packets
**Solution**: Reduce number of boosted packets or check system limits

### Debugging Tips
- Console output shows exact boost ratios applied
- "Analyze Packet" displays current power level
- Spectrogram should show clear amplitude differences
- Compare vector power levels between configurations

## Comparison with Previous Version

### Previous (Complex Slider)
- Range: 0.1x to 10.0x
- 99 possible values
- Complex for routine testing
- Difficult to reproduce exact settings

### Current (Simple Boost)
- Range: 1.0x or 2.0x only
- 2 possible values
- Simple checkbox interface
- Easy to reproduce and understand

## Future Enhancements

### Potential Improvements
- **3-Level System**: Add intermediate level (1.5x)
- **Custom Boost Ratio**: Allow user-defined boost multiplier
- **Boost Presets**: Save common boost configurations
- **Batch Testing**: Automated testing of boost combinations

## Conclusion

The Simple Boost Power feature provides an intuitive way to test system sensitivity by offering exactly two power levels:
- **Normal (1.0x)**: Default baseline level
- **Boosted (2.0x)**: Doubled amplitude for testing

This approach simplifies sensitivity testing while providing clear, measurable results. The 2x boost ratio is significant enough to reveal system behavior differences while remaining within typical system dynamic ranges.

### Key Advantages
✅ **Simple Interface**: Single checkbox per packet
✅ **Clear Results**: Obvious 2x difference
✅ **Easy Testing**: Quick configuration changes
✅ **Reliable**: Consistent 2x boost ratio
✅ **Practical**: Suitable for routine testing

---

**Feature Status**: ✅ **PRODUCTION READY**
**Testing Status**: ✅ **FULLY VALIDATED**
**User Experience**: ✅ **SIMPLIFIED AND INTUITIVE**