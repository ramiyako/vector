# Dynamic Normalization Feature Documentation

## Overview
The Dynamic Normalization feature allows you to control the normalization ratio for each individual packet in a vector. This enables sensitivity testing by boosting or reducing specific frequencies relative to others in the vector.

## Key Benefits
- **Sensitivity Testing**: Test how your system responds to different amplitude levels for specific frequencies
- **Fine Control**: Precisely adjust the relative power of each packet in the vector
- **Flexible Testing**: Experiment with different amplitude ratios without regenerating packets
- **Real-time Adjustment**: Dynamically modify normalization ratios during vector creation

## How It Works

### Traditional Normalization
Previously, all packets in a vector were normalized to the same level:
```
All packets → Same amplitude → Combined in vector → Global normalization
```

### Dynamic Normalization
Now, each packet can have its own normalization ratio:
```
Packet 1 → 0.5x normalization (reduced amplitude)
Packet 2 → 2.0x normalization (boosted amplitude)  
Packet 3 → 1.0x normalization (unchanged)
→ Combined in vector → Optional global normalization
```

## User Interface

### Normalization Ratio Control
Each packet configuration now includes:
- **Slider**: Range from 0.1x to 10.0x (0.1 to 1000% of original amplitude)
- **Display**: Shows current ratio (e.g., "2.5x")
- **Reset Button**: Quickly return to 1.0x (no modification)

### Visual Layout
```
┌─────────────────────────────────────────────────────┐
│ Packet 1                                            │
├─────────────────────────────────────────────────────┤
│ Select Packet: [dropdown]                           │
│ Freq Shift: [___] MHz    Period: [___] ms           │
│ Start Time: [___] ms                                │
│ Normalization Ratio: [slider] 2.5x [Reset]         │
│ [Show Spectrogram] [Analyze Packet]                │
└─────────────────────────────────────────────────────┘
```

## Usage Examples

### Example 1: Sensitivity Testing
Test how your system responds to a weak signal at 5MHz:
- Packet 1 (5MHz): Set normalization ratio to 0.1x (very weak)
- Packet 2 (2MHz): Set normalization ratio to 1.0x (reference)
- Generate vector and observe system response

### Example 2: Interference Testing
Simulate strong interference at one frequency:
- Packet 1 (7MHz): Set normalization ratio to 5.0x (strong interference)
- Packet 2 (3MHz): Set normalization ratio to 1.0x (normal signal)
- Test how system handles strong interference

### Example 3: Comparative Analysis
Compare different signal strengths:
- Create multiple vectors with different ratio combinations
- Analyze system performance across different scenarios

## Technical Implementation

### Configuration Structure
Each packet configuration now includes:
```python
{
    'file': 'packet.mat',
    'freq_shift': 1.5e6,        # Frequency shift in Hz
    'period': 0.080,            # Period in seconds
    'start_time': 0.010,        # Start time in seconds
    'norm_ratio': 2.5           # NEW: Normalization ratio
}
```

### Vector Generation Process
1. Load packet data
2. Apply frequency shift (if specified)
3. **Apply individual normalization ratio** (NEW STEP)
4. Insert packet instances at specified intervals
5. Combine all packets in final vector
6. Apply global normalization (if enabled)

### Code Changes
The key implementation is in the vector generation logic:
```python
# Apply individual normalization ratio
if cfg['norm_ratio'] != 1.0:
    y = y * cfg['norm_ratio']
    print(f"  Applied normalization ratio: {cfg['norm_ratio']:.1f}x")
```

## Testing and Validation

### Comprehensive Testing Performed
1. **Basic Functionality**: Verify ratios from 0.1x to 10.0x work correctly
2. **Vector Generation**: Test with multiple packets and different ratios
3. **Power Analysis**: Confirm amplitude changes match expected ratios
4. **Spectrogram Verification**: Visual confirmation of amplitude differences
5. **Edge Cases**: Test extreme values and reset functionality

### Test Results
- ✅ All normalization ratios apply correctly
- ✅ Vector generation works with mixed ratios
- ✅ Power levels scale as expected
- ✅ Spectrograms show clear amplitude differences
- ✅ No performance impact on vector generation

## Usage Instructions

### Step 1: Configure Packets
1. Select your desired packets using the dropdown
2. Set frequency shifts and timing as usual
3. **Adjust normalization ratio** for each packet using the slider

### Step 2: Analyze Settings
- Use "Analyze Packet" to see current normalization ratio
- Check packet details to verify settings
- Reset ratios to 1.0x if needed

### Step 3: Generate Vector
- Click "Generate MAT Vector" or "Generate WV Vector"
- Monitor console output for normalization confirmation
- Review final spectrogram to verify results

### Step 4: Sensitivity Testing
- Create multiple vectors with different ratio combinations
- Test your system's response to each vector
- Analyze performance differences

## Best Practices

### Recommended Ratio Ranges
- **Weak Signal Testing**: 0.1x - 0.5x
- **Normal Operation**: 0.8x - 1.2x
- **Strong Signal Testing**: 2.0x - 5.0x
- **Extreme Cases**: 0.1x or 10.0x

### Systematic Testing Approach
1. Start with all ratios at 1.0x (baseline)
2. Modify one packet at a time
3. Document system responses
4. Test combinations gradually

### Performance Considerations
- Extreme ratios (0.1x or 10.0x) may affect system dynamic range
- Consider overall vector power when using high ratios
- Monitor for clipping or saturation effects

## Troubleshooting

### Common Issues
**Issue**: Ratios not applying correctly
**Solution**: Verify packet configuration includes 'norm_ratio' field

**Issue**: Vector too loud/quiet
**Solution**: Adjust individual ratios or enable global normalization

**Issue**: System not responding to weak signals
**Solution**: Try higher ratios (2.0x - 5.0x) for weak signal testing

### Debugging Tips
- Check console output for normalization confirmation
- Use "Analyze Packet" to verify current settings
- Compare spectrograms before and after ratio changes
- Test with simple packets first, then complex ones

## Future Enhancements

### Potential Improvements
- **Preset Ratios**: Save and load common ratio combinations
- **Relative Normalization**: Set ratios relative to other packets
- **Automated Testing**: Script different ratio combinations
- **Visual Feedback**: Real-time amplitude preview

### Integration Possibilities
- Export ratio settings with vectors
- Import ratio configurations from files
- Batch processing with different ratios
- Integration with measurement systems

## Conclusion

The Dynamic Normalization feature provides powerful control over packet amplitudes in vectors, enabling sophisticated sensitivity testing and system analysis. By allowing individual control of each packet's normalization ratio, you can:

- Test system sensitivity to specific frequencies
- Simulate various interference scenarios
- Conduct comparative analysis across different conditions
- Maintain precise control over vector composition

This feature enhances the application's testing capabilities while maintaining backward compatibility with existing workflows.

---

**Feature Status**: ✅ **READY FOR PRODUCTION USE**
**Testing Status**: ✅ **COMPREHENSIVE TESTING COMPLETED**
**Documentation Status**: ✅ **COMPLETE DOCUMENTATION PROVIDED**