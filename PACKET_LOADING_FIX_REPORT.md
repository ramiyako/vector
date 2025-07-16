# Packet Loading Fix Report

## Issue Description

The application was failing to start due to an error loading packet data from `data/final_test_vector.mat`:

```
Error loading packet from data/final_test_vector.mat: Ambiguous packet data in data/final_test_vector.mat. Available keys: ['__header__', '__version__', '__globals__', 'vector', 'sample_rate', 'duration_ms']
```

## Root Cause

The `load_packet()` and `load_packet_info()` functions in `utils.py` were designed to handle MAT files with either:
1. A 'Y' key containing the packet data
2. A single non-metadata key containing the packet data

However, the `final_test_vector.mat` file contains multiple keys:
- `vector`: The actual signal data (shape: 1 × 5,600,000)  
- `sample_rate`: Sample rate information ([[56000000.]])
- `duration_ms`: Duration in milliseconds

This caused the functions to throw an "ambiguous packet data" error.

## Solution Implemented

### 1. Enhanced Packet Data Detection

Updated both `load_packet()` and `load_packet_info()` functions to handle vector files by:

1. **Added explicit 'vector' key support**: Check for 'vector' key specifically after 'Y'
2. **Improved metadata filtering**: Exclude known metadata keys ('sample_rate', 'duration_ms', 'fs', 'sr') from data candidates
3. **Intelligent key preference**: When multiple data keys exist, try common names in order: 'packet', 'signal', 'data', 'waveform'
4. **Better error messages**: More descriptive error messages distinguishing between "ambiguous" and "not found" scenarios

### 2. Enhanced Sample Rate Extraction

Updated `get_sample_rate_from_mat()` function to:

1. **Handle nested arrays**: Properly extract scalar values from nested array structures like [[56000000.]]
2. **Robust data type handling**: Check if sample rate is array-like and extract the scalar value appropriately
3. **Consistent loading**: Use same loading parameters (`squeeze_me=True, struct_as_record=False`) as packet functions

### 3. Code Changes

**File: `utils.py`**

#### load_packet() function:
```python
if 'Y' in data:
    packet = data['Y']
elif 'vector' in data:
    # Handle vector files (like final_test_vector.mat)
    packet = data['vector']
else:
    # Find the first non-metadata key
    candidates = [k for k in data.keys() if not k.startswith('__') and k not in ['sample_rate', 'duration_ms', 'fs', 'sr']]
    if len(candidates) == 1:
        packet = data[candidates[0]]
    elif len(candidates) > 1:
        # Try common packet data keys in order of preference
        for key in ['packet', 'signal', 'data', 'waveform']:
            if key in candidates:
                packet = data[key]
                break
        else:
            raise ValueError(f"Ambiguous packet data in {file_path}. Available data keys: {candidates}")
    else:
        raise ValueError(f"No packet data found in {file_path}. Available keys: {list(data.keys())}")
```

#### get_sample_rate_from_mat() function:
```python
data = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
if 'sample_rate' in data:
    sample_rate = data['sample_rate']
    # Handle both scalar and array values
    if hasattr(sample_rate, '__len__') and len(sample_rate) == 1:
        return float(sample_rate[0])
    else:
        return float(sample_rate)
```

## Test Results

The fix was validated with the problematic `final_test_vector.mat` file:

```
✅ Sample rate: 56.0 MHz
✅ Packet loaded successfully
   - Shape: (5600000,)
   - Data type: complex64
   - Length: 5,600,000 samples
   - Duration: 0.100 seconds
   - Memory usage: 42.7 MB
```

## Compatibility

- **Backward compatible**: All existing MAT file formats continue to work
- **Forward compatible**: Supports new vector file formats
- **Robust**: Better handling of various MAT file structures
- **Error handling**: More informative error messages for debugging

## Supported File Formats

The enhanced loading functions now support:

1. **Standard packet files**: `{'Y': packet_data}`
2. **Vector files**: `{'vector': signal_data, 'sample_rate': rate, ...}`
3. **Single data files**: Any file with one non-metadata data key
4. **Multiple data files**: Files with common data key names (packet, signal, data, waveform)
5. **Various sample rate formats**: Scalar values, arrays, nested arrays

## Benefits

1. **Application startup**: GUI now starts successfully without packet loading errors
2. **File format flexibility**: Supports diverse MAT file structures
3. **Better diagnostics**: More informative error messages for troubleshooting
4. **Memory efficiency**: Maintains existing optimizations for large files
5. **Future-proof**: Extensible framework for new file formats