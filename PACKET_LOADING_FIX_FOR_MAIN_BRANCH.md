# Fix for Main Branch - Packet Loading Issue

## 🎯 Problem
The main branch fails to load `data/final_test_vector.mat` with error:
```
Error loading packet from data/final_test_vector.mat: Ambiguous packet data in data/final_test_vector.mat. Available keys: ['__header__', '__version__', '__globals__', 'vector', 'sample_rate', 'duration_ms']
```

## 🔧 Solution
The `load_packet()` function in `utils.py` needs to be updated to handle files with `'vector'` key instead of `'Y'` key.

## 📝 Code Changes Needed

### File: `utils.py`

**Find this section (around line 70-85):**
```python
if 'Y' in data:
    packet = data['Y']
else:
    # Find the first non-metadata key
    candidates = [k for k in data.keys() if not k.startswith('__')]
    if len(candidates) == 1:
        packet = data[candidates[0]]
    else:
        raise ValueError(f"Ambiguous packet data in {file_path}. Available keys: {list(data.keys())}")
```

**Replace with:**
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

### Also update `load_packet_info()` function (around line 112):

**Find:**
```python
def load_packet_info(file_path):
    """Load packet data and pre-buffer info from MAT file."""
    data = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    if 'Y' in data:
        packet = data['Y']
    else:
        candidates = [k for k in data.keys() if not k.startswith('__')]
        if len(candidates) == 1:
            packet = data[candidates[0]]
        else:
            raise ValueError(f"Ambiguous packet data in {file_path}. Available keys: {list(data.keys())}")
```

**Replace with:**
```python
def load_packet_info(file_path):
    """Load packet data and pre-buffer info from MAT file."""
    data = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    if 'Y' in data:
        packet = data['Y']
    elif 'vector' in data:
        # Handle vector files (like final_test_vector.mat)
        packet = data['vector']
    else:
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

### Also update `get_sample_rate_from_mat()` function (around line 30):

**Find the data loading section:**
```python
data = sio.loadmat(file_path)
if 'sample_rate' in data:
    return float(data['sample_rate'])
```

**Replace with:**
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

## ✅ Expected Result
After applying these changes, the main branch should show:
```
Starting Unified Vector Generator...
All dependencies are available
Successfully loaded packet: data/final_test_vector.mat
Basic functionality test passed
Packet list refreshed: found 17 packets
```

## 🧪 Test the Fix
Run this command to verify the fix works:
```python
python -c "from utils import load_packet; packet = load_packet('data/final_test_vector.mat'); print('Success! Loaded packet with shape:', packet.shape)"
```

Should output: `Success! Loaded packet with shape: (5600000,)`