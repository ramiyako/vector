import numpy as np
import os
import scipy.io as sio
from datetime import datetime
from pathlib import Path

def mat2wv(path_signal, sFilename, fSampleRate, bNormalize=True, var_name=None):
    """
    Python version of mat2wv for Colab.

    Parameters:
    - path_signal: str (path to .mat file) or numpy array of complex samples
    - sFilename:   Output filename for the SMU waveform file
    - fSampleRate: Sample rate in Hz
    - bNormalize:  Whether to normalize the signal (True/False)
    - var_name:    Name of the variable inside the .mat file (required if path_signal is a file)
    """
    # Load signal from .mat or use provided array
    if isinstance(path_signal, str):
        mat = sio.loadmat(path_signal)
        if var_name is None:
            raise ValueError("When path_signal is a filename, var_name must be specified")
        signal = np.asarray(mat[var_name]).squeeze()
    else:
        signal = np.asarray(path_signal).flatten()

    N = signal.size

    # Normalize and compute power metrics
    if bNormalize:
        print("Normalize signal")
        signal = signal / np.max(np.abs(signal))
        fPeakPower = np.max(np.abs(signal)**2)
        fPeakPowerdBfs = -10 * np.log10(fPeakPower)
        fMeanPower = np.mean(np.abs(signal)**2)
        fRMSdBfs = -10 * np.log10(fMeanPower)
    else:
        fPeakPowerdBfs = 0.0
        fRMSdBfs = 0.0

    # Quantize to 16-bit integer range
    iMaxInt = 32767
    vicData = signal * iMaxInt
    real = np.real(vicData).astype(np.int16)
    imag = np.imag(vicData).astype(np.int16)

    # Interleave real and imaginary parts
    interleaved = np.empty(2 * N, dtype=np.int16)
    interleaved[0::2] = real
    interleaved[1::2] = imag

    # Compute block length (MATLAB uses 4*N+1)
    total_bytes = 4 * N + 1

    # Write SMU waveform file
    with open(sFilename, 'wb') as fid:
        fid.write(f"{{TYPE: SMU-WV,0}}".encode())
        fid.write(f"{{COMMENT: Generated by mat2wv.py}}".encode())
        fid.write(f"{{DATE: {datetime.now().strftime('%Y-%m-%d;%H:%M:%S')}}}".encode())
        fid.write(f"{{LEVEL OFFS: {fRMSdBfs}, {fPeakPowerdBfs}}}".encode())
        fid.write(f"{{CLOCK: {fSampleRate}}}".encode())
        fid.write(f"{{SAMPLES: {N}}}".encode())
        fid.write(f"{{WAVEFORM-{total_bytes}:#".encode())
        fid.write(interleaved.tobytes())
        fid.write(b"}")

def main():
    # תיקיית המקור של קבצי ה-.mat
    input_dir = "vectors"
    
    # תיקיית היעד לקבצי ה-.wv
    output_dir = "vectors_wv"
    
    # יצירת תיקיית היעד אם היא לא קיימת
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # המרת כל קבצי ה-.mat בתיקייה
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.mat'):
            mat_file_path = os.path.join(input_dir, file_name)
            wv_file_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.wv")
            
            # הגדרת פרמטרים
            sample_rate = 40e6  # 40 MHz
            var_name = 'Y'  # שם המשתנה בקובץ ה-.mat
            
            # המרת הקובץ
            mat2wv(mat_file_path, wv_file_path, sample_rate, bNormalize=True, var_name=var_name)
            print(f"הקובץ {mat_file_path} הומר בהצלחה ל-{wv_file_path}")

if __name__ == "__main__":
    main() 
