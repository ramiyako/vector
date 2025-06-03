import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

for i in range(1, 7):
    try:
        data = sio.loadmat(f'data/packet_{i}.mat')
        y = data['Y'].flatten()
        print(f"פקטה {i} - ערכים ראשונים:", y[:10])
        print(f"פקטה {i} - סכום מוחלטים:", np.abs(y).sum())
        f, t, Sxx = signal.spectrogram(y, 44100)
        plt.figure(figsize=(8, 4))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12))
        plt.title(f'ספקטוגרמה פקטה {i}')
        plt.ylabel('תדירות [Hz]')
        plt.xlabel('זמן [שניות]')
        plt.colorbar(label='עוצמה [dB]')
        plt.show()
    except Exception as e:
        print(f"בעיה בפקטה {i}: {e}") 