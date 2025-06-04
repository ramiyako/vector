import scipy.io as sio
import numpy as np

def check_mat_structure(file_path):
    try:
        data = sio.loadmat(file_path)
        print("מפתחות בקובץ:", data.keys())
        for key in data.keys():
            if not key.startswith('__'):
                print(f"\nמפתח: {key}")
                print(f"סוג: {type(data[key])}")
                if isinstance(data[key], np.ndarray):
                    print(f"צורה: {data[key].shape}")
                    print(f"סוג נתונים: {data[key].dtype}")
    except Exception as e:
        print(f"שגיאה בטעינת הקובץ: {e}")

if __name__ == "__main__":
    check_mat_structure("vectors/channel_5240MHz.mat") 
