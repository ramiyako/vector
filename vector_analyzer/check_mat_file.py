from scipy.io import loadmat

def check_mat_structure(file_path):
    data = loadmat(file_path)
    print("מפתחות בקובץ MATLAB:")
    for key in data.keys():
        if not key.startswith('__'):
            print(f"מפתח: {key}")
            print(f"צורה: {data[key].shape}")
            print(f"סוג: {type(data[key])}")
            print("---")

if __name__ == '__main__':
    check_mat_structure('vectors/OS40-5600-40-air3-5230.mat') 