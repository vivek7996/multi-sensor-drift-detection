import joblib, os, numpy as np
def save_scaler(scaler, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)

def load_scaler(path):
    return joblib.load(path)

def save_numpy(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)
