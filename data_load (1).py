import pandas as pd
import numpy as np
from pathlib import Path

def read_cmapss(path):
    colnames = ["unit","cycle"] + [f"op{i}" for i in range(1,4)] + [f"s{i}" for i in range(1,22)]
    df = pd.read_csv(path, sep=r'\s+', header=None, names=colnames)
    return df

def concat_train_files(file_paths):
    dfs = [read_cmapss(p) for p in file_paths]
    df = pd.concat(dfs, ignore_index=True)
    df['global_t'] = np.arange(len(df)).astype(float)
    return df

def quick_stats(df, cols=None, nrows=5):
    if cols is None:
        cols = df.columns.tolist()
    print("shape:", df.shape)
    print("head:")
    print(df.head(nrows))
    print("describe on numeric columns:")
    print(df.describe().T)
