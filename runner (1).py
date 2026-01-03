import os, glob, json, sys, joblib, numpy as np, pandas as pd
from importlib import import_module, util as importlib_util

# ---- Configurable paths (override with env vars) ----
DATA_DIR = os.environ.get("DATA_DIR", "/content/drive/MyDrive/ml 2 projects/data")
MODEL_DIR = os.environ.get("MODEL_DIR", "/content/drive/MyDrive/ml 2 projects/models")
SCALER_DIR = os.environ.get("SCALER_DIR", "/content/drive/MyDrive/ml 2 projects/scalers")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

# ---- Attempt robust import of untitled24 ----
def load_untitled24():
    try:
        mod = import_module("untitled24")
        print("[runner] imported untitled24 from:", getattr(mod, "__file__", "unknown"))
        return mod
    except Exception as e:
        # try probable locations
        candidates = [
            os.path.join(os.getcwd(), "untitled24.py"),
            os.path.join(DATA_DIR, "untitled24.py"),
            "/mnt/data/untitled24.py",
        ]
        for path in candidates:
            if path and os.path.exists(path):
                try:
                    spec = importlib_util.spec_from_file_location("untitled24", path)
                    mod = importlib_util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    print(f"[runner] dynamically loaded untitled24 from: {path}")
                    return mod
                except Exception as ex:
                    print(f"[runner] failed loading untitled24 from {path}: {ex}")
        raise ModuleNotFoundError(
            "Could not import or locate untitled24.py. Searched locations:\n  - normal import\n  - "
            + "\n  - ".join(candidates)
        )

untitled24 = load_untitled24()

# expose functions used below
sliding_windows = untitled24.sliding_windows
train_per_modality_autoencoders = untitled24.train_per_modality_autoencoders
build_Z_aligned_per_window = untitled24.build_Z_aligned_per_window
train_transition_model = untitled24.train_transition_model
DenseAutoencoder = untitled24.DenseAutoencoder

# ---- Auto-discover train files ----
pattern = os.path.join(DATA_DIR, "train_FD*.txt")
found = sorted(glob.glob(pattern))
print("[runner] DATA_DIR =", DATA_DIR)
print("[runner] Found train files:", found)
if len(found) == 0:
    print("[runner] ERROR: No train_FD*.txt files found in DATA_DIR:", DATA_DIR)
    sys.exit(1)
TRAIN_FILES = found

# ---- Pipeline hyperparameters (edit as needed) ----
CONFIG = {
    "window_size": 50,
    "step": 25,
    "latent_dim": 8,
    "ae_hidden": [64, 32],
    "transition_hidden": [64, 32],
    "batch_size": 128,
    "ae_epochs": 5,
    "trans_epochs": 20,
}

# ---- Helpers ----
def concat_train_files(files):
    dfs = []
    for f in files:
        print(f"[runner] reading {f}")
        df = pd.read_csv(f, delim_whitespace=True, header=None)
        dfs.append(df)
    all_df = pd.concat(dfs, axis=0, ignore_index=True)
    return all_df

def main():
    print("[runner] Loading train files...")
    df_all = concat_train_files(TRAIN_FILES)
    print("[runner] Loaded dataframe shape:", df_all.shape)

    X = df_all.values.astype(np.float32)
    print("[runner] Using X shape:", X.shape)
    w = CONFIG["window_size"]
    step = CONFIG["step"]
    print(f"[runner] Building sliding windows (win={w}, step={step})")
    windows = sliding_windows(X, window_size=w, step=step, aggregate="mean")
    print("[runner] windows shape:", windows.shape)
    if windows.shape[0] == 0:
        print("[runner] ERROR: No windows produced. Lower window_size or use longer series.")
        sys.exit(1)


        # ---------- robust sliding_windows call (tries many signatures) ----------



    print("[runner] windows shape:", windows.shape)
    if windows.shape[0] == 0:
        print("[runner] ERROR: No windows produced. Lower window_size or use longer series.")
        sys.exit(1)

    modality_data = {"sensors": windows}
    modality_dims = {"sensors": windows.shape[1]}

    print("[runner] Training per-modality autoencoders...")
    ae_models, encoders, scaler = train_per_modality_autoencoders(
        modality_data,
        latent_dim=CONFIG["latent_dim"],
        hidden_dims=CONFIG["ae_hidden"],
        batch_size=CONFIG["batch_size"],
        epochs=CONFIG["ae_epochs"],
        scaler_save_path=os.path.join(SCALER_DIR, "scaler_sensors.pkl")
    )

    print("[runner] Encoding windows to latent space...")
    latent_aligned = {}
    import torch
    for m in modality_data.keys():
        encoder = encoders[m]
        Xs = modality_data[m]
        encoder.eval()
        with torch.no_grad():
            Xt = torch.from_numpy(Xs.astype(np.float32))
            Z = encoder.encoder(Xt).numpy()
        latent_aligned[m] = Z
        print(f"[runner] {m} latent shape: {Z.shape}")

    print("[runner] Building aligned Z snapshots...")
    Z_concat = build_Z_aligned_per_window(latent_aligned)
    print("[runner] Z_concat shape:", Z_concat.shape)

    z_path = os.path.join(MODEL_DIR, "Z_concat.npy")
    np.save(z_path, Z_concat)
    print("[runner] Saved Z_concat ->", z_path)

    print("[runner] Training transition model...")
    trans_model = train_transition_model(
        Z_concat,
        hidden_dims=CONFIG["transition_hidden"],
        epochs=CONFIG["trans_epochs"],
        batch_size=CONFIG["batch_size"]
    )

    # Save artifacts
    import torch
    ae_state_path = os.path.join(MODEL_DIR, "ae_sensors.pth")
    torch.save(ae_models["sensors"].state_dict(), ae_state_path)
    trans_state_path = os.path.join(MODEL_DIR, "trans_model.pth")
    torch.save(trans_model.state_dict(), trans_state_path)
    print("[runner] Saved AE and transition model states")

    cfg_save = {
        "window_size": CONFIG["window_size"],
        "step": CONFIG["step"],
        "latent_dim": CONFIG["latent_dim"],
        "modalities": list(modality_dims.keys()),
        "ae_hidden": CONFIG["ae_hidden"],
        "transition_hidden": CONFIG["transition_hidden"],
        "ae_state_path": os.path.basename(ae_state_path),
        "trans_state_path": os.path.basename(trans_state_path),
        "z_path": os.path.basename(z_path),
        "scaler_path": os.path.basename(os.path.join(SCALER_DIR, "scaler_sensors.pkl"))
    }
    with open(os.path.join(MODEL_DIR, "config.json"), "w") as fh:
        json.dump(cfg_save, fh, indent=2)
    print("[runner] Saved config.json")

    try:
        joblib.dump(scaler, os.path.join(SCALER_DIR, "scaler_sensors.pkl"))
        print("[runner] Saved scaler ->", os.path.join(SCALER_DIR, "scaler_sensors.pkl"))
    except Exception as e:
        print("[runner] Warning: could not save scaler:", e)

    print("[runner] DONE.")

if __name__ == "__main__":
    main()
