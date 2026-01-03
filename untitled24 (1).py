
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance
from scipy.spatial import procrustes as scipy_procrustes
from sklearn.metrics.pairwise import cosine_distances
import joblib

# -----------------------
# Sliding windows utility
# -----------------------
def sliding_windows(X, window_size=50, step=25, aggregate="mean"):
    """
    X: (T, D) time series
    returns: (n_windows, D) where each window aggregated by 'aggregate' (mean/std/median)
    """
    T, D = X.shape
    windows = []
    for start in range(0, max(0, T - window_size + 1), step):
        w = X[start:start + window_size]
        if aggregate == "mean":
            windows.append(np.mean(w, axis=0))
        elif aggregate == "std":
            windows.append(np.std(w, axis=0))
        elif aggregate == "median":
            windows.append(np.median(w, axis=0))
        else:
            windows.append(np.mean(w, axis=0))
    if len(windows) == 0:
        return np.zeros((0, D))
    return np.vstack(windows)

# -----------------------
# Dense Autoencoder (PyTorch)
# -----------------------
class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64,32], latent_dim=8):
        super().__init__()
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(prev, h))
            enc_layers.append(nn.ReLU())
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(prev, h))
            dec_layers.append(nn.ReLU())
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        xrec = self.decoder(z)
        return xrec

# -----------------------
# Train per-modality AEs
# -----------------------
def train_per_modality_autoencoders(modality_data, latent_dim=8, hidden_dims=[64,32],
                                    batch_size=128, epochs=5, scaler_save_path=None):
    """
    modality_data: dict modality -> numpy array (n_samples, n_features)
    Returns:
      ae_models: dict modality -> PyTorch model (trained)
      encoders: dict modality -> model (same model)
      scaler: fit StandardScaler on concatenated features (returned)
    """
    # Fit global scaler on concatenated features
    all_X = np.vstack(list(modality_data.values()))
    scaler = StandardScaler().fit(all_X)
    if scaler_save_path:
        joblib.dump(scaler, scaler_save_path)

    ae_models = {}
    encoders = {}
    import torch.utils.data as D
    for m, X in modality_data.items():
        Xs = scaler.transform(X)
        input_dim = Xs.shape[1]
        model = DenseAutoencoder(input_dim=input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim)
        model.train()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        dataset = D.TensorDataset(torch.from_numpy(Xs.astype(np.float32)))
        loader = D.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for ep in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                rec = model(batch)
                loss = criterion(rec, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            # optional: print progress
            # print(f"[AE:{m}] ep {ep+1}/{epochs}, loss {epoch_loss:.6f}")

        model.cpu()
        ae_models[m] = model
        encoders[m] = model  # encoder uses model.encoder in inference

    return ae_models, encoders, scaler

# -----------------------
# Procrustes alignment
# -----------------------
def procrustes_align(X_ref, X_to_align):
    """
    Uses scipy.spatial.procrustes to align X_to_align to X_ref.
    Input shapes: (n_samples, latent_dim)
    Returns: aligned version of X_to_align
    """
    # SciPy's procrustes returns (m1, m2, disparity)
    try:
        mtx1, mtx2, disparity = scipy_procrustes(X_ref, X_to_align)
        return mtx2
    except Exception as e:
        # fallback: identity (no alignment)
        print("[procrustes_align] warning, procrustes failed:", e)
        return X_to_align

# -----------------------
# Build Z_aligned per window
# -----------------------
def build_Z_aligned_per_window(latent_aligned):
    """
    latent_aligned: dict modality -> Z (n_windows, latent_dim_mod)
    Align modalities to the first modality and concatenate per-window latents.
    Returns Z_concat: (n_windows, sum(latent_dims))
    """
    modalities = list(latent_aligned.keys())
    assert len(modalities) >= 1
    # ensure same n_windows
    n_windows = min([latent_aligned[m].shape[0] for m in modalities])
    if any(latent_aligned[m].shape[0] != n_windows for m in modalities):
        # truncate to min length
        for m in modalities:
            latent_aligned[m] = latent_aligned[m][:n_windows]

    ref = latent_aligned[modalities[0]]
    aligned = {modalities[0]: ref}
    for m in modalities[1:]:
        try:
            aligned_m = procrustes_align(ref, latent_aligned[m])
        except Exception as e:
            print("[build_Z_aligned_per_window] procrustes failed for", m, e)
            aligned_m = latent_aligned[m]
        aligned[m] = aligned_m

    # Concatenate latents per window
    Zs = []
    for i in range(n_windows):
        parts = [aligned[m][i] for m in modalities]
        Zs.append(np.concatenate(parts))
    return np.vstack(Zs)

# -----------------------
# Mahalanobis distance
# -----------------------
def mahalanobis_distance(a, b, cov_est):
    """
    a, b: 1D arrays
    cov_est: either a covariance matrix (2D np.array) or sklearn EmpiricalCovariance instance (fitted)
    returns scalar distance
    """
    a = np.atleast_1d(a)
    b = np.atleast_1d(b)
    diff = a - b
    if hasattr(cov_est, "precision_"):
        inv = cov_est.precision_
    elif isinstance(cov_est, np.ndarray) and cov_est.ndim == 2:
        inv = np.linalg.inv(cov_est)
    else:
        raise ValueError("Unsupported cov_est type for Mahalanobis")
    val = float(np.sqrt(np.dot(diff.T, np.dot(inv, diff))))
    return val

# -----------------------
# Transition model (simple MLP)
# -----------------------
class TransitionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64,32]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, input_dim))  # predict next Z (same dim)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_transition_model(Z_sequence, hidden_dims=[64,32], epochs=20, batch_size=128):
    """
    Z_sequence: (n_windows, latent_dim_total)
    Trains model to predict Z_{t+1} from Z_t (one-step).
    Returns trained model (stateful PyTorch nn.Module).
    """
    if Z_sequence.shape[0] < 2:
        raise ValueError("Need at least 2 windows to train transition model")
    X = Z_sequence[:-1]
    Y = Z_sequence[1:]
    input_dim = X.shape[1]
    model = TransitionMLP(input_dim, hidden_dims=hidden_dims)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    import torch.utils.data as D
    dataset = D.TensorDataset(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(Y.astype(np.float32)))
    loader = D.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for ep in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(dataset)
        # optional progress print
        # print(f"[TRANS] ep {ep+1}/{epochs} loss {epoch_loss:.6f}")
    model.cpu()
    return model

# -----------------------
# Example quick smoke demo (lightweight)
# -----------------------
def example_run_quick(quick=True):
    """
    Builds small synthetic dataset, trains AEs and transition model quickly,
    returns simple metrics.
    """
    # synthetic sinusoidal + noise, 3 sensors
    T = 200 if not quick else 120
    ts = np.linspace(0, 10, T)
    s1 = np.sin(ts) + 0.01 * np.random.randn(T)
    s2 = np.cos(ts * 0.8) + 0.01 * np.random.randn(T)
    s3 = np.sin(ts * 1.2 + 0.5) + 0.01 * np.random.randn(T)
    X = np.vstack([s1, s2, s3]).T  # (T, 3)

    windows = sliding_windows(X, window_size=20, step=10, aggregate="mean")
    modality_data = {"sensors": windows}
    ae_models, encoders, scaler = train_per_modality_autoencoders(
        modality_data, latent_dim=4, hidden_dims=[32,16], batch_size=64, epochs=3, scaler_save_path=None
    )
    latent_aligned = {}
    import torch
    for m in modality_data:
        model = encoders[m]
        model.eval()
        with torch.no_grad():
            Z = model.encoder(torch.from_numpy(modality_data[m].astype(np.float32))).numpy()
        latent_aligned[m] = Z
    Z_concat = build_Z_aligned_per_window(latent_aligned)
    trans = train_transition_model(Z_concat, hidden_dims=[32,16], epochs=5, batch_size=16)
    # compute simple euclidean diffs between consecutive windows as pseudo-drift
    eu = np.linalg.norm(np.diff(Z_concat, axis=0), axis=1)
    return {"euclid": eu, "Z_vectors": Z_concat}

# -----------------------
# Utility distances
# -----------------------
def euclidean_distance(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))

def cosine_distance(a, b):
    return float(cosine_distances([a], [b])[0,0])
