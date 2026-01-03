import sys
sys.path.insert(0, "/content/drive/MyDrive/ml 2 projects/src")

import os, sys, json, argparse, joblib, numpy as np, pandas as pd
import torch
from untitled24 import (sliding_windows, DenseAutoencoder, euclidean_distance,
                        cosine_distance, mahalanobis_distance, procrustes_align,
                        TransitionMLP)
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
import warnings

# default dirs (match runner.py)
MODEL_DIR = os.environ.get("MODEL_DIR", "/content/temp_models") # Changed to /content/temp_models
SCALER_DIR = os.environ.get("SCALER_DIR", "/content/drive/MyDrive/ml 2 projects/scalers")


def load_artifacts():
    cfg_path = os.path.join(MODEL_DIR, "config.json")
    if not os.path.exists(cfg_path):
        # Fallback to original path if not found in new temp_models
        cfg_path = os.path.join(os.environ.get("MODEL_DIR", "/content/drive/MyDrive/ml 2 projects/models"), "config.json")
        if not os.path.exists(cfg_path):
             raise FileNotFoundError("config.json not found in MODEL_DIR: " + MODEL_DIR)

    cfg = json.load(open(cfg_path, "r"))
    # load scaler
    scaler_path = os.path.join(SCALER_DIR, cfg.get("scaler_path", "scaler_sensors.pkl"))
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Scaler not found: " + scaler_path)
    scaler = joblib.load(scaler_path)
    # load Z baseline
    z_path = os.path.join(MODEL_DIR, cfg.get("z_path", "Z_concat.npy"))
    if not os.path.exists(z_path):
        # Fallback to original path if not found in new temp_models
        z_path = os.path.join(os.environ.get("MODEL_DIR", "/content/drive/MyDrive/ml 2 projects/models"), cfg.get("z_path", "Z_concat.npy"))
        if not os.path.exists(z_path):
            raise FileNotFoundError("Z_concat.npy not found: " + z_path)
    Z_baseline = np.load(z_path)
    # load AE model
    ae_state_path = os.path.join(MODEL_DIR, cfg.get("ae_state_path", "ae_sensors.pth"))
    if not os.path.exists(ae_state_path):
        # Fallback to original path if not found in new temp_models
        ae_state_path = os.path.join(os.environ.get("MODEL_DIR", "/content/drive/MyDrive/ml 2 projects/models"), cfg.get("ae_state_path", "ae_sensors.pth"))
        if not os.path.exists(ae_state_path):
            raise FileNotFoundError("AE state not found: " + ae_state_path)

    cfg_hidden = cfg.get("ae_hidden", [64, 32])
    latent_dim = cfg.get("latent_dim", 8)
    return cfg, scaler, Z_baseline, ae_state_path, cfg_hidden, latent_dim


def build_test_windows_from_file(test_file, window_size=50, step=25, aggregate="mean"):
    df = pd.read_csv(test_file, delim_whitespace=True, header=None)
    X = df.values.astype(np.float32)
    windows = sliding_windows(X, window_size=window_size, step=step, aggregate=aggregate)
    return windows


def encode_windows_with_ae(X_windows, ae_state_path, hidden_dims, latent_dim, scaler):
    # scale windows
    Xs = scaler.transform(X_windows)
    input_dim = Xs.shape[1]
    ae_model = DenseAutoencoder(input_dim=input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim)
    state = torch.load(ae_state_path, map_location="cpu")
    # handle possible state dict structures (state vs {"model_state":...})
    if isinstance(state, dict) and set(state.keys()) & {"state_dict", "model_state", "ae_state"}:
        # try common keys
        if "state_dict" in state:
            sd = state["state_dict"]
        elif "model_state" in state:
            sd = state["model_state"]
        elif "ae_state" in state:
            sd = state["ae_state"]
        else:
            sd = state
        try:
            ae_model.load_state_dict(sd)
        except Exception:
            # try loading directly (state may already be a raw state_dict)
            ae_model.load_state_dict(state)
    else:
        ae_model.load_state_dict(state)
    ae_model.eval()
    with torch.no_grad():
        Z = ae_model.encoder(torch.from_numpy(Xs.astype(np.float32))).numpy()
    return Z


def compute_drift_metrics(Z_baseline, Z_test):
    """
    For each test window i, compare to corresponding baseline window (if lengths equal)
    or compare to last baseline snapshot otherwise.
    Returns dictionary of arrays (euclid, cosine, mahalanobis)
    """
    n_b = Z_baseline.shape[0]
    n_t = Z_test.shape[0]
    cov_est = EmpiricalCovariance().fit(Z_baseline)
    distances = {"euclid": [], "cosine": [], "mahal": []}
    for i in range(n_t):
        if i < n_b:
            ref = Z_baseline[i]
        else:
            ref = Z_baseline[-1]
        zt = Z_test[i]
        distances["euclid"].append(euclidean_distance(ref, zt))
        distances["cosine"].append(cosine_distance(ref, zt))
        distances["mahal"].append(mahalanobis_distance(ref, zt, cov_est))
    for k in distances:
        distances[k] = np.array(distances[k])
    return distances


def compute_pairwise_distances(Z_ref_seq, Z_test_seq):
    """
    Compute euclid/cosine/mahal distances between corresponding elements of two sequences:
    Z_ref_seq[i] <-> Z_test_seq[i].
    Arrays must have same length. Returns dict of numpy arrays.
    """
    if Z_ref_seq.shape[0] != Z_test_seq.shape[0]:
        raise ValueError("Reference and test expected sequences must have same length for pairwise distances.")
    cov_est = EmpiricalCovariance().fit(Z_ref_seq)
    n = Z_ref_seq.shape[0]
    eu = np.zeros(n)
    co = np.zeros(n)
    ma = np.zeros(n)
    for i in range(n):
        a = Z_ref_seq[i]
        b = Z_test_seq[i]
        eu[i] = euclidean_distance(a, b)
        co[i] = cosine_distance(a, b)
        ma[i] = mahalanobis_distance(a, b, cov_est)
    return {"euclid": eu, "cosine": co, "mahal": ma}

# robust_procrustes: works when n_rows differ (but dims must match)
def robust_procrustes_align(X_ref, X_to_align, allow_scale=False):
    """
    Align X_to_align -> X_ref using orthogonal Procrustes estimated by SVD of cross-covariance.
    Works when X_ref.shape[0] != X_to_align.shape[0] but X_ref.shape[1] == X_to_align.shape[1].
    Returns: X_to_align_aligned, dict(info)
    """
    X_ref = np.asarray(X_ref)
    X_to = np.asarray(X_to_align)
    if X_ref.ndim != 2 or X_to.ndim != 2:
        raise ValueError("Inputs must be 2D arrays")
    if X_ref.shape[1] != X_to.shape[1]:
        raise ValueError("Dimensionality mismatch: cols must be equal")

    # center
    mu_ref = X_ref.mean(axis=0)
    mu_to = X_to.mean(axis=0)
    A0 = X_ref - mu_ref    # (n_ref, d)
    B0 = X_to - mu_to      # (n_to, d)

    # optional isotropic scale
    if allow_scale:
        normA = np.linalg.norm(A0)
        normB = np.linalg.norm(B0)
        s = (normA / (normB + 1e-12))
        B0s = B0 * s
    else:
        s = 1.0
        B0s = B0

    # cross-covariance (d x d)
    C = A0.T @ B0s        # shape (d, d)
    try:
        U, _, Vt = np.linalg.svd(C)
        R = U @ Vt
    except np.linalg.LinAlgError as e:
        # fallback: identity
        return X_to, {"success": False, "reason": f"SVD failed: {e}"}

    # apply rotation and optional scale, then translate to ref mean
    B_aligned = (B0s @ R) + mu_ref
    info = {"R": R, "scale": s, "mu_ref": mu_ref, "mu_to": mu_to, "success": True}
    return B_aligned, info



def print_summary(distances, prefix=""):
    for k, arr in distances.items():
        print(f"{prefix}{k}: n={len(arr)} mean={arr.mean():.4f} std={arr.std():.4f} max={arr.max():.4f}")


def linear_interpolation_expected(Z_baseline_proj, Z_test_aligned, alpha=0.5):
    """
    Build expected latent sequence by linearly interpolating between baseline snapshots.
    Returns Z_expected (n_t, D)
    """
    n_b = Z_baseline_proj.shape[0]
    n_t = Z_test_aligned.shape[0]
    Z_expected = []
    for i in range(n_t):
        idx = min(i, n_b - 1)
        idx2 = min(idx + 1, n_b - 1)
        z1, z2 = Z_baseline_proj[idx], Z_baseline_proj[idx2]
        z_interp = (1.0 - alpha) * z1 + alpha * z2
        Z_expected.append(z_interp)
    return np.vstack(Z_expected)


def transition_model_expected(Z_baseline_proj, cfg, device="cpu"):
    """
    Load TransitionMLP (if available) and produce predicted next-state sequence for baseline.
    Returns Z_pred (n_b, D) where each row is predicted next Z from baseline row.
    """
    # trans_state_path = os.path.join(MODEL_DIR, cfg.get("trans_state_path", "trans_model.pth"))
    trans_state_path = "/content/drive/MyDrive/ml 2 projects/models/trans_model.pth"

    if not os.path.exists(trans_state_path):
        raise FileNotFoundError("Transition model state not found: " + trans_state_path)
    input_dim = Z_baseline_proj.shape[1]
    hidden = cfg.get("transition_hidden", [64, 32])
    model = TransitionMLP(input_dim=input_dim, hidden_dims=hidden)
    # safe load
    state = torch.load(trans_state_path, map_location="cpu")
    try:
        model.load_state_dict(state)
    except Exception:
        # try common wrapping keys
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)
    model.eval()
    model.to(device)
    with torch.no_grad():
        inp = torch.from_numpy(Z_baseline_proj.astype(np.float32))
        pred = model(inp).cpu().numpy()
    return pred  # predicted Z_{t+1} for each baseline row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", required=False, help="path to test file (whitespace delimited)")
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--interp_method", choices=["none", "linear", "trans", "both"], default="none",
                        help="dynamic interpolation method to use for expected path")
    parser.add_argument("--alpha", type=float, default=0.5, help="alpha for linear interpolation (0..1)")
    args = None

    if hasattr(sys, "ps1") or "google.colab" in sys.modules or "ipykernel" in sys.modules:
        default_test = os.environ.get("INFERENCE_TEST_FILE")
        if default_test is None:
            data_dir_guess = os.environ.get("DATA_DIR", "/content/drive/MyDrive/ml 2 projects/data")
            import glob
            files = sorted(glob.glob(os.path.join(data_dir_guess, "train_FD*.txt")))
            default_test = files[0] if files else None
        class _A: pass
        args = _A()
        args.test_file = default_test
        args.window_size = None
        args.step = None
        args.interp_method = "none"
        args.alpha = 0.5
        if args.test_file is None:
            raise FileNotFoundError("No default test file found; please set INFERENCE_TEST_FILE env var.")
    else:
        args = parser.parse_args()

    cfg, scaler, Z_baseline, ae_state_path, hidden_dims, latent_dim = load_artifacts()

    # Ensure MODEL_DIR exists for saving generated numpy files
    os.makedirs(MODEL_DIR, exist_ok=True)

    win = args.window_size or cfg.get("window_size", 50)
    step = args.step or cfg.get("step", 25)
    print("[inference] Using window_size", win, "step", step)
    X_windows = build_test_windows_from_file(args.test_file, window_size=win, step=step, aggregate="mean")
    if X_windows.shape[0] == 0:
        print("[inference] No windows generated from test file.")
        return
    print("[inference] Test windows:", X_windows.shape)

    Z_test = encode_windows_with_ae(X_windows, ae_state_path, hidden_dims, latent_dim, scaler)
    print("[inference] Encoded Z_test shape:", Z_test.shape)

    # Dimension match & Procrustes alignment
    if Z_test.shape[1] != Z_baseline.shape[1]:
        print("[inference] Dim mismatch. Applying PCA to baseline:",
              Z_baseline.shape[1], "\u2192", Z_test.shape[1])
        p = PCA(n_components=Z_test.shape[1])
        Z_baseline_proj = p.fit_transform(Z_baseline)
    else:
        Z_baseline_proj = Z_baseline

    # try:
    #     Z_test_aligned = procrustes_align(Z_baseline_proj, Z_test)
    #     print("[inference] Procrustes-aligned Z_test to Z_baseline manifold")
    # except Exception as e:
    #     print("[inference] Procrustes failed, using unaligned Z_test:", e)
    #     Z_test_aligned = Z_test
              # Try robust orthogonal alignment that tolerates different #rows
    try:
        Z_test_aligned, align_info = robust_procrustes_align(Z_baseline_proj, Z_test, allow_scale=True)
        if align_info.get("success", False):
            print("[inference] Robust Procrustes alignment succeeded (allow_scale=True).")
        else:
            print("[inference] Robust Procrustes alignment reported failure; using unaligned Z_test:", align_info.get("reason"))
            Z_test_aligned = Z_test
    except Exception as e:
        print("[inference] Robust Procrustes failed; falling back to unaligned Z_test:", e)
        Z_test_aligned = Z_test

        # Normalize latents using baseline stats (zero mean, unit var) — optional but stabilizes distances
    baseline_mean = Z_baseline_proj.mean(axis=0)
    baseline_std = Z_baseline_proj.std(axis=0)
    baseline_std[baseline_std == 0] = 1.0
    Z_baseline_norm = (Z_baseline_proj - baseline_mean) / baseline_std
    Z_test_aligned_norm = (Z_test_aligned - baseline_mean) / baseline_std

    # Use normalized matrices for drift calculations
    distances_baseline = compute_drift_metrics(Z_baseline_norm, Z_test_aligned_norm)

    # Baseline drift (aligned)
    # distances_baseline = compute_drift_metrics(Z_baseline_proj, Z_test_aligned)
    print_summary(distances_baseline, prefix="[aligned->baseline] ")

    # Dynamic interpolation / prediction checks
    method = args.interp_method
    if method in ("linear", "both"):
        alpha = float(args.alpha)
        Z_expected_lin = linear_interpolation_expected(Z_baseline_proj, Z_test_aligned, alpha=alpha)
        np.save(os.path.join(MODEL_DIR, "Z_expected_linear.npy"), Z_expected_lin) # Save to MODEL_DIR
        dr_lin = compute_pairwise_distances(Z_expected_lin, Z_test_aligned)
        print_summary(dr_lin, prefix="[interp-linear->test] ")
        np.save(os.path.join(MODEL_DIR, "interp_linear_drift_euclid.npy"), dr_lin["euclid"])
        np.save(os.path.join(MODEL_DIR, "interp_linear_drift_cosine.npy"), dr_lin["cosine"])
        np.save(os.path.join(MODEL_DIR, "interp_linear_drift_mahal.npy"), dr_lin["mahal"])

    if method in ("trans", "both"):
        # Load and predict with transition MLP
        try:
            Z_pred = transition_model_expected(Z_baseline_proj, cfg, device="cpu")
            # Z_pred is predicted next-state for each baseline row. We need an expected sequence aligned to test windows.
            # We'll shift Z_pred by one so that expected for test index i is Z_pred[i-1] (for i>0), and for i==0 use Z_baseline_proj[0]
            n_t = Z_test_aligned.shape[0]
            n_b = Z_pred.shape[0]
            Z_expected_trans = []
            for i in range(n_t):
                if i == 0:
                    zexp = Z_baseline_proj[0]
                else:
                    idx = min(i - 1, n_b - 1)
                    zexp = Z_pred[idx]
                Z_expected_trans.append(zexp)
            Z_expected_trans = np.vstack(Z_expected_trans)
            np.save(os.path.join(MODEL_DIR, "Z_expected_trans.npy"), Z_expected_trans) # Save to MODEL_DIR
            dr_trans = compute_pairwise_distances(Z_expected_trans, Z_test_aligned)
            print_summary(dr_trans, prefix="[trans-pred->test] ")
            np.save(os.path.join(MODEL_DIR, "interp_trans_drift_euclid.npy"), dr_trans["euclid"])
            np.save(os.path.join(MODEL_DIR, "interp_trans_drift_cosine.npy"), dr_trans["cosine"])
            np.save(os.path.join(MODEL_DIR, "interp_trans_drift_mahal.npy"), dr_trans["mahal"])
        except Exception as e:
            warnings.warn(f"[inference] Transition-model interpolation failed: {e}")
            print("[inference] Skipping transition interpolation.")

    # save aligned results and baseline drifts
    # np.save(os.path.join(MODEL_DIR, "Z_test_aligned.npy"), Z_test_aligned) # Save to MODEL_DIR
    np.save(os.path.join(MODEL_DIR, "Z_test_aligned_raw.npy"), Z_test_aligned)
    np.save(os.path.join(MODEL_DIR, "Z_test_aligned.npy"), Z_test_aligned_norm)
    np.save(os.path.join(MODEL_DIR, "aligned_baseline.npy"), Z_baseline_proj) # Save to MODEL_DIR
    np.save(os.path.join(MODEL_DIR, "aligned_baseline_drift_euclid.npy"), distances_baseline["euclid"])
    np.save(os.path.join(MODEL_DIR, "aligned_baseline_drift_cosine.npy"), distances_baseline["cosine"])
    np.save(os.path.join(MODEL_DIR, "aligned_baseline_drift_mahal.npy"), distances_baseline["mahal"])

    # print first 10 euclid values (aligned baseline version)
    print("euclid first 10 (aligned->baseline):", distances_baseline["euclid"][:10])

        # ============================================================
    # OPTION 1 — Statistical Precursor Scoring (POST-PROCESSING)
    # ============================================================

    # Use Euclidean drift as the precursor signal
    drift_signal = distances_baseline["euclid"]

    # Robust baseline statistics
    mu = np.mean(drift_signal)
    sigma = np.std(drift_signal) + 1e-8

    # Z-score
    z_scores = (drift_signal - mu) / sigma

    # Sigmoid → probability-like precursor score
    precursor_score = 1.0 / (1.0 + np.exp(-z_scores))

    np.save(os.path.join(MODEL_DIR, "precursor_score.npy"), precursor_score)
    print("[precursor] Saved precursor_score.npy")


    # ============================================================
    # OPTION 2 — Trend & Change-Point Detection
    # ============================================================

    trend_window = 10  # rolling window size
    rolling_mean = np.convolve(
        drift_signal,
        np.ones(trend_window) / trend_window,
        mode="same"
    )

    # First derivative → trend
    drift_trend = np.gradient(rolling_mean)

    # Simple change-point rule (upper percentile)
    trend_threshold = np.percentile(drift_trend, 90)
    change_points = np.where(drift_trend > trend_threshold)[0]

    np.save(os.path.join(MODEL_DIR, "drift_trend.npy"), drift_trend)
    np.save(os.path.join(MODEL_DIR, "change_point_indices.npy"), change_points)

    print(f"[trend] Detected {len(change_points)} potential degradation onsets")


    # ============================================================
    # OPTION 3 — Remaining Useful Life (RUL) Proxy
    # ============================================================

    # Cumulative drift → degradation progress
    cumulative_drift = np.cumsum(drift_signal)
    cumulative_drift_norm = cumulative_drift / (cumulative_drift.max() + 1e-8)

    total_windows = len(drift_signal)

    # RUL in windows
    RUL_windows = (1.0 - cumulative_drift_norm) * total_windows
    RUL_windows = np.clip(RUL_windows, 0, total_windows)

    # Convert to time units using step size
    RUL_time_units = RUL_windows * step

    np.save(os.path.join(MODEL_DIR, "RUL_windows.npy"), RUL_windows)
    np.save(os.path.join(MODEL_DIR, "RUL_time_units.npy"), RUL_time_units)

    print("[RUL] Saved RUL_windows.npy and RUL_time_units.npy")



if __name__ == "__main__":
    main()
