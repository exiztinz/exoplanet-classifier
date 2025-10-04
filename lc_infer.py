# lc_infer.py
# Inference helpers for the light-curve CNN
# - Robust preprocessing: median-normalize, center transit, invert dips, robust standardize, clip
# - Interpolate to fixed length (no pad/trim artifacts)
# - Safe class mapping: use model.classes if present; otherwise fall back

import io
import numpy as np
import pandas as pd
import torch
from cnn_model import load_cnn


# Default class order fallback (will be overridden if model exposes .classes)
FALLBACK_CLASSES = ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"]


def _resample_to_length(x: np.ndarray, length: int) -> np.ndarray:
    """Resample 1-D array x to a fixed length via linear interpolation."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) == length:
        return x
    # create phase grids and interpolate
    xp = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    xi = np.linspace(0.0, 1.0, num=length, endpoint=False)
    return np.interp(xi, xp, x).astype(np.float32)


def _preprocess_flux(x: np.ndarray) -> np.ndarray:
    """
    Apply common LC-CNN inference transforms.
    Adjust this to exactly mirror your training pipeline if it differs.
    Steps:
      1) median normalize (divide by median)
      2) center deepest dip at the middle (phase-center)
      3) convert dips to positive signal: (1 - flux)
      4) robust standardize by MAD
      5) clip outliers to a sane range
    """
    x = np.asarray(x, dtype=np.float32)

    # 1) median normalize
    med = np.nanmedian(x)
    if np.isfinite(med) and med != 0.0:
        x = x / med

    # 2) center deepest dip
    try:
        k = int(np.nanargmin(x))
        shift = len(x) // 2 - k
        x = np.roll(x, shift)
    except Exception:
        pass  # if all-NaN or empty, just continue

    # 3) invert so transit is positive "signal"
    x = 1.0 - x

    # 4) robust standardize
    m = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - m))
    scale = mad if (np.isfinite(mad) and mad > 0) else (np.nanstd(x) + 1e-6)
    x = (x - m) / (scale + 1e-6)

    # 5) clip
    x = np.clip(x, -5.0, 10.0)

    # replace remaining NaNs (if any)
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    return x.astype(np.float32)


def _get_classes_from_model(model) -> list[str]:
    # Prefer an attribute set during training (e.g., model.classes = [...])
    classes = getattr(model, "classes", None)
    if isinstance(classes, (list, tuple)) and len(classes) == 3:
        return list(classes)
    return FALLBACK_CLASSES


# Expect a CSV with a 'flux' column, arbitrary length; we resample to 'length' (default 512)
def predict_lc_from_csv_bytes(csv_bytes: bytes, model_path: str = "models/cnn.pt", length: int = 512):
    df = pd.read_csv(io.BytesIO(csv_bytes))
    if "flux" not in df.columns:
        raise ValueError("CSV must include a 'flux' column.")
    x = df["flux"].to_numpy().astype("float32")
    if x.ndim != 1:
        raise ValueError("'flux' must be 1-D.")

    # Interpolate to fixed length (avoids repeating edges)
    x = _resample_to_length(x, length)
    # Preprocess to match CNN training expectations
    x = _preprocess_flux(x)

    # Run model
    model = load_cnn(model_path, n_classes=3)
    model.eval()
    device = next(model.parameters()).device if any(p.is_cuda for p in model.parameters()) else torch.device("cpu")
    with torch.no_grad():
        t = torch.from_numpy(x).view(1, 1, -1).to(device)  # (B, C, L)
        logits = model(t)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

    labels = _get_classes_from_model(model)
    return dict(zip(labels, map(float, probs)))