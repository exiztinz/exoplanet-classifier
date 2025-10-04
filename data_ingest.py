# data_ingest.py
import io
import warnings
import requests
import pandas as pd
import numpy as np

# Use TOP N during development so we iterate quickly. We'll remove TOP later.
KOI_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
    "query=select+*+from+cumulative&format=csv"
)

# Conservative, widely-present columns (we'll auto-skip any missing):
CANDIDATE_FEATURES = [
    "koi_period",      # days
    "koi_duration",    # hours
    "koi_depth",       # ppm
    "koi_prad",        # Earth radii
    "koi_snr",         # model SNR (exists in cumulative; if missing we'll skip)
    "koi_impact",      # impact parameter
    "koi_steff",       # stellar Teff (K)
    "koi_slogg",       # log g
    "koi_srad",        # stellar radius (Rsun)
]
LABEL_COL = "koi_disposition"  # CONFIRMED / CANDIDATE / FALSE POSITIVE
ID_COL = "kepid"

def _read_csv_robust(url: str) -> pd.DataFrame:
    """
    Some environments block pandas' direct HTTP; using requests gives clearer errors.
    """
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))

def load_koi(url: str = KOI_URL):
    """
    Returns:
      df: cleaned dataframe with [features + label + kepid]
      features_in_use: list[str] features actually present in this download
    """
    df = _read_csv_robust(url)
    # Replace infs and drop rows with any NaN among used columns
    df = df.replace({np.inf: np.nan, -np.inf: np.nan})

    # Only keep the columns we care about and that actually exist
    features_in_use = [c for c in CANDIDATE_FEATURES if c in df.columns]

    needed_cols = features_in_use + [LABEL_COL]
    if ID_COL in df.columns:
        needed_cols.append(ID_COL)

    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        warnings.warn(f"Some expected columns are missing and will be skipped: {missing}")

    keep = [c for c in needed_cols if c in df.columns]
    df = df[keep].dropna()

    # Keep only the 3 classes we care about
    valid = {"CONFIRMED", "CANDIDATE", "FALSE POSITIVE"}
    if LABEL_COL in df.columns:
        df = df[df[LABEL_COL].isin(valid)]

    # Final sanity check
    if len(df) == 0:
        raise RuntimeError("No rows left after cleaning; try removing TOP or loosening dropna.")

    return df, features_in_use