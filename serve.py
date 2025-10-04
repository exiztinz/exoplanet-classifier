# serve.py — Minimal Flask upload UI (no Gradio)
import io
import base64
from io import BytesIO
import matplotlib
matplotlib.use("Agg")  # server-side rendering (no GUI)
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from flask import Flask, request, render_template_string, redirect, url_for
from flask import jsonify
import os
try:
    from flask_cors import CORS
except Exception:
    CORS = None  # optional; app runs without CORS if package isn't installed
import numpy as np
from lightkurve import search_lightcurve
from lc_infer import predict_lc_from_csv_bytes
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
from astropy import units as u
from pathlib import Path
import concurrent.futures
from cnn_model import load_cnn

HAS_CNN = os.path.exists("models/cnn.pt")

CACHE_DIR = Path("cache/folded")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

model_path = "models/cnn_best.pt" if os.path.exists("models/cnn_best.pt") else "models/cnn.pt"
MODELS = {"cnn": load_cnn(model_path)}

# ---- Cache utilities ----
def _cache_iter():
    try:
        for p in CACHE_DIR.glob("*.npy"):
            if p.is_file():
                yield p
    except Exception:
        return

def _cache_stats():
    n = 0
    bytes_ = 0
    for p in _cache_iter():
        try:
            n += 1
            bytes_ += p.stat().st_size
        except Exception:
            pass
    return n, bytes_

def _cache_clear():
    removed = 0
    for p in _cache_iter():
        try:
            p.unlink(missing_ok=True)
            removed += 1
        except Exception:
            pass
    return removed

# Load trained artifact
pack = joblib.load("models/gbm_koi.joblib")
scaler = pack["scaler"]
model = pack["model"]
features = pack["features"]
label_map = pack["label_map"]
inv_label = {v: k for k, v in label_map.items()}

app = Flask(__name__)

# Small top navigation bar
NAV_HTML = """
<div style="margin-bottom:10px;">
  <a href="/">Tabular Classifier</a> ·
  <a href="/lc">Light Curve Upload</a> ·
  <a href="/fold">Fold (download via Lightkurve)</a> ·
  <a href="/eval">Evaluate</a> ·
  <a href="/cache/status" target="_blank">Cache</a> ·
  <a href="/api/schema" target="_blank">API Schema</a>
</div>
<hr>
"""
HTML_EVAL = """
<!doctype html>
<html>
  <head><meta charset="utf-8"><title>Model Evaluation</title>
  <style>
    body { font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif; margin:2rem; }
    .card { background:#f8f9fb; padding:1rem 1.25rem; border-radius:10px; border:1px solid #e8ebf0; margin-bottom: 1rem; }
    table { border-collapse: collapse; }
    th, td { border: 1px solid #ddd; padding: 6px 8px; }
    th { background: #f5f5f5; }
    .error { background:#ffe3e3; color:#7a0000; padding:.5rem 1rem; border-radius:8px; }
    .ok { background:#e6ffed; color:#064d1e; padding:.5rem 1rem; border-radius:8px; }
    code { background:#f5f5f5; padding: 1px 4px; border-radius:4px; }
    label { display:block; margin: 6px 0; }
    input[type="number"] { width: 110px; }
  </style>
  </head>
  <body>
    {{ nav|safe }}
    <h1>Evaluate Tabular Classifier vs NASA KOI labels</h1>
    <div class="card">
      <form action="{{ url_for('eval_page') }}" method="post">
        <label>Sample size (rows to fetch): <input name="n" type="number" value="1000" min="50" max="20000"></label>
        <label>Random seed: <input name="seed" type="number" value="42"></label>
        <label>Only use rows where all required features are non-null <input type="checkbox" name="dropna" checked></label>
        <label><input type="checkbox" name="run_cnn"> Also evaluate Light-Curve CNN (slow; downloads LCs)</label>
        <div style="margin-left:1.2rem;">
            <label>LC sample size: <input name="lc_n" type="number" value="40" min="10" max="200"></label>
            <label>Fast mode (download first file only) <input type="checkbox" name="lc_fast" checked></label>
            <label>Time bin (minutes): <input name="lc_bin" type="number" value="10" min="0" max="60"></label>
            <label>Resampled length: <input name="lc_len" type="number" value="512" min="128" max="2048"></label>
            <label>Use cache (speeds up re-runs): <input name="lc_cache" type="checkbox" checked></label>
            <label>Parallel workers: <input name="lc_workers" type="number" value="6" min="1" max="16"></label>
        </div>
        <button type="submit">Run evaluation</button>
      </form>
      <p class="help">This pulls live rows from NASA’s <code>cumulative</code> KOI table and compares our predictions to the archive’s <code>koi_disposition</code> (CONFIRMED / CANDIDATE / FALSE POSITIVE).</p>
    </div>
    {% if error %}<div class="error">{{ error }}</div>{% endif %}
    {% if summary %}
      <div class="ok">Accuracy: <strong>{{ acc }}%</strong></div>
      <div class="card">
        <h3>Classification Report</h3>
        <pre style="white-space:pre-wrap">{{ summary }}</pre>
      </div>
      <div class="card">
        <h3>Confusion Matrix</h3>
        {{ cm_table|safe }}
      </div>
      {% if misses_table %}
      <div class="card">
        <h3>Examples of mismatches (up to 20)</h3>
        {{ misses_table|safe }}
      </div>
      {% endif %}
    {% endif %}
    {% if lc_summary %}
        <div class="ok" style="margin-top:1rem;">LC-CNN Accuracy: <strong>{{ lc_acc }}%</strong> (success {{ lc_success }}/{{ lc_attempted }}; failed {{ lc_failed }})</div>
        <div class="card">
            <h3>LC-CNN Classification Report</h3>
            <pre style="white-space:pre-wrap">{{ lc_summary }}</pre>
        </div>
        <div class="card">
            <h3>LC-CNN Confusion Matrix</h3>
            {{ lc_cm_table|safe }}
        </div>
        {% if lc_failures %}
        <div class="card">
            <h3>Examples of fold/predict failures (up to 20)</h3>
            <pre style="white-space:pre-wrap">{{ lc_failures }}</pre>
        </div>
        {% endif %}
    {% endif %}
    <p style="margin-top:1rem;"><a href="{{ url_for('home') }}">← Back</a></p>
  </body>
</html>
"""

# Helper to fetch KOI rows with our features and labels from the NASA API
def _fetch_koi_dataframe(n=1000, seed=42):
    # Build TAP SQL selecting our features + koi_disposition
    cols = ["koi_disposition"] + list(features)
    select_cols = ",".join(cols)
    # TOP with random ordering using seed via ORDER BY koi_kepmag (proxy) + seed is not used in TAP; we shuffle locally.
    url = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
        f"query=select+top+{int(n)}+{select_cols}+from+cumulative&format=csv"
    )
    df = pd.read_csv(url)
    # Shuffle locally with seed for reproducibility
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df

# Helper to format confusion matrix into HTML
def _cm_to_html(cm, labels):
    # cm shape (L,L), rows = true, cols = pred
    header = "<tr><th></th>" + "".join(f"<th>pred {l}</th>" for l in labels) + "</tr>"
    rows_html = []
    for i, l in enumerate(labels):
        cells = "".join(f"<td>{int(cm[i,j])}</td>" for j in range(len(labels)))
        rows_html.append(f"<tr><th>true {l}</th>{cells}</tr>")
    return f"<table>{header}{''.join(rows_html)}</table>"

# Helper: Convert a Matplotlib figure to a base64-encoded PNG string
def _fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")

# Limit uploads to 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Enable CORS for /api/* if flask-cors is available
if CORS is not None:
    CORS(app, resources={r"/api/*": {"origins": "*"}})

# Check for optional CNN model presence
HAS_CNN = os.path.exists("models/cnn.pt")

HTML_FORM = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Exoplanet (KOI) Classifier — Tabular MVP</title>
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; margin: 2rem; }
      .container { max-width: 1000px; margin: auto; }
      h1 { margin-bottom: 0.5rem; }
      .help { color: #555; margin-bottom: 1rem; }
      .error { background: #ffe3e3; padding: 0.75rem 1rem; border-radius: 8px; color: #7a0000; }
      .ok { background: #e6ffed; padding: 0.75rem 1rem; border-radius: 8px; color: #064d1e; }
      form { margin: 1rem 0; }
      input[type="file"] { margin: 0.5rem 0; }
      table { border-collapse: collapse; width: 100%; }
      th, td { border: 1px solid #ddd; padding: 8px; }
      th { background: #f5f5f5; }
      .footer { margin-top: 2rem; color: #777; font-size: 0.9rem; }
      code { background: #f5f5f5; padding: 2px 4px; border-radius: 4px; }
    </style>
  </head>
  <body>
    {{ nav|safe }}
    <div class="container">
      <h1>Exoplanet (KOI) Classifier — Tabular MVP</h1>
      <div class="help">
        Upload a CSV with these numeric columns:
        <code>{{ feature_list }}</code>
      </div>
      {% if error %}
        <div class="error">{{ error }}</div>
      {% endif %}
      {% if ok %}
        <div class="ok">{{ ok }}</div>
      {% endif %}
      <form action="{{ url_for('predict') }}" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept=".csv" required>
        <div>
          <button type="submit">Predict</button>
        </div>
      </form>
      {% if table_html %}
        <h2>Results</h2>
        {{ table_html|safe }}
      {% endif %}
      <div class="footer">
        Tip: create a valid template quickly:
        <pre>python -c "import joblib,pandas as pd; p=joblib.load('models/gbm_koi.joblib'); cols=p['features']; pd.DataFrame([{c:1.0 for c in cols}]).to_csv('demo.csv',index=False)"</pre>
      </div>
      <div class="footer">
        <h2 style="margin:1.2rem 0 0.3rem 0;">Light‑Curve CNN evaluation API (advanced)</h2>
        <p>You can programmatically evaluate the light‑curve CNN via <code>GET /api/eval_lc</code>. It requires query parameters:</p>
        <ul>
          <li><code>n</code>: number of KOIs to sample (e.g., 40)</li>
          <li><code>seed</code>: random seed</li>
          <li><code>fast</code>: true/false (download first file only)</li>
          <li><code>bin</code>: time bin in minutes (e.g., 10)</li>
          <li><code>length</code>: resampled length (e.g., 512)</li>
          <li><code>cache</code>: true/false (use folded-LC cache)</li>
          <li><code>workers</code>: parallel threads (e.g., 6)</li>
        </ul>
        <p>Example:</p>
        <pre>curl -s "http://127.0.0.1:7860/api/eval_lc?n=40&amp;seed=42&amp;fast=true&amp;bin=10&amp;length=512&amp;cache=true&amp;workers=6" | python -m json.tool</pre>
      </div>
    </div>
  </body>
</html>
"""

HTML_LC = """
<!doctype html>
<html>
  <head><meta charset="utf-8"><title>Light Curve (CNN) — MVP</title></head>
  <body style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;margin:2rem;">
    {{ nav|safe }}
    <h1>Light Curve (CNN) — MVP</h1>
    <p>Upload a phase-folded, detrended light curve CSV with a single column: <code>flux</code> (≈512 rows).</p>
    <form action="{{ url_for('predict_lc') }}" method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept=".csv" required>
      <div><button type="submit">Predict</button></div>
    </form>
    {% if error %}<div style="color:#7a0000;background:#ffe3e3;padding:.5rem 1rem;border-radius:8px;margin-top:1rem;">{{ error }}</div>{% endif %}
    {% if top_label %}
      <h2>Result</h2>
      <div style="background:#e6ffed;padding:.5rem 1rem;border-radius:8px;display:inline-block;">
        Top prediction: <strong>{{ top_label }}</strong> ({{ top_confidence }}%)
      </div>
      {% if planet_score %}
        <div style="margin-top:10px;background:#eef6ff;padding:.5rem 1rem;border-radius:8px;display:inline-block;">
          Planet‑likeness: <strong>{{ planet_score }}%</strong> — <em>{{ planet_verdict }}</em>
          <span style="color:#555;margin-left:8px;">(computed as P(CONFIRMED) + α·P(CANDIDATE), α={{ alpha }}, thresholds: high ≥ {{ hi*100 }}%, low ≤ {{ lo*100 }}%)</span>
        </div>
      {% endif %}
    {% endif %}
    {% if probs_table %}
      <h3 style="margin-top:1rem;">Class probabilities</h3>
      {{ probs_table|safe }}
    {% endif %}
    {% if lc_img %}
      <h3 style="margin-top:1rem;">Folded light curve</h3>
      <img alt="Folded light curve" src="data:image/png;base64,{{ lc_img }}" style="max-width:720px;border:1px solid #eee;border-radius:6px;">
    {% endif %}
    {% if probs_img %}
      <h3 style="margin-top:1rem;">Probability breakdown</h3>
      <img alt="Class probabilities" src="data:image/png;base64,{{ probs_img }}" style="max-width:480px;border:1px solid #eee;border-radius:6px;">
    {% endif %}
    <p style="margin-top:2rem;"><a href="{{ url_for('home') }}">← Back to tabular</a></p>
  </body>
</html>
"""

HTML_FOLD = """
<!doctype html>
<html>
  <head><meta charset="utf-8"><title>Fold Light Curve — (Lightkurve)</title></head>
  <body style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;margin:2rem;">
    {{- nav|safe -}}
    <div class="container">
    <h1>Fold Light Curve (download via Lightkurve)</h1>
    <p>Enter a target and period to download, flatten, and phase-fold a light curve. Output is resampled to a fixed length.</p>
    <form action="{{ url_for('fold_page') }}" method="post">
      <label>Mission:
        <select name="mission">
          <option value="TESS">TESS</option>
          <option value="Kepler">Kepler</option>
          <option value="K2">K2</option>
        </select>
      </label>
      <br><br>
      <label>Target ID (TIC for TESS, KIC for Kepler/K2): <input type="text" name="target" required></label>
      <br><br>
      <label>Period (days): <input type="number" step="any" name="period" required></label>
      <br><br>
      <label>T0 / reference time (BKJD/BJD; optional): <input type="number" step="any" name="t0"></label>
      <br><br>
      <label>Output length (samples): <input type="number" name="length" value="512"></label>
      <br><br>
      <label>Use cache: <input type="checkbox" name="use_cache" checked></label>
      <br><br>
      <button type="submit">Download &amp; Fold</button>
    </form>
    {% if error %}<div style="color:#7a0000;background:#ffe3e3;padding:.5rem 1rem;border-radius:8px;margin-top:1rem;">{{ error }}</div>{% endif %}
    {% if preview_json %}
      <h2>Preview (first 40 samples, JSON)</h2>
      <pre style="white-space:pre-wrap">{{ preview_json|e }}</pre>
      <p>Use this JSON directly with <code>/api/predict_lc</code> as the value for key <code>flux</code>.</p>
    {% endif %}
    {% if full_json %}
      <details style="margin-top:1rem;">
        <summary>Show full JSON ({{ length }} samples)</summary>
        <textarea style="width:100%;height:180px;">{{ full_json|e }}</textarea>
      </details>
      <p style="margin-top:0.5rem;">
        <a href="{{ download_url }}">⬇️ Download CSV</a> ·
        <a href="{{ api_url }}" target="_blank">View JSON</a>
      </p>
    {% endif %}
    {% if plot_url %}
      <h3 style="margin-top:1rem;">Folded light curve</h3>
      <img alt="Folded light curve" src="{{ plot_url }}" style="max-width:720px;border:1px solid #eee;border-radius:6px;">
    {% endif %}
    <p style="margin-top:2rem;"><a href="{{ url_for('home') }}">← Back to tabular</a> · <a href="{{ url_for('lc_home') }}">LC upload</a></p>
    </div>
  </body>
</html>
"""

def _infer_from_dataframe(df: pd.DataFrame):
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    Xs = scaler.transform(df[features].to_numpy())
    proba = model.predict_proba(Xs)
    pred_idx = proba.argmax(axis=1)
    pred_labels = [inv_label[i] for i in pred_idx]
    proba_named = [
        {inv_label[i]: float(p[i]) for i in range(len(inv_label))}
        for p in proba
    ]
    return pred_labels, proba_named

def _cache_key(mission: str, target: str, period_days: float, length: int, bin_minutes: int | None, fast: bool) -> Path:
    safe_target = target.replace(" ", "_").replace("/", "_")
    bn = bin_minutes if (bin_minutes is not None) else 0
    fname = f"{mission}_{safe_target}_P{period_days:.8f}_L{length}_B{bn}_F{int(bool(fast))}.npy"
    return CACHE_DIR / fname

def _fold_cached(mission: str, target: str, period_days: float, t0: float | None, length: int, fast: bool, bin_minutes: int | None) -> np.ndarray:
    p = _cache_key(mission, target, period_days, length, bin_minutes, fast)
    if p.exists():
        try:
            return np.load(p)
        except Exception:
            pass
    arr = fold_light_curve(mission, target, period_days, t0=t0, length=length, fast=fast, time_bin_minutes=bin_minutes)
    try:
        np.save(p, arr)
    except Exception:
        pass
    return arr

def _fetch_koi_for_lc(n=80, seed=42):
    # Grab extra rows to account for potential fold/download failures
    url = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
        f"query=select+top+{int(n*3)}+kepid,koi_period,koi_disposition+from+cumulative&format=csv"
    )
    df = pd.read_csv(url)
    df = df[df["koi_disposition"].isin(list(inv_label.values()))].copy()
    df = df.dropna(subset=["kepid", "koi_period"])
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df

def _eval_cnn_on_koi(n=50, seed=42, fast=True, bin_minutes=10, length=512, use_cache=True, workers=6):
    rows = _fetch_koi_for_lc(n=n, seed=seed)

    def _one(row):
        kic = int(row["kepid"])
        period = float(row["koi_period"])
        true_lbl = str(row["koi_disposition"])
        target = f"KIC {kic}"
        try:
            flux = _fold_cached("Kepler", target, period, t0=None, length=length, fast=fast, bin_minutes=bin_minutes) if use_cache else \
                   fold_light_curve("Kepler", target, period, t0=None, length=length, fast=fast, time_bin_minutes=bin_minutes)
            buf = io.StringIO()
            pd.DataFrame({"flux": flux}).to_csv(buf, index=False)
            probs = predict_lc_from_csv_bytes(buf.getvalue().encode("utf-8"), model_path="models/cnn.pt", length=length)
            pred_lbl = max(probs.items(), key=lambda kv: kv[1])[0]
            return {"ok": True, "target": target, "period": period, "true": true_lbl, "pred": pred_lbl}
        except Exception as e:
            return {"ok": False, "target": target, "period": period, "error": str(e)}

    y_true, y_pred, failures = [], [], []
    attempted = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = [ex.submit(_one, r) for _, r in rows.iterrows()]
        for fut in concurrent.futures.as_completed(futs):
            res = fut.result()
            attempted += 1
            if res["ok"]:
                y_true.append(res["true"])
                y_pred.append(res["pred"])
                if len(y_true) >= n:
                    break
            else:
                failures.append({"target": res["target"], "period": res["period"], "error": res["error"]})

    if len(y_true) == 0:
        raise RuntimeError("No successful LC evaluations; check network or Lightkurve install.")

    labels = ["CONFIRMED","CANDIDATE","FALSE POSITIVE"]
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "n_attempted": attempted,
        "n_success": int(len(y_true)),
        "n_failed": int(len(failures)),
        "accuracy": acc,
        "labels": labels,
        "cm": cm,
        "report": report,
        "failures": failures[:20],
    }

def fold_light_curve(mission: str, target: str, period_days: float, t0: float | None = None, length: int = 512, fast: bool = True, time_bin_minutes: int | None = 10):
    """
    Downloads a light curve with Lightkurve, removes NaNs, flattens (detrends),
    folds on the given period, sorts by phase, normalizes to median=1,
    and resamples to a fixed length array.

    Returns: np.ndarray of shape (length,), dtype float32
    """
    # Search by mission
    if mission not in {"TESS", "Kepler", "K2"}:
        raise ValueError("mission must be one of: TESS, Kepler, K2")

    srch = search_lightcurve(target, mission=mission)
    if len(srch) == 0:
        raise RuntimeError(f"No light curve found for {mission} {target}")

    # FAST mode: try a single file first to speed up cold-start
    lc = None
    if fast:
        try:
            lc = srch[0].download()
        except Exception:
            lc = None

    # If fast failed or not fast, download & stitch all available light curves
    if lc is None:
        lc = srch.download_all().stitch()

    if lc is None:
        raise RuntimeError("Failed to download/stitch light curve.")

    # Clean and flatten
    lc = lc.remove_nans()
    # Optional: bin in time to reduce size (faster flatten/fold)
    if time_bin_minutes is not None and time_bin_minutes > 0:
        try:
            lc = lc.bin(time_bin_size=time_bin_minutes * u.min)
        except Exception:
            pass
    # lightkurve flatten uses window_length in cadences; pick a modest default
    try:
        lc = lc.flatten(window_length=401)
    except Exception:
        # fallback smaller window if cadence is short
        lc = lc.flatten(window_length=201)

    # Fold on the provided period (returns a FoldedLightCurve)
    folded = lc.fold(period=period_days, t0=t0) if t0 is not None else lc.fold(period=period_days)

    # Extract numpy arrays from FoldedLightCurve (Quantity -> .value)
    phase = np.array(folded.phase.value, dtype=np.float64)
    flux  = np.array(folded.flux.value,  dtype=np.float64)

    # Sort by phase for a stable vector
    order = np.argsort(phase)
    phase = phase[order]
    flux  = flux[order]

    # Normalize to median ~ 1 to stabilize scale across missions
    med = np.nanmedian(flux) if flux.size else 1.0
    if not np.isfinite(med) or med == 0:
        med = 1.0
    flux = flux / med

    # Replace any remaining NaNs with local median
    if np.any(~np.isfinite(flux)):
        good = np.isfinite(flux)
        if good.any():
            flux[~good] = np.nanmedian(flux[good])
        else:
            flux[~good] = 1.0

    # Resample to fixed length via linear interpolation over phase domain
    # (phase already extracted & sorted above)
    mask = np.isfinite(phase) & np.isfinite(flux)
    phase, flux = phase[mask], flux[mask]
    if phase.size < 4:
        raise RuntimeError("Not enough valid points after cleaning.")

    # target grid
    grid = np.linspace(phase.min(), phase.max(), length, endpoint=False)
    # simple interpolation
    resampled = np.interp(grid, phase, flux).astype("float32")

    return resampled

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_FORM, nav=NAV_HTML, feature_list=", ".join(features), error=None, ok=None, table_html=None)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template_string(HTML_FORM, nav=NAV_HTML, feature_list=", ".join(features), error="No file uploaded.", ok=None, table_html=None)
    f = request.files["file"]
    if f.filename == "":
        return render_template_string(HTML_FORM, nav=NAV_HTML, feature_list=", ".join(features), error="Empty filename.", ok=None, table_html=None)
    try:
        data = f.read()
        df = pd.read_csv(io.BytesIO(data))
    except Exception as e:
        return render_template_string(HTML_FORM, nav=NAV_HTML, feature_list=", ".join(features), error=f"Could not read CSV: {e}", ok=None, table_html=None)

    try:
        # Ensure numeric types for required feature columns
        df[features] = df[features].apply(pd.to_numeric, errors="raise")
        preds, probas = _infer_from_dataframe(df)
        out = df.copy()
        out["prediction"] = preds
        for name in inv_label.values():
            out[f"P({name})"] = [p[name] for p in probas]
        table_html = out.to_html(index=False)
    except Exception as e:
        return render_template_string(HTML_FORM, nav=NAV_HTML, feature_list=", ".join(features), error=f"Inference error: {e}", ok=None, table_html=None)

    return render_template_string(HTML_FORM, nav=NAV_HTML, feature_list=", ".join(features), error=None, ok="OK", table_html=table_html)

@app.route("/api/schema", methods=["GET"])
def api_schema():
    return jsonify({
        "features": features,
        "classes": list(inv_label.values())
    })

@app.route("/api/predict", methods=["POST"])
def api_predict():
    # Accept either JSON with {"rows": [ {...}, {...} ]} or a raw CSV file via multipart/form-data
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        rows = payload.get("rows")
        if not isinstance(rows, list) or not rows:
            return jsonify({"error": "Provide JSON body with key 'rows': list[dict]."}), 400
        df = pd.DataFrame(rows)
        try:
            df[features] = df[features].apply(pd.to_numeric, errors="raise")
            preds, probas = _infer_from_dataframe(df)
            return jsonify({
                "predictions": preds,
                "probabilities": probas
            })
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 400
        except Exception as e:
            return jsonify({"error": f"Inference error: {e}"}), 500

    # If not JSON, try to read a file part named 'file'
    if "file" in request.files:
        try:
            data = request.files["file"].read()
            df = pd.read_csv(io.BytesIO(data))
            df[features] = df[features].apply(pd.to_numeric, errors="raise")
            preds, probas = _infer_from_dataframe(df)
            return jsonify({
                "predictions": preds,
                "probabilities": probas
            })
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 400
        except Exception as e:
            return jsonify({"error": f"Inference error: {e}"}), 500

    return jsonify({"error": "Unsupported content. Send JSON {'rows': [...]} or multipart 'file' CSV."}), 415

@app.route("/lc", methods=["GET"])
def lc_home():
    return render_template_string(
        HTML_LC,
        nav=NAV_HTML,
        error=None if HAS_CNN else "CNN model not found (models/cnn.pt). Run `python train_cnn.py` first.",
        top_label=None,
        top_confidence=None,
        probs_table=None,
        planet_score=None,
        planet_verdict=None,
        alpha=1.0,
        hi=0.80,
        lo=0.20,
    )

@app.route("/lc", methods=["POST"])
def predict_lc():
    if not HAS_CNN:
        return render_template_string(
            HTML_LC,
            nav=NAV_HTML,
            error="CNN model not found (models/cnn.pt). Run `python train_cnn.py` first.",
            top_label=None,
            top_confidence=None,
            probs_table=None,
            planet_score=None,
            planet_verdict=None,
            alpha=1.0,
            hi=0.80,
            lo=0.20,
            lc_img=None,
            probs_img=None,
        )
    if "file" not in request.files:
        return render_template_string(
            HTML_LC,
            nav=NAV_HTML,
            error="No file uploaded.",
            top_label=None,
            top_confidence=None,
            probs_table=None,
            planet_score=None,
            planet_verdict=None,
            alpha=1.0,
            hi=0.80,
            lo=0.20,
            lc_img=None,
            probs_img=None,
        )
    f = request.files["file"]
    try:
        csv_bytes = f.read()
        probs = predict_lc_from_csv_bytes(csv_bytes, model_path="models/cnn.pt", length=512)
        # Build friendly table and top prediction
        items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        # ---------- Build LC plot from uploaded CSV ----------
        try:
            df_plot = pd.read_csv(io.BytesIO(csv_bytes))
            if "flux" in df_plot.columns:
                flux_arr = df_plot["flux"].to_numpy(dtype="float32")
                # x-axis as phase [0,1)
                x = np.linspace(0.0, 1.0, num=len(flux_arr), endpoint=False)
                fig1, ax1 = plt.subplots(figsize=(6.5, 3))
                ax1.plot(x, flux_arr, lw=1)
                ax1.set_xlabel("Phase (0 → 1)")
                ax1.set_ylabel("Normalized Flux")
                ax1.set_title("Uploaded folded light curve")
                ax1.grid(True, alpha=0.3)
                lc_img = _fig_to_base64(fig1)
            else:
                lc_img = None
        except Exception:
            lc_img = None

        # ---------- Build probability bar chart ----------
        labels = [lbl for lbl, _ in items]
        vals = [p for _, p in items]
        fig2, ax2 = plt.subplots(figsize=(4.5, 2.6))
        y_pos = np.arange(len(labels))
        ax2.barh(y_pos, vals)
        ax2.set_yticks(y_pos, labels=labels)
        ax2.set_xlim(0, 1)
        ax2.set_xlabel("Probability")
        ax2.set_title("Class probabilities")
        for i, v in enumerate(vals):
            ax2.text(v + 0.01, i, f"{v*100:.1f}%", va="center")
        probs_img = _fig_to_base64(fig2)

        # Planet-likeness computation (binary view)
        alpha = 1.0   # weight for CANDIDATE contribution
        hi, lo = 0.80, 0.20  # decision thresholds
        p_conf = float(probs.get("CONFIRMED", 0.0))
        p_cand = float(probs.get("CANDIDATE", 0.0))
        planet_score = p_conf + alpha * p_cand  # in [0,1]
        if planet_score >= hi:
            verdict = "Planet-like"
        elif planet_score <= lo:
            verdict = "Not planet-like"
        else:
            verdict = "Uncertain"
        top_label, top_p = items[0]
        # HTML table
        rows = "".join(
            f"<tr><td>{lbl}</td><td>{p*100:.2f}%</td></tr>"
            for lbl, p in items
        )
        table_html = (
            "<table style='border-collapse:collapse;min-width:360px;'>"
            "<thead><tr><th style='text-align:left;border-bottom:1px solid #ddd;padding:6px 8px;'>Class</th>"
            "<th style='text-align:left;border-bottom:1px solid #ddd;padding:6px 8px;'>Probability</th></tr></thead>"
            "<tbody>" + rows + "</tbody></table>"
        )
        return render_template_string(
            HTML_LC,
            nav=NAV_HTML,
            error=None,
            top_label=top_label,
            top_confidence=f"{top_p*100:.2f}",
            probs_table=table_html,
            planet_score=f"{planet_score*100:.2f}",
            planet_verdict=verdict,
            alpha=alpha,
            hi=hi,
            lo=lo,
            lc_img=lc_img,
            probs_img=probs_img,
        )
    except Exception as e:
        return render_template_string(
            HTML_LC,
            nav=NAV_HTML,
            error=f"Error: {e}",
            top_label=None,
            top_confidence=None,
            probs_table=None,
            planet_score=None,
            planet_verdict=None,
            alpha=1.0,
            hi=0.80,
            lo=0.20,
            lc_img=None,
            probs_img=None,
        )

@app.route("/api/predict_lc", methods=["GET", "POST"])
def api_predict_lc():
    if not HAS_CNN:
        return jsonify({"error": "CNN model not found (models/cnn.pt). Run `python train_cnn.py` first."}), 503
    if request.method == "GET":
        return jsonify({
            "usage": "POST JSON {'flux': [ ... ]} or {'rows': [{'flux': ...}, ...]} OR multipart 'file' (CSV with 'flux').",
            "optional_params_in_json": {"alpha": "float, default 1.0", "hi": "float threshold in [0,1], default 0.80", "lo": "float threshold in [0,1], default 0.20"},
            "notes": [
                "Planet-likeness = P(CONFIRMED) + alpha * P(CANDIDATE).",
                "Verdict: 'Planet-like' if score >= hi; 'Not planet-like' if score <= lo; else 'Uncertain'."
            ],
            "example": {"flux": [1.0, 1.0, 0.99, 0.98, 1.0, "..."], "alpha": 1.0, "hi": 0.8, "lo": 0.2}
        })
    try:
        if request.is_json:
            # JSON body: {"flux": [ ... ]} OR {"rows": [{"flux": ...}, ...]} (first wins)
            js = request.get_json(silent=True) or {}
            alpha = float(js.get("alpha", 1.0))
            hi = float(js.get("hi", 0.80))
            lo = float(js.get("lo", 0.20))
            # --- BEGIN REPLACEMENT BLOCK ---
            # Determine target length from loaded CNN (fallback 512)
            try:
                target_len = int(getattr(MODELS.get("cnn"), "length", 512))
            except Exception:
                target_len = 512

            if "flux" in js:
                import numpy as np, pandas as pd, io
                arr = np.asarray(js["flux"], dtype=np.float32)
                # Resample to the CNN's expected length if needed
                if arr.size != target_len and arr.size > 1:
                    x_old = np.linspace(0.0, 1.0, arr.size, endpoint=False, dtype=np.float32)
                    x_new = np.linspace(0.0, 1.0, target_len, endpoint=False, dtype=np.float32)
                    arr = np.interp(x_new, x_old, arr).astype(np.float32)
                df = pd.DataFrame({"flux": arr})
                csv_bytes = df.to_csv(index=False).encode("utf-8")
            elif "rows" in js:
                import pandas as pd, io
                df = pd.DataFrame(js["rows"])  # expect a column named 'flux'
                # If provided length doesn't match, resample here too
                if "flux" in df.columns:
                    import numpy as np
                    arr = df["flux"].to_numpy(dtype=np.float32)
                    if arr.size != target_len and arr.size > 1:
                        x_old = np.linspace(0.0, 1.0, arr.size, endpoint=False, dtype=np.float32)
                        x_new = np.linspace(0.0, 1.0, target_len, endpoint=False, dtype=np.float32)
                        arr = np.interp(x_new, x_old, arr).astype(np.float32)
                        df = pd.DataFrame({"flux": arr})
                csv_bytes = df.to_csv(index=False).encode("utf-8")
            else:
                return jsonify({"error":"Provide JSON with 'flux': list[float] or 'rows': [{'flux':...},...]"}), 400
            # --- END REPLACEMENT BLOCK ---
            probs = predict_lc_from_csv_bytes(csv_bytes, model_path="models/cnn.pt", length=512)
            items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
            p_conf = float(probs.get("CONFIRMED", 0.0))
            p_cand = float(probs.get("CANDIDATE", 0.0))
            planet_score = p_conf + alpha * p_cand
            if planet_score >= hi:
                verdict = "Planet-like"
            elif planet_score <= lo:
                verdict = "Not planet-like"
            else:
                verdict = "Uncertain"
            top_label, top_p = items[0]
            return jsonify({
                "prediction": top_label,
                "confidence": top_p,
                "probabilities": probs,
                "probabilities_sorted": [{"label": lbl, "prob": p} for lbl, p in items],
                "planet_score": planet_score,
                "planet_likeness_params": {"alpha": alpha, "hi": hi, "lo": lo},
                "verdict": verdict
            })
        elif "file" in request.files:
            alpha = float(request.args.get("alpha", 1.0))
            hi = float(request.args.get("hi", 0.80))
            lo = float(request.args.get("lo", 0.20))
            probs = predict_lc_from_csv_bytes(request.files["file"].read(), model_path="models/cnn.pt", length=512)
            items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
            p_conf = float(probs.get("CONFIRMED", 0.0))
            p_cand = float(probs.get("CANDIDATE", 0.0))
            planet_score = p_conf + alpha * p_cand
            if planet_score >= hi:
                verdict = "Planet-like"
            elif planet_score <= lo:
                verdict = "Not planet-like"
            else:
                verdict = "Uncertain"
            top_label, top_p = items[0]
            return jsonify({
                "prediction": top_label,
                "confidence": top_p,
                "probabilities": probs,
                "probabilities_sorted": [{"label": lbl, "prob": p} for lbl, p in items],
                "planet_score": planet_score,
                "planet_likeness_params": {"alpha": alpha, "hi": hi, "lo": lo},
                "verdict": verdict
            })
        else:
            return jsonify({"error":"Send JSON or multipart 'file' with a CSV containing 'flux'."}), 415
    except Exception as e:
        return jsonify({"error": f"{e}"}), 500

@app.route("/fold", methods=["GET", "POST"])
def fold_page():
    if request.method == "GET":
        return render_template_string(
            HTML_FOLD,
            nav=NAV_HTML,
            error=None,
            preview_json=None,
            full_json=None,
            length=None,
            download_url=None,
            api_url=None,
            plot_url=None,
        )
    # POST:
    try:
        mission = request.form.get("mission", "TESS")
        target = request.form.get("target", "").strip()
        period = float(request.form.get("period"))
        t0_raw = request.form.get("t0", "").strip()
        t0 = float(t0_raw) if t0_raw else None
        length = int(request.form.get("length", 512))
        if not target:
            raise ValueError("Target is required.")
        use_cache = (request.form.get("use_cache") == "on")
        if use_cache:
            flux = _fold_cached(mission, target, period, t0=t0, length=length, fast=True, bin_minutes=10)
        else:
            flux = fold_light_curve(mission, target, period, t0=t0, length=length)
        preview_json = json.dumps([float(v) for v in flux[:40].tolist()], indent=2)
        full_json = json.dumps([float(v) for v in flux.tolist()], indent=2)
        download_url = url_for('fold_download', mission=mission, target=target, period=period, t0=t0, length=length)
        api_url = url_for('api_fold', mission=mission, target=target, period=period, t0=t0, length=length)
        plot_url = url_for('fold_plot_png', mission=mission, target=target, period=period, t0=t0, length=length)
        return render_template_string(
            HTML_FOLD,
            nav=NAV_HTML,
            error=None,
            preview_json=preview_json,
            full_json=full_json,
            length=length,
            download_url=download_url,
            api_url=api_url,
            plot_url=plot_url,
        )
    except Exception as e:
        return render_template_string(
            HTML_FOLD,
            nav=NAV_HTML,
            error=f"{e}",
            preview_json=None,
            full_json=None,
            length=None,
            download_url=None,
            api_url=None,
            plot_url=None,
        )

@app.route("/fold/plot.png", methods=["GET"])
def fold_plot_png():
    try:
        mission = request.args.get("mission", "TESS")
        target = (request.args.get("target") or "").strip()
        if not target:
            return jsonify({"error": "Missing 'target'"}), 400
        period = float(request.args["period"])
        t0 = request.args.get("t0")
        t0 = float(t0) if t0 not in (None, "",) else None
        length = int(request.args.get("length", 512))
        use_cache = request.args.get("cache", "true").lower() in ("1","true","yes","on")
        if use_cache:
            flux = _fold_cached(mission, target, period, t0=t0, length=length, fast=True, bin_minutes=10)
        else:
            flux = fold_light_curve(mission, target, period, t0=t0, length=length)
        phase = np.linspace(0.0, 1.0, num=len(flux), endpoint=False)
        fig, ax = plt.subplots(figsize=(6.5, 3))
        ax.plot(phase, flux, lw=1)
        ax.set_xlabel("Phase (0 → 1)")
        ax.set_ylabel("Normalized Flux")
        ax.set_title(f"{mission} {target} — P={period:.5f} d")
        ax.grid(True, alpha=0.3)
        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return app.response_class(buf.getvalue(), mimetype="image/png")
    except Exception as e:
        return jsonify({"error": f"{e}"}), 500
@app.route("/fold/download", methods=["GET"])
def fold_download():
    try:
        mission = request.args.get("mission", "TESS")
        target = (request.args.get("target") or "").strip()
        if not target:
            return jsonify({"error": "Missing 'target'"}), 400
        period = float(request.args["period"])
        t0 = request.args.get("t0")
        t0 = float(t0) if t0 not in (None, "",) else None
        length = int(request.args.get("length", 512))
        use_cache = request.args.get("cache", "true").lower() in ("1","true","yes","on")
        if use_cache:
            flux = _fold_cached(mission, target, period, t0=t0, length=length, fast=True, bin_minutes=10)
        else:
            flux = fold_light_curve(mission, target, period, t0=t0, length=length)
        csv = "flux\n" + "\n".join(str(float(v)) for v in flux.tolist())
        return app.response_class(
            csv,
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename=folded_{mission}_{target.replace(' ','_')}.csv"}
        )
    except Exception as e:
        return jsonify({"error": f"{e}"}), 500

@app.route("/api/fold", methods=["GET"])
def api_fold():
    """
    Query params:
      mission: TESS|Kepler|K2
      target: TIC... or KIC...
      period: float (days)
      t0: float (optional)
      length: int (default 512)
      cache: bool-like (true/false, default true)
    """
    try:
        mission = request.args.get("mission", "TESS")
        target = (request.args.get("target") or "").strip()
        if not target:
            return jsonify({"error": "Missing 'target'"}), 400
        period = float(request.args["period"])  # raises if missing
        t0 = request.args.get("t0")
        t0 = float(t0) if t0 not in (None, "",) else None
        length = int(request.args.get("length", 512))
        use_cache = request.args.get("cache", "true").lower() in ("1","true","yes","on")
        if use_cache:
            flux = _fold_cached(mission, target, period, t0=t0, length=length, fast=True, bin_minutes=10)
        else:
            flux = fold_light_curve(mission, target, period, t0=t0, length=length)
        return jsonify({
            "mission": mission,
            "target": target,
            "period": period,
            "t0": t0,
            "length": length,
            "flux": flux.tolist()
        })
    except KeyError as ke:
        return jsonify({"error": f"Missing parameter: {ke}"}), 400
    except Exception as e:
        return jsonify({"error": f"{e}"}), 500

# ---- Cache status and clear endpoints ----
@app.route("/cache/status", methods=["GET"])
def cache_status():
    n, bytes_ = _cache_stats()
    mb = bytes_ / (1024*1024.0)
    return render_template_string("""
    <!doctype html><html><head><meta charset="utf-8"><title>Cache</title>
    <style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;margin:2rem}</style>
    </head><body>
    {{ nav|safe }}
    <h1>Folded LC Cache</h1>
    <p>Files: <strong>{{ n }}</strong> &middot; Size: <strong>{{ mb|round(2) }} MB</strong></p>
    <p><a href="{{ url_for('cache_clear') }}">🧹 Clear cache</a></p>
    </body></html>
    """, nav=NAV_HTML, n=n, mb=mb)

@app.route("/cache/clear", methods=["GET"])
def cache_clear():
    removed = _cache_clear()
    return render_template_string("""
    <!doctype html><html><head><meta charset="utf-8"><title>Cache Cleared</title>
    <style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;margin:2rem}</style>
    </head><body>
    {{ nav|safe }}
    <h1>Cache cleared</h1>
    <p>Removed {{ removed }} file(s).</p>
    <p><a href="{{ url_for('cache_status') }}">Back to cache status</a></p>
    </body></html>
    """, nav=NAV_HTML, removed=removed)

@app.route("/eval", methods=["GET", "POST"])
def eval_page():
    if request.method == "GET":
        return render_template_string(HTML_EVAL, nav=NAV_HTML, error=None, summary=None, acc=None, cm_table=None, misses_table=None)
    try:
        n = int(request.form.get("n", 1000))
        seed = int(request.form.get("seed", 42))
        dropna = request.form.get("dropna") == "on"
        df = _fetch_koi_dataframe(n=n, seed=seed)
        # Standardize label names to match our model's labels
        df = df[df["koi_disposition"].isin(list(inv_label.values()))].copy()
        if dropna:
            df = df.dropna(subset=features)
        if df.empty:
            return render_template_string(HTML_EVAL, nav=NAV_HTML, error="No rows available after filtering. Try disabling 'dropna' or increasing sample size.", summary=None, acc=None, cm_table=None, misses_table=None)
        # Ensure numeric
        for c in features:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if dropna:
            df = df.dropna(subset=features)
            if df.empty:
                return render_template_string(HTML_EVAL, nav=NAV_HTML, error="No rows left after numeric coercion.", summary=None, acc=None, cm_table=None, misses_table=None)
        Xs = scaler.transform(df[features].to_numpy())
        y_true = df["koi_disposition"].to_numpy()
        y_pred_idx = model.predict(Xs)
        # Some sklearn models return label indices; map to labels if needed
        try:
            # If y_pred_idx already labels (strings), leave; else map via inv_label
            if isinstance(y_pred_idx[0], str):
                y_pred = y_pred_idx
            else:
                y_pred = pd.Series(y_pred_idx).map(inv_label).to_numpy()
        except Exception:
            # Fallback via predict_proba argmax
            proba = model.predict_proba(Xs)
            y_pred = pd.Series(proba.argmax(axis=1)).map(inv_label).to_numpy()

        acc = accuracy_score(y_true, y_pred)
        labels = ["CONFIRMED","CANDIDATE","FALSE POSITIVE"]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_table = _cm_to_html(cm, labels)
        report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
        # Show up to 20 mismatches
        miss_mask = (y_true != y_pred)
        misses = df.loc[miss_mask, ["koi_disposition"] + features].copy()
        if len(misses) > 0:
            misses.insert(1, "predicted", pd.Series(y_pred)[miss_mask].values)
            misses_table = misses.head(20).to_html(index=False)
        else:
            misses_table = "<em>No mismatches in this sample.</em>"
                # Optional Light-Curve CNN evaluation
        lc_summary = None
        lc_cm_table = None
        lc_acc = None
        lc_failures_txt = None
        lc_attempted = 0
        lc_success = 0
        lc_failed = 0
        if request.form.get("run_cnn") == "on":
            if not HAS_CNN:
                lc_summary = "CNN model not found (models/cnn.pt). Run `python train_cnn.py` first."
            else:
                lc_n = int(request.form.get("lc_n", 40))
                lc_fast = request.form.get("lc_fast") == "on"
                lc_bin = int(request.form.get("lc_bin", 10))
                lc_len = int(request.form.get("lc_len", 512))
                lc_cache = request.form.get("lc_cache") == "on"
                lc_workers = int(request.form.get("lc_workers", 6))
                try:
                    res = _eval_cnn_on_koi(
                        n=lc_n,
                        seed=seed,
                        fast=lc_fast,
                        bin_minutes=lc_bin,
                        length=lc_len,
                        use_cache=lc_cache,
                        workers=lc_workers
                    )
                    lc_summary = res["report"]
                    lc_cm_table = _cm_to_html(res["cm"], ["CONFIRMED","CANDIDATE","FALSE POSITIVE"])
                    lc_acc = f"{res['accuracy']*100:.2f}"
                    lc_attempted = res["n_attempted"]
                    lc_success = res["n_success"]
                    lc_failed = res["n_failed"]
                    lc_failures_txt = json.dumps(res["failures"], indent=2)
                except Exception as e:
                    lc_summary = f"LC-CNN eval error: {e}"
        return render_template_string(
            HTML_EVAL,
            nav=NAV_HTML,
            error=None,
            summary=report,
            acc=f"{acc*100:.2f}",
            cm_table=cm_table,
            misses_table=misses_table,
            lc_summary=lc_summary,
            lc_cm_table=lc_cm_table,
            lc_acc=lc_acc,
            lc_attempted=lc_attempted,
            lc_success=lc_success,
            lc_failed=lc_failed,
            lc_failures=lc_failures_txt,
        )
    except Exception as e:
        return render_template_string(HTML_EVAL, nav=NAV_HTML, error=f"{e}", summary=None, acc=None, cm_table=None, misses_table=None)


# JSON API for evaluation for scripts
@app.route("/api/eval", methods=["GET"])
def api_eval():
    try:
        n = int(request.args.get("n", 1000))
        seed = int(request.args.get("seed", 42))
        dropna = request.args.get("dropna", "true").lower() in ("1","true","yes","on")
        df = _fetch_koi_dataframe(n=n, seed=seed)
        df = df[df["koi_disposition"].isin(list(inv_label.values()))].copy()
        if dropna:
            df = df.dropna(subset=features)
        for c in features:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if dropna:
            df = df.dropna(subset=features)
        Xs = scaler.transform(df[features].to_numpy())
        y_true = df["koi_disposition"].to_numpy()
        try:
            y_pred_idx = model.predict(Xs)
            if isinstance(y_pred_idx[0], str):
                y_pred = y_pred_idx
            else:
                y_pred = pd.Series(y_pred_idx).map(inv_label).to_numpy()
        except Exception:
            proba = model.predict_proba(Xs)
            y_pred = pd.Series(proba.argmax(axis=1)).map(inv_label).to_numpy()
        acc = accuracy_score(y_true, y_pred)
        labels = ["CONFIRMED","CANDIDATE","FALSE POSITIVE"]
        cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        return jsonify({
            "n": int(len(df)),
            "accuracy": acc,
            "labels": labels,
            "confusion_matrix": cm,
            "classification_report": report
        })
    except Exception as e:
        return jsonify({"error": f"{e}"}), 500
    
@app.route("/api/eval_lc", methods=["GET"])
def api_eval_lc():
    if not HAS_CNN:
        return jsonify({"error": "CNN model not found (models/cnn.pt). Run `python train_cnn.py` first."}), 503
    try:
        n = int(request.args.get("n", 40))
        seed = int(request.args.get("seed", 42))
        fast = request.args.get("fast", "true").lower() in ("1","true","yes","on")
        bin_minutes = int(request.args.get("bin", 10))
        length = int(request.args.get("length", 512))
        use_cache = request.args.get("cache", "true").lower() in ("1","true","yes","on")
        workers = int(request.args.get("workers", 6))
        res = _eval_cnn_on_koi(
            n=n,
            seed=seed,
            fast=fast,
            bin_minutes=bin_minutes,
            length=length,
            use_cache=use_cache,
            workers=workers
        )
        return jsonify({
            "n_attempted": res["n_attempted"],
            "n_success": res["n_success"],
            "n_failed": res["n_failed"],
            "accuracy": res["accuracy"],
            "labels": res["labels"],
            "confusion_matrix": res["cm"].tolist(),
            "classification_report": res["report"],
            "failures": res["failures"],
        })
    except Exception as e:
        return jsonify({"error": f"{e}"}), 500


# Move the main block to the very end of the file, after all route definitions
if __name__ == "__main__":
    # Run on the same host/port you used before
    app.run(host="127.0.0.1", port=7860, debug=False)