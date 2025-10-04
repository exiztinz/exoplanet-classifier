# train_baseline.py
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

from data_ingest import load_koi, LABEL_COL

# Map labels to integers (consistent, explicit)
LABEL_MAP = {"CONFIRMED": 0, "CANDIDATE": 1, "FALSE POSITIVE": 2}
INV_LABEL = {v: k for k, v in LABEL_MAP.items()}

def main():
    # --- Load data ---
    df, feats = load_koi()
    X = df[feats].to_numpy(dtype=np.float32)
    y = df[LABEL_COL].map(LABEL_MAP).to_numpy()

    print("Rows:", X.shape[0], "| Features:", feats)
    print("Class counts:", df[LABEL_COL].value_counts().to_dict())

    # --- Scale numeric features (helps some learners; harmless for trees post-calibration) ---
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # --- Base model ---
    gbm = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        learning_rate=0.05,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
    )

    # --- Quick CV fit for a sanity check, then full-fit with probability calibration ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    for tr, va in skf.split(Xs, y):
        gbm.fit(Xs[tr], y[tr])
        pred = gbm.predict(Xs[va])
        print(f"\n[Fold {fold}]")
        print(classification_report(y[va], pred, target_names=[INV_LABEL[i] for i in range(3)], digits=4))
        fold += 1

    # --- Calibrate on all data (for nicer, thresholdable probabilities) ---
    calib = CalibratedClassifierCV(gbm, method="isotonic", cv=3)
    calib.fit(Xs, y)

    # --- Final fit report on the same data (just to sanity-check; not a test metric) ---
    pred_all = calib.predict(Xs)
    print("\n[All data (post-calibration, not a test metric)]")
    print(classification_report(y, pred_all, target_names=[INV_LABEL[i] for i in range(3)], digits=4))
    print("Confusion matrix (rows=true, cols=pred):\n", confusion_matrix(y, pred_all))

    # --- Persist model & schema ---
    os.makedirs("models", exist_ok=True)
    artifact = {
        "scaler": scaler,
        "model": calib,
        "features": feats,
        "label_map": LABEL_MAP,
        "schema_note": "Expect a CSV with these numeric feature columns; predict_proba gives P per class."
    }
    joblib.dump(artifact, "models/gbm_koi.joblib")
    pd.Series(feats).to_csv("models/feature_list.csv", index=False, header=False)
    print("\nSaved model -> models/gbm_koi.joblib")
    print("Saved feature list -> models/feature_list.csv")

if __name__ == "__main__":
    main()