from __future__ import annotations
# train_cnn.py
# Train a larger LC CNN on folded light curves using the existing fold cache for speed.

import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter

from cnn_model import build_model, DEFAULT_CLASSES, DEFAULT_LENGTH
from serve import _fold_cached  # reuse your folding (and its cache)
from tqdm.auto import tqdm
import os, time

# Make network calls less sticky and show immediate startup output
print("[train_cnn] Starting… (configuring astroquery timeouts/retries)")
try:
    from astroquery import conf as astro_conf
    astro_conf.timeout = 20  # seconds per request
    astro_conf.max_retries = 2
except Exception:
    pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = Path("models"); MODELS_DIR.mkdir(parents=True, exist_ok=True)


def fetch_manifest(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    print(f"[train_cnn] Fetching manifest (top {n}) from NASA Exoplanet Archive…")
    url = (f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
           f"query=select+top+{n}+kepid,koi_period,koi_disposition+from+cumulative&format=csv")
    df = pd.read_csv(url)
    print(f"[train_cnn] Manifest fetched: {len(df)} rows before dropna")
    df = df.dropna(subset=["kepid","koi_period","koi_disposition"]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(f"[train_cnn] Manifest ready: {len(df)} usable rows")
    return df


def preprocess_flux(x: np.ndarray) -> np.ndarray:
    # mirror inference: median-normalize -> center deepest dip -> 1-x -> robust standardize -> clip
    x = x.astype(np.float32)
    med = np.nanmedian(x)
    if np.isfinite(med) and med != 0.0:
        x = x / med
    try:
        k = int(np.nanargmin(x))
        x = np.roll(x, len(x)//2 - k)
    except Exception:
        pass
    x = 1.0 - x
    m = np.nanmedian(x); mad = np.nanmedian(np.abs(x - m))
    scale = mad if (np.isfinite(mad) and mad > 0) else (np.nanstd(x) + 1e-6)
    x = (x - m) / (scale + 1e-6)
    x = np.clip(x, -5.0, 10.0)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def augment_flux(x: np.ndarray, p_shift: float = 0.5, p_noise: float = 0.5) -> np.ndarray:
    L = len(x)
    y = x.copy()
    if random.random() < p_shift:
        y = np.roll(y, random.randint(-L//16, L//16))
    if random.random() < p_noise:
        y = y + np.random.normal(0, 0.05, size=L).astype(np.float32)
    return y


class FoldedLCDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, length: int = DEFAULT_LENGTH, train: bool = True, verbose: bool = True):
        self.items = []
        self.length = int(length)
        self.train = train
        self.verbose = verbose
        n_total = 0
        n_ok = 0
        for _, r in manifest.iterrows():
            if self.verbose and (n_total % 10 == 0):
                print(f"[dataset] building… {n_total} tried / {n_ok} ok so far")
            n_total += 1
            kic = int(r.kepid); per = float(r.koi_period); disp = str(r.koi_disposition)
            target = f"KIC {kic}"
            try:
                flux = _fold_cached(
                    "Kepler", target, per,
                    t0=None,
                    length=self.length,
                    fast=True,
                    bin_minutes=10
                )
                x = preprocess_flux(flux)
                if not np.all(np.isfinite(x)):
                    raise ValueError("non-finite flux after preprocess")
                self.items.append((x, disp))
                n_ok += 1
            except Exception as e:
                if self.verbose and (n_total <= 10 or n_total % 25 == 0):
                    print(f"[dataset] skip {target} (P={per}d): {type(e).__name__}: {e}")
                continue
        if self.verbose:
            print(f"[dataset] prepared {n_ok}/{n_total} samples (length={self.length})")
        self.classes = list(DEFAULT_CLASSES)
        self.cls2idx = {c:i for i,c in enumerate(self.classes)}

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        x, label = self.items[idx]
        if self.train:
            x = augment_flux(x)
        t = torch.from_numpy(x).float().unsqueeze(0)
        y = self.cls2idx.get(label, 1)  # default 'CANDIDATE' if unseen
        return t, y


def train(n: int = 3000, seed: int = 42, length: int = DEFAULT_LENGTH, batch_size: int = 64, epochs: int = 15, lr: float = 1e-3, p_drop: float = 0.1, resume_path: str = ""):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    manifest = fetch_manifest(n=n, seed=seed)
    mtrain, mval = train_test_split(manifest, test_size=0.2, random_state=seed, stratify=manifest["koi_disposition"])
    print(f"[train_cnn] Splitting manifest: train/val = {len(mtrain)}/{len(mval)}")
    print(f"[train_cnn] Building training dataset (length={length})… this may download/fold light curves")
    ds_train = FoldedLCDataset(mtrain, length=length, train=True, verbose=True)
    print("[train_cnn] Building validation dataset…")
    ds_val = FoldedLCDataset(mval, length=length, train=False, verbose=True)

    if len(ds_train) == 0 or len(ds_val) == 0:
        raise RuntimeError("No training/validation samples were prepared. Warm the fold cache first or reduce --n. Try: curl -s \"http://127.0.0.1:7860/api/eval_lc?n=120&seed=42&fast=true&bin=10&length=512&cache=true&workers=12\" > /dev/null, then re-run training.")

    counts = Counter([lab for _, lab in ds_train.items])
    weights = torch.tensor([1.0 / max(1, counts.get(c, 1)) for c in DEFAULT_CLASSES], dtype=torch.float32, device=DEVICE)
    weights = weights / weights.mean()

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    print(f"[train_cnn] Dataset sizes: train={len(ds_train)} val={len(ds_val)} | batch_size={batch_size}")

    model = build_model(n_classes=len(DEFAULT_CLASSES), length=length).to(DEVICE)
    model.classes = list(DEFAULT_CLASSES)
    model.length = int(length)

    # Resume from a previous checkpoint if provided
    if resume_path:
        try:
            ckpt = torch.load(resume_path, map_location=DEVICE)
            state = ckpt.get("state_dict", ckpt)
            model.load_state_dict(state, strict=True)
            # Keep classes/length from checkpoint if present
            if isinstance(ckpt, dict):
                if "classes" in ckpt and isinstance(ckpt["classes"], (list, tuple)):
                    model.classes = list(ckpt["classes"])[:len(DEFAULT_CLASSES)]
                if "length" in ckpt:
                    try:
                        model.length = int(ckpt["length"])
                    except Exception:
                        pass
            print(f"[train_cnn] Resumed weights from {resume_path}")
        except Exception as e:
            print(f"[train_cnn] Could not resume from {resume_path}: {e}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=weights)

    print("Starting training...")

    best_val = float("inf")
    best_state = None
    for epoch in range(1, epochs+1):
        model.train()
        loss_sum = 0.0
        # persistent progress bar for this epoch
        pbar = tqdm(
            total=len(dl_train),
            desc=f"Epoch {epoch}/{epochs}",
            ncols=120,  # make the bar wide
            leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        )
        for step, (x, y) in enumerate(dl_train):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if step % 20 == 0:
                tqdm.write(f"Epoch {epoch}/{epochs}, Step {step}, Loss={loss.item():.5f}")
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.item()) * x.size(0)

            # update progress bar and show live loss
            pbar.set_postfix(loss=f"{loss.item():.5f}")
            pbar.update(1)
        pbar.close()
        print(f"Finished epoch {epoch}/{epochs}")

        sched.step()

        model.eval()
        vloss = 0.0
        n_correct = 0
        n_total = 0
        with torch.no_grad():
            for x, y in dl_val:
                x = x.to(DEVICE); y = y.to(DEVICE)
                logits = model(x)
                loss = criterion(logits, y)
                vloss += float(loss.item()) * x.size(0)
                pred = torch.argmax(logits, dim=1)
                n_correct += int((pred == y).sum().item())
                n_total += int(y.numel())
        vloss /= max(1, len(ds_val))
        acc = (n_correct / n_total) if n_total else 0.0
        print(f"Epoch {epoch:02d}: val_loss={vloss:.4f} val_acc={acc:.4f}")

        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    assert best_state is not None, "No best state captured"
    print("Training complete. Saving model...")

    # Save a timestamped archive for this run and include val_loss for cross-run comparison
    ckpt = {
        "state_dict": best_state,
        "classes": model.classes,
        "length": model.length,
        "val_loss": float(best_val),
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_ts = MODELS_DIR / f"cnn_{ts}.pt"
    torch.save(ckpt, out_ts)
    print("Saved", out_ts)

    # Update cnn_best.pt only if this run's validation loss improved
    best_path = MODELS_DIR / "cnn_best.pt"
    update_best = True
    prev_loss = float("inf")
    if best_path.exists():
        try:
            prev = torch.load(best_path, map_location="cpu")
            prev_loss = float(prev.get("val_loss", float("inf")))
            update_best = best_val < prev_loss
        except Exception as e:
            print("Warning: could not read existing cnn_best.pt:", e)
            update_best = True

    if update_best:
        try:
            torch.save(ckpt, best_path)
            print(f"Updated {best_path} (val_loss={best_val:.4f})")
        except Exception as e:
            print("Could not update cnn_best.pt:", e)

        # Keep the serving alias in sync with the new best
        try:
            torch.save(ckpt, MODELS_DIR / "cnn.pt")
            print("Updated models/cnn.pt")
        except Exception as e:
            print("Could not update cnn.pt:", e)
    else:
        print(f"Kept existing {best_path} (prev={prev_loss:.4f} <= new={best_val:.4f})")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--resume", type=str, default="", help="Path to a previous checkpoint (.pt) to resume from")
    args = ap.parse_args()
    train(n=args.n, seed=args.seed, length=args.length, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, p_drop=args.dropout, resume_path=args.resume)