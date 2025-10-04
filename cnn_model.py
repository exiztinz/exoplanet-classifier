import torch
import torch.nn as nn

# ---- defaults used by loader & model metadata ----
DEFAULT_LENGTH = 512
DEFAULT_CLASSES = ("CONFIRMED", "CANDIDATE", "FALSE POSITIVE")

def load_cnn(model_path: str = "models/cnn.pt",
             n_classes: int = 3,
             length: int = DEFAULT_LENGTH,
             map_location: str | None = None) -> nn.Module:
    """
    Loads a CNN checkpoint. If the file is missing or incompatible with the current
    architecture, return a freshly-initialized model instead of raising.
    Requires a build_model(n_classes, length) factory defined in this module.
    """
    device = torch.device(map_location) if map_location else torch.device("cpu")

    def _fresh() -> nn.Module:
        m = build_model(n_classes=n_classes, length=length)
        m.classes = list(DEFAULT_CLASSES)[:n_classes]
        m.length = int(length)
        return m

    # Try reading the checkpoint from disk
    try:
        ckpt = torch.load(model_path, map_location=device)
    except FileNotFoundError:
        print(f"[cnn_model] No checkpoint at '{model_path}'. Using fresh model.")
        return _fresh()
    except Exception as e:
        print(f"[cnn_model] Failed to read checkpoint '{model_path}': {e}. Using fresh model.")
        return _fresh()

    classes = list(DEFAULT_CLASSES)[:n_classes]
    L = int(length)

    # Case 1: training script saved a dict with metadata
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        if "classes" in ckpt and isinstance(ckpt["classes"], (list, tuple)) and len(ckpt["classes"]) >= 1:
            classes = list(ckpt["classes"])[:n_classes]
        if "length" in ckpt:
            try:
                L = int(ckpt["length"])
            except Exception:
                pass
        model = build_model(n_classes=len(classes), length=L)
        try:
            model.load_state_dict(ckpt["state_dict"], strict=True)
        except Exception as e:
            print(f"[cnn_model] Incompatible checkpoint for current architecture: {e}. Using fresh model.")
            return _fresh()
        model.classes = classes
        model.length = L
        return model

    # Case 2: a bare state_dict (older code path)
    model = build_model(n_classes=n_classes, length=L)
    try:
        model.load_state_dict(ckpt, strict=True)
    except Exception as e:
        print(f"[cnn_model] Incompatible bare state_dict: {e}. Using fresh model.")
        return _fresh()
    model.classes = classes
    model.length = L
    return model

# -------------------------------
# Minimal LC CNN and factory
# -------------------------------
class LCBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 9, s: int = 1, p: int | None = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=k, stride=s, padding=p)
        self.bn = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class LCNet(nn.Module):
    def __init__(self, n_classes: int = 3, length: int = DEFAULT_LENGTH):
        super().__init__()
        self.length = int(length)
        self.backbone = nn.Sequential(
            LCBlock(1, 16, 9, 1),
            LCBlock(16, 32, 9, 2),
            LCBlock(32, 64, 9, 2),
            LCBlock(64, 128, 9, 2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.length)
            feat = self.backbone(dummy)
            feat_dim = feat.shape[1] * feat.shape[2]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )
        self.classes = list(DEFAULT_CLASSES)[:n_classes]
    def forward(self, x):
        return self.head(self.backbone(x))

def build_model(n_classes: int = 3, length: int = DEFAULT_LENGTH) -> nn.Module:
    """Factory used by loaders/trainers to construct the CNN."""
    return LCNet(n_classes=n_classes, length=length)