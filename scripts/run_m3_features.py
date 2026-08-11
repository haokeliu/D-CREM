#!/usr/bin/env python3
"""M3: Feature quality ceiling test — extract & cache image features.

Downloads VOC2007 / MS-COCO via torchvision, extracts three feature types:
  ① PCA-reduced raw features (hand-crafted baseline, no GPU needed)
  ② Frozen ResNet-50 (ImageNet pre-trained, 2048-d)
  ③ Frozen CLIP ViT-B/32 (512-d)

Dual-mode design:
  - CPU: PCA features and CREM runs need no GPU
  - GPU: ResNet-50 / CLIP extraction is much faster
    The script auto-detects CUDA; use --device cuda to force GPU.

Saves CREM-compatible .mat files to cache/{dataset}_{feature_type}.mat and
runs CREM experiments without modifying protocol documentation.

Usage:
  # Step 1: extract features (GPU recommended for ResNet/CLIP)
  python scripts/run_m3_features.py --extract --device cuda --batch-size 128

  # Step 2: run CREM experiments (CPU only, no GPU needed)
  python scripts/run_m3_features.py --run --seeds 10

  # One-shot:
  python scripts/run_m3_features.py --all --device cuda
"""

import argparse
import json
import os
import ssl
import sys
import time
from datetime import datetime

# ── SSL workaround for Windows certificate store corruption ──
# Python 3.9 on some Windows 11 machines fails to load the Windows cert
# store (ASN1: NOT_ENOUGH_DATA).  Fall back to certifi or unverified context
# so torchvision can download VOC2007/MS-COCO.
try:
    ssl._create_default_https_context()
except ssl.SSLError:
    try:
        import certifi
        ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
from scipy.io import loadmat, savemat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
IMAGE_CACHE = os.path.join(CACHE_DIR, "images")

# ── Config per dataset ──────────────────────────────────────────────────
DATASET_CONFIG = {
    "voc2007": {
        "num_classes": 20,
        "num_samples": 5011,       # trainval set
    },
    "coco2014": {
        "num_classes": 80,
        "num_samples": 40504,      # val2014 set
        "max_samples": 5000,       # subset for speed (full 40k is slow on CREM)
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# GPU / device utilities
# ═══════════════════════════════════════════════════════════════════════════

def get_device(requested=None):
    """Resolve torch device.  Respects --device flag, falls back to auto-detect."""
    import torch
    if requested:
        if requested == "cuda" and not torch.cuda.is_available():
            print("  WARNING: --device cuda but CUDA not available, falling back to cpu")
            return torch.device("cpu")
        return torch.device(requested)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  Auto-detected GPU: {torch.cuda.get_device_name(0)}")
        return device
    print("  Auto-detected: CPU (no CUDA)")
    return torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════════════
# Feature extractors
# ═══════════════════════════════════════════════════════════════════════════

def get_resnet50_features(images_batch, device, batch_size=128):
    """Extract ResNet-50 global avg pool features (2048-d)."""
    import torch
    import torchvision.models as models
    from torchvision import transforms

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    model = torch.nn.Sequential(*list(model.children())[:-1])  # strip fc
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    features = []
    n = len(images_batch)
    for i in range(0, n, batch_size):
        batch = images_batch[i:i + batch_size]
        tensors = torch.stack([transform(img) for img in batch]).to(device)
        with torch.no_grad():
            feats = model(tensors).squeeze(-1).squeeze(-1)
        features.append(feats.cpu().numpy())
        if (i // batch_size + 1) % 10 == 0:
            print(f"    ResNet-50: {min(i + batch_size, n)}/{n}")

    return np.concatenate(features, axis=0).astype(np.float64)


def get_clip_features(images_batch, device, batch_size=128):
    """Extract CLIP ViT-B/32 image features (512-d, l2-normalised)."""
    import torch
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k")
    model.eval()
    model.to(device)

    features = []
    n = len(images_batch)
    for i in range(0, n, batch_size):
        batch = images_batch[i:i + batch_size]
        tensors = torch.stack([preprocess(img) for img in batch]).to(device)
        with torch.no_grad():
            feats = model.encode_image(tensors)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        features.append(feats.cpu().numpy())
        if (i // batch_size + 1) % 10 == 0:
            print(f"    CLIP: {min(i + batch_size, n)}/{n}")

    return np.concatenate(features, axis=0).astype(np.float64)


def get_pca_features(images_batch, target_dim=512):
    """PCA-reduced raw pixels (hand-crafted baseline).  No GPU required.

    Resize to 64×64, flatten, standardise, PCA to target_dim.
    """
    from sklearn.decomposition import PCA
    from PIL import Image

    flat_features = []
    for img in images_batch:
        img_resized = img.resize((64, 64), Image.LANCZOS)
        arr = np.array(img_resized, dtype=np.float64).ravel()
        if len(arr) == 64 * 64:           # grayscale → repeat channels
            arr = np.repeat(arr, 3)
        flat_features.append(arr)

    X_raw = np.stack(flat_features, axis=0)
    mean = X_raw.mean(axis=0, keepdims=True)
    std = X_raw.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    X_raw = (X_raw - mean) / std

    k = min(target_dim, X_raw.shape[0], X_raw.shape[1])
    pca = PCA(n_components=k)
    X_pca = pca.fit_transform(X_raw)
    print(f"    PCA: {X_raw.shape[1]} → {k} "
          f"(explained var: {pca.explained_variance_ratio_.sum():.3f})")

    return X_pca.astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# Dataset loaders
# ═══════════════════════════════════════════════════════════════════════════

def load_voc2007(data_root, max_samples=None):
    """Load VOC2007 multi-label dataset (trainval, 20 classes)."""
    import torchvision.datasets as ds

    dataset = ds.VOCDetection(
        root=data_root, year="2007", image_set="trainval", download=True)

    LABEL_NAMES = [
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
        "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
        "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
    ]
    name_to_idx = {n: i for i, n in enumerate(LABEL_NAMES)}

    images, labels = [], []
    for i, (img, ann) in enumerate(dataset):
        if max_samples and i >= max_samples:
            break
        images.append(img.convert("RGB"))
        vec = np.zeros(len(LABEL_NAMES), dtype=np.float64)
        for obj in ann["annotation"]["object"]:
            nm = obj["name"]
            if nm in name_to_idx:
                vec[name_to_idx[nm]] = 1.0
        labels.append(vec)
        if (i + 1) % 1000 == 0:
            print(f"    VOC2007: {i + 1} images...")

    print(f"  VOC2007: {len(images)} images, {len(LABEL_NAMES)} classes")
    return images, np.stack(labels, axis=0), LABEL_NAMES


def load_coco2014(data_root, max_samples=5000):
    """Load MS-COCO 2014 multi-label dataset (val, 80 classes)."""
    import torchvision.datasets as ds

    dataset = ds.CocoDetection(
        root=os.path.join(data_root, "val2014"),
        annFile=os.path.join(data_root, "annotations/instances_val2014.json"),
    )

    coco = dataset.coco
    cat_ids = sorted(coco.getCatIds())
    label_names = [coco.loadCats([c])[0]["name"] for c in cat_ids]
    id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}

    images, labels = [], []
    for i, (img, anns) in enumerate(dataset):
        if max_samples and i >= max_samples:
            break
        images.append(img.convert("RGB"))
        vec = np.zeros(len(cat_ids), dtype=np.float64)
        for ann in anns:
            cid = ann["category_id"]
            if cid in id_to_idx:
                vec[id_to_idx[cid]] = 1.0
        labels.append(vec)
        if (i + 1) % 1000 == 0:
            print(f"    COCO: {i + 1} images...")

    print(f"  COCO2014: {len(images)} images, {len(cat_ids)} classes")
    return images, np.stack(labels, axis=0), label_names


# ═══════════════════════════════════════════════════════════════════════════
# Feature extraction pipeline
# ═══════════════════════════════════════════════════════════════════════════

def extract_and_save(dataset_name, feature_type, device=None,
                     batch_size=128, data_root=None):
    """Extract features and save as .mat for CREM (idempotent — skips if cached)."""
    if data_root is None:
        data_root = os.path.join(IMAGE_CACHE, "raw")

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(data_root, exist_ok=True)

    cache_path = os.path.join(CACHE_DIR, f"{dataset_name}_{feature_type}.mat")
    if os.path.exists(cache_path):
        print(f"  [skip] Cache exists: {cache_path}")
        return cache_path

    print(f"\n{'='*60}")
    print(f"Extracting: {dataset_name} / {feature_type}")
    print(f"  Device: {device}, Batch size: {batch_size}")
    print(f"{'='*60}")

    # ── Load ──
    t0 = time.time()
    cfg = DATASET_CONFIG[dataset_name]
    max_n = cfg.get("max_samples", cfg["num_samples"])

    if dataset_name == "voc2007":
        images, Y, label_names = load_voc2007(data_root, max_samples=max_n)
    elif dataset_name == "coco2014":
        images, Y, label_names = load_coco2014(data_root, max_samples=max_n)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"  Load time: {time.time() - t0:.1f}s")

    # ── Filter (keep images with ≥ 1 label) ──
    keep = Y.sum(axis=1) > 0
    images = [img for i, img in enumerate(images) if keep[i]]
    Y = Y[keep, :]
    print(f"  After filtering (≥1 label): {len(images)} images")

    # ── Extract ──
    t0 = time.time()
    if feature_type == "pca":
        X = get_pca_features(images, target_dim=512)
    elif feature_type == "resnet50":
        X = get_resnet50_features(images, device=device, batch_size=batch_size)
    elif feature_type == "clip":
        X = get_clip_features(images, device=device, batch_size=batch_size)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    print(f"  Feature shape: {X.shape}")
    print(f"  Extract time: {time.time() - t0:.1f}s")

    # ── Save ──
    savemat(cache_path, {
        "X": X.astype(np.float64),
        "Y": Y.astype(np.float64),
        "label_names": np.array(label_names, dtype=object),
    }, do_compression=True)

    size_mb = os.path.getsize(cache_path) / 1024 / 1024
    print(f"  Saved: {cache_path} ({size_mb:.1f} MB)")
    return cache_path


# ═══════════════════════════════════════════════════════════════════════════
# CREM experiment runner for M3
# ═══════════════════════════════════════════════════════════════════════════

def run_crem_on_features(dataset_name, feature_type, known_ratio=0.5,
                         seeds=10, method="crem_v2"):
    """Run CREM on pre-extracted features (CPU-only — no GPU needed)."""
    from crem import crem_train, crem_validate_and_test, kernelization
    from crem.config import get_params
    from crem.data import apply_crem_split

    cache_path = os.path.join(CACHE_DIR, f"{dataset_name}_{feature_type}.mat")
    if not os.path.exists(cache_path):
        print(f"  ERROR: Feature cache not found: {cache_path}")
        print(f"  Run with --extract first.")
        return

    mat = loadmat(cache_path)
    X = mat["X"].astype(np.float64)
    Y = mat["Y"].astype(np.float64)
    raw_names = mat.get("label_names", [[]])
    label_names = [
        str(x[0]) if (hasattr(x, 'size') and x.size > 0) else str(x)
        for x in raw_names.ravel()
    ] if len(raw_names) > 0 else [f"L{i}" for i in range(Y.shape[1])]

    # Use default params for image features (no per-dataset tuning yet)
    nominal, effective = get_params(None, {
        "lamda1": 1, "lamda2": 0.1, "lamda3": 10, "alpha": 1, "gamma": 0.05,
    })

    for seed in range(seeds):
        data = apply_crem_split(X, Y, label_names, known_ratio, seed,
                                standardize=False)

        train_Kernel = kernelization(data["train_data"], data["train_data"],
                                     "RBF", (effective["gamma"],))
        val_Kernel = kernelization(data["val_data"], data["train_data"],
                                   "RBF", (effective["gamma"],))
        test_Kernel = kernelization(data["test_data"], data["train_data"],
                                    "RBF", (effective["gamma"],))

        t0 = time.time()
        model = crem_train(data["train_target"], train_Kernel, nominal,
                           verbose=False)
        t_train = time.time() - t0

        t0 = time.time()
        result, selection = crem_validate_and_test(
            data["train_target"], data["val_target"], data["test_target"],
            data["val_osr_labels"], data["osr_labels"], model,
            train_Kernel, val_Kernel, test_Kernel, nominal, verbose=False)
        t_test = time.time() - t0

        # ── Save with feature_type in setting path ──
        setting = f"protocol_v2_m3_{feature_type}_known_ratio={known_ratio}"
        run_dir = os.path.join(BASE_DIR, "results", method, dataset_name, setting)
        os.makedirs(run_dir, exist_ok=True)

        record = {
            "metrics": {
                "AUROC": float(result["AUROC"]),
                "AUPR": float(result["AUPR"]),
                "macroAUC": float(result["macroAUC"]),
                "AveragePrecision": float(result["AveragePrecision"]),
                "RankingLoss": float(result["RankingLoss"]),
                "Coverage": float(result["Coverage"]),
                "OneError": float(result["OneError"]),
            },
            "config": {"nominal": nominal, "effective": effective},
            "protocol": {
                "version": 2, "train_ratio": 0.4, "val_ratio": 0.1,
                "test_ratio": 0.5, "preprocessing_fit": "train_only",
                "standardized": False, "selection": selection,
            },
            "time": {
                "train_s": round(t_train, 3),
                "test_s": round(t_test, 3),
                "total_s": round(t_train + t_test, 3),
            },
            "timestamp": datetime.now().isoformat(),
            "known_ratio": known_ratio,
            "seed": seed,
            "dataset": dataset_name,
            "method": method,
            "feature_type": feature_type,
        }
        with open(os.path.join(run_dir, f"seed{seed}.json"), "w") as f:
            json.dump(record, f, indent=2, default=str)

        print(f"  {feature_type} seed={seed}: AUROC={result['AUROC']:.4f}  "
              f"AUPR={result['AUPR']:.4f}  macroAUC={result['macroAUC']:.4f}")


def build_m3_table(datasets, feature_types, known_ratio=0.5, method="crem_v2"):
    """Generate Table_M3 markdown from saved results."""
    from crem.logger import mean_std

    lines = [
        f"## Table_M3: Feature quality comparison (known_ratio={known_ratio}, mean ± std)",
        "",
        "| Dataset | Feature | AUROC | AUPR | macroAUC | AvgPrec |",
        "|---|---|---|---|---|---|",
    ]

    for ds in datasets:
        for ft in feature_types:
            setting = f"protocol_v2_m3_{ft}_known_ratio={known_ratio}"
            run_dir = os.path.join(BASE_DIR, "results", method, ds, setting)
            if not os.path.isdir(run_dir):
                lines.append(f"| {ds} | {ft} | N/A | N/A | N/A | N/A |")
                continue

            runs = []
            for fn in sorted(os.listdir(run_dir)):
                if fn.startswith("seed") and fn.endswith(".json"):
                    with open(os.path.join(run_dir, fn)) as f:
                        runs.append(json.load(f))

            if not runs:
                lines.append(f"| {ds} | {ft} | N/A | N/A | N/A | N/A |")
                continue

            auroc_m, auroc_s = mean_std(runs, "AUROC")
            aupr_m, aupr_s = mean_std(runs, "AUPR")
            mauc_m, mauc_s = mean_std(runs, "macroAUC")
            ap_m, ap_s = mean_std(runs, "AveragePrecision")

            lines.append(
                f"| {ds} | {ft} | "
                f"{auroc_m:.4f} ± {auroc_s:.4f} | "
                f"{aupr_m:.4f} ± {aupr_s:.4f} | "
                f"{mauc_m:.4f} ± {mauc_s:.4f} | "
                f"{ap_m:.4f} ± {ap_s:.4f} |")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="M3: Feature quality ceiling test (CPU/GPU dual-mode)")

    # Actions
    parser.add_argument("--extract", action="store_true",
                        help="Download datasets and extract features")
    parser.add_argument("--run", action="store_true",
                        help="Run CREM on extracted features (CPU)")
    parser.add_argument("--all", action="store_true",
                        help="Extract + run")

    # Data selection
    parser.add_argument("--dataset", type=str, default="voc2007",
                        choices=["voc2007", "coco2014", "all"],
                        help="Dataset (default: voc2007)")
    parser.add_argument("--feature", type=str, default="all",
                        choices=["pca", "resnet50", "clip", "all"],
                        help="Feature type (default: all)")

    # GPU / performance
    parser.add_argument("--device", type=str, default=None,
                        choices=["cpu", "cuda"],
                        help="Torch device (default: auto-detect)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for feature extraction (default: 128)")

    # CREM settings
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of seeds (default: 10)")
    parser.add_argument("--known-ratio", type=float, default=0.5,
                        help="Known label ratio (default: 0.5)")

    args = parser.parse_args()

    feature_types = (["pca", "resnet50", "clip"] if args.feature == "all"
                     else [args.feature])
    datasets = (["voc2007", "coco2014"] if args.dataset == "all"
                else [args.dataset])

    do_extract = args.extract or args.all
    do_run = args.run or args.all

    # ── Resolve device once ──
    device = None
    if do_extract:
        device = get_device(args.device)

    # ── Extract ──
    if do_extract:
        for ds in datasets:
            for ft in feature_types:
                try:
                    extract_and_save(ds, ft, device=device,
                                     batch_size=args.batch_size)
                except Exception as e:
                    print(f"  ERROR ({ds}/{ft}): {e}")
                    import traceback
                    traceback.print_exc()

    # ── Run CREM ──
    if do_run:
        for ds in datasets:
            print(f"\n{'='*60}")
            print(f"Running CREM on: {ds}")
            print(f"{'='*60}")
            for ft in feature_types:
                print(f"\n  --- {ft} ---")
                try:
                    run_crem_on_features(ds, ft,
                                         known_ratio=args.known_ratio,
                                         seeds=args.seeds)
                except Exception as e:
                    print(f"  ERROR ({ds}/{ft}): {e}")
                    import traceback
                    traceback.print_exc()

        from scripts.build_results_report import write_report
        print(f"\nProtocol-v2 summary updated: {write_report()}")


if __name__ == "__main__":
    main()
