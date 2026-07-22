"""
ablation_hyperparams.py

Hyperparameter tuning on the FULL 3-modality model (image + caption + audio).
One-factor-at-a-time sweep over margin, dropout, and batch size, 3 seeds each.

7 unique hyperparameter combinations x 3 seeds = 21 runs.
Same 2,895-artist pool, same train/val/test split as all other experiments.

Submitted via SLURM (see ablation_hyperparams.sbatch).
"""

import os
import sys
import time
import pickle
import random
import logging
import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Configuration
# ============================================================

SCRATCH = Path(os.environ.get("SCRATCH_FLASH", str(Path.home())))
DATA_DIR    = SCRATCH / "thesis" / "data"
OUTPUTS_DIR = SCRATCH / "thesis" / "outputs" / (datetime.date.today().isoformat() + "_hyperparams")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 7]

EPOCHS     = 30
LEARNING_RATE = 0.001
OUTPUT_DIM = 256

# Fixed configuration for all hyperparameter runs: the full 3-modality model
MODALITIES = ("image", "caption", "audio")

THRESHOLDS  = [round(x, 2) for x in np.arange(0.50, 0.91, 0.05)]
TOPK_VALUES = [5, 10, 20]

# One-factor-at-a-time hyperparameter combinations.
# Baseline = margin 0.3, dropout 0.5, batch 32.
HYPERPARAM_CONFIGS = [
    {"id": "margin_0.2",  "margin": 0.2, "dropout": 0.5, "batch_size": 32},
    {"id": "baseline",    "margin": 0.3, "dropout": 0.5, "batch_size": 32},
    {"id": "margin_0.4",  "margin": 0.4, "dropout": 0.5, "batch_size": 32},
    {"id": "dropout_0.3", "margin": 0.3, "dropout": 0.3, "batch_size": 32},
    {"id": "dropout_0.7", "margin": 0.3, "dropout": 0.7, "batch_size": 32},
    {"id": "batch_16",    "margin": 0.3, "dropout": 0.5, "batch_size": 16},
    {"id": "batch_64",    "margin": 0.3, "dropout": 0.5, "batch_size": 64},
]


# ============================================================
# Logging
# ============================================================

LOG_PATH = OUTPUTS_DIR / "run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("hyperparams")


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Data loading (done ONCE)
# ============================================================

def load_data():
    log.info("Loading data from %s", DATA_DIR)

    artists_df = pd.read_csv(DATA_DIR / "artists_mio_3000_updated.csv")
    artists_df = artists_df.drop(columns=[c for c in artists_df.columns if c.startswith("Unnamed")])
    artists_df = artists_df.set_index("musicbrainz_id")

    with open(DATA_DIR / "embeddings_img.pkl", "rb") as f:
        emb_img = pickle.load(f)
    with open(DATA_DIR / "embeddings_captions1.pkl", "rb") as f:
        emb_cap = pickle.load(f)
    with open(DATA_DIR / "embeddings_audio.pkl", "rb") as f:
        emb_aud = pickle.load(f)

    usable_uuids = (
        set(artists_df.index)
        & set(emb_img.keys())
        & set(emb_cap.keys())
        & set(emb_aud.keys())
    )
    usable_list = sorted(usable_uuids)
    usable_set = set(usable_list)
    log.info("Usable artists (all 3 modalities): %d", len(usable_list))

    triplets_df = pd.read_csv(DATA_DIR / "triplets_ids_music_spot_updated.csv")
    mask = (
        triplets_df["anchor"].isin(usable_set)
        & triplets_df["positive"].isin(usable_set)
        & triplets_df["negative"].isin(usable_set)
    )
    triplets_df = triplets_df[mask].reset_index(drop=True)
    triplets_df = triplets_df.sample(frac=1, random_state=42).reset_index(drop=True)

    n_total = len(triplets_df)
    n_train = int(0.70 * n_total)
    n_val   = int(0.20 * n_total)

    train_triplets = list(zip(triplets_df["anchor"][:n_train],
                              triplets_df["positive"][:n_train],
                              triplets_df["negative"][:n_train]))
    val_triplets   = list(zip(triplets_df["anchor"][n_train:n_train+n_val],
                              triplets_df["positive"][n_train:n_train+n_val],
                              triplets_df["negative"][n_train:n_train+n_val]))
    test_triplets  = list(zip(triplets_df["anchor"][n_train+n_val:],
                              triplets_df["positive"][n_train+n_val:],
                              triplets_df["negative"][n_train+n_val:]))
    log.info("Split: train=%d  val=%d  test=%d", len(train_triplets), len(val_triplets), len(test_triplets))

    gt_pairs = defaultdict(set)
    for a, p in zip(triplets_df["anchor"], triplets_df["positive"]):
        gt_pairs[a].add(p)

    log.info("Building fused 3-modality vectors...")
    fused = {}
    for uuid in usable_list:
        img = emb_img[uuid].astype(np.float32).flatten()
        cap = emb_cap[uuid].astype(np.float32).flatten()
        aud = emb_aud[uuid].astype(np.float32).flatten()
        fused[uuid] = np.concatenate([img, cap, aud])
    input_dim = fused[usable_list[0]].shape[0]
    del emb_img, emb_cap, emb_aud
    log.info("Fused input dim: %d", input_dim)

    return {
        "artists_df": artists_df,
        "usable_list": usable_list,
        "fused": fused,
        "input_dim": input_dim,
        "train_triplets": train_triplets,
        "val_triplets": val_triplets,
        "test_triplets": test_triplets,
        "gt_pairs": gt_pairs,
    }


# ============================================================
# Model (dropout is now a parameter)
# ============================================================

class Encoder1DCNN(nn.Module):
    def __init__(self, input_dim, output_dim=256, dropout=0.5):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(), nn.MaxPool1d(2), nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(), nn.MaxPool1d(2), nn.BatchNorm1d(128),
            nn.Conv1d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(), nn.MaxPool1d(2), nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(), nn.MaxPool1d(2), nn.BatchNorm1d(64),
            nn.Conv1d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(), nn.BatchNorm1d(32),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            conv_out = self.conv_blocks(dummy).flatten(1).shape[1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),               # <-- now configurable
            nn.Linear(conv_out, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.head(self.conv_blocks(x.unsqueeze(1)))


class SiameseNet(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, a, p, n):
        return self.encoder(a), self.encoder(p), self.encoder(n)


# ============================================================
# Training / evaluation utilities
# ============================================================

def fetch_batch(triplets, indices, fused, device):
    a = np.stack([fused[triplets[i][0]] for i in indices])
    p = np.stack([fused[triplets[i][1]] for i in indices])
    n = np.stack([fused[triplets[i][2]] for i in indices])
    return (torch.from_numpy(a).to(device, non_blocking=True),
            torch.from_numpy(p).to(device, non_blocking=True),
            torch.from_numpy(n).to(device, non_blocking=True))


def triplet_loss_cosine(a, p, n, margin):
    d_ap = 1 - F.cosine_similarity(a, p)
    d_an = 1 - F.cosine_similarity(a, n)
    return F.relu(d_ap - d_an + margin).mean()


@torch.no_grad()
def eval_triplets(model, triplets, fused, device, margin, batch_size=64):
    model.eval()
    total_loss = 0.0
    correct = 0
    n_total = 0
    d_ap_all, d_an_all = [], []
    for i in range(0, len(triplets), batch_size):
        idx = list(range(i, min(i + batch_size, len(triplets))))
        a, p, n = fetch_batch(triplets, idx, fused, device)
        emb_a, emb_p, emb_n = model(a, p, n)
        loss = triplet_loss_cosine(emb_a, emb_p, emb_n, margin)
        d_ap = 1 - F.cosine_similarity(emb_a, emb_p)
        d_an = 1 - F.cosine_similarity(emb_a, emb_n)
        total_loss += loss.item() * len(idx)
        correct += (d_ap < d_an).sum().item()
        n_total += len(idx)
        d_ap_all.append(d_ap.cpu().numpy())
        d_an_all.append(d_an.cpu().numpy())
    return {
        "loss":      total_loss / n_total,
        "accuracy":  correct / n_total,
        "mean_d_ap": float(np.concatenate(d_ap_all).mean()),
        "mean_d_an": float(np.concatenate(d_an_all).mean()),
    }


def compute_artist_embeddings(model, data, device):
    model.eval()
    emb = {}
    with torch.no_grad():
        for uuid in data["usable_list"]:
            vec = torch.from_numpy(data["fused"][uuid]).unsqueeze(0).to(device)
            emb[uuid] = model.encoder(vec).squeeze(0).cpu().numpy()
    return emb


def topk_retrieval(artist_emb, gt_pairs, usable_list, k_values, sample_anchors=400, seed=42):
    rng = np.random.default_rng(seed)
    uuids = usable_list
    M = np.stack([artist_emb[u] for u in uuids]).astype(np.float32)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    M = M / norms
    uuid_to_row = {u: i for i, u in enumerate(uuids)}

    anchors_with_gt = [a for a in gt_pairs.keys() if a in uuid_to_row and len(gt_pairs[a]) > 0]
    if len(anchors_with_gt) > sample_anchors:
        anchors = list(rng.choice(anchors_with_gt, size=sample_anchors, replace=False))
    else:
        anchors = anchors_with_gt

    hits = {k: 0 for k in k_values}
    totals = {k: 0 for k in k_values}
    max_k = max(k_values)
    for a in anchors:
        a_row = uuid_to_row[a]
        sims = M @ M[a_row]
        sims[a_row] = -1.0
        gt_rows = {uuid_to_row[g] for g in gt_pairs[a] if g in uuid_to_row}
        if not gt_rows:
            continue
        top = np.argpartition(sims, -max_k)[-max_k:]
        top = top[np.argsort(sims[top])[::-1]]
        for k in k_values:
            topk = set(top[:k].tolist())
            hits[k]   += len(topk & gt_rows)
            totals[k] += len(gt_rows)
    return {k: (hits[k] / totals[k] if totals[k] > 0 else 0.0) for k in k_values}


# ============================================================
# Train one (hyperparam config, seed)
# ============================================================

def train_one(hp, seed, data, device):
    hp_id      = hp["id"]
    margin     = hp["margin"]
    dropout    = hp["dropout"]
    batch_size = hp["batch_size"]

    run_dir = OUTPUTS_DIR / hp_id / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log.info("-" * 70)
    log.info("HP=%s (margin=%.1f dropout=%.1f batch=%d) | seed=%d",
             hp_id, margin, dropout, batch_size, seed)
    log.info("-" * 70)

    set_seed(seed)
    encoder = Encoder1DCNN(input_dim=data["input_dim"], output_dim=OUTPUT_DIM, dropout=dropout)
    model = SiameseNet(encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_triplets = data["train_triplets"]
    val_triplets   = data["val_triplets"]
    test_triplets  = data["test_triplets"]
    fused = data["fused"]

    history = []
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_seen = 0
        perm = np.random.permutation(len(train_triplets))
        for i in range(0, len(train_triplets), batch_size):
            idx = perm[i:i + batch_size]
            a, p, n = fetch_batch(train_triplets, idx, fused, device)
            optimizer.zero_grad()
            emb_a, emb_p, emb_n = model(a, p, n)
            loss = triplet_loss_cosine(emb_a, emb_p, emb_n, margin)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
            n_seen += len(idx)
        history.append({"epoch": epoch, "train_loss": epoch_loss / n_seen})
        torch.save({"epoch": epoch, "model_state": model.state_dict()}, run_dir / "checkpoint_last.pth")

    elapsed_min = (time.time() - start) / 60
    test_m = eval_triplets(model, test_triplets, fused, device, margin)
    artist_emb = compute_artist_embeddings(model, data, device)
    topk = topk_retrieval(artist_emb, data["gt_pairs"], data["usable_list"], TOPK_VALUES, seed=seed)

    log.info(
        "%s seed %d | test_acc=%.4f gap=%.4f R@10=%.4f | %.1f min",
        hp_id, seed, test_m["accuracy"],
        test_m["mean_d_an"] - test_m["mean_d_ap"], topk[10], elapsed_min,
    )

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    with open(run_dir / "embeddings_256.pkl", "wb") as f:
        pickle.dump(artist_emb, f)
    torch.save(model.state_dict(), run_dir / "model.pth")

    return {
        "hp_config":  hp_id,
        "margin":     margin,
        "dropout":    dropout,
        "batch_size": batch_size,
        "seed":       seed,
        "minutes":    round(elapsed_min, 1),
        "test_acc":   test_m["accuracy"],
        "test_loss":  test_m["loss"],
        "gap":        test_m["mean_d_an"] - test_m["mean_d_ap"],
        "recall_at_5":  topk[5],
        "recall_at_10": topk[10],
        "recall_at_20": topk[20],
    }


# ============================================================
# Main
# ============================================================

def main():
    log.info("=" * 70)
    log.info("Hyperparameter tuning on FULL 3-modality model | seeds = %s", SEEDS)
    log.info("Output directory: %s", OUTPUTS_DIR)
    log.info("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(0))

    data = load_data()

    all_rows = []
    for hp in HYPERPARAM_CONFIGS:
        for seed in SEEDS:
            try:
                row = train_one(hp, seed, data, device)
                all_rows.append(row)
                pd.DataFrame(all_rows).to_csv(OUTPUTS_DIR / "all_runs.csv", index=False)
            except Exception as e:
                log.exception("FAILED hp=%s seed=%d: %s", hp["id"], seed, e)
                continue

    df = pd.DataFrame(all_rows)
    if not df.empty:
        agg = df.groupby(["hp_config", "margin", "dropout", "batch_size"]).agg(
            test_acc_mean=("test_acc", "mean"),
            test_acc_std=("test_acc", "std"),
            gap_mean=("gap", "mean"),
            recall_at_10_mean=("recall_at_10", "mean"),
            n_seeds=("seed", "count"),
        ).reset_index()
        agg = agg.sort_values("test_acc_mean", ascending=False)
        agg.to_csv(OUTPUTS_DIR / "summary_aggregated.csv", index=False)
        log.info("=" * 70)
        log.info("AGGREGATED SUMMARY (mean +/- std across seeds)")
        log.info("\n%s", agg.to_string(index=False))
        log.info("=" * 70)

    log.info("ALL DONE.")


if __name__ == "__main__":
    main()
