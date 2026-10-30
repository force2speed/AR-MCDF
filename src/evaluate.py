"""
AR-MCDF  |  Model Evaluation Script
=====================================
Loads saved ST-GCN weights and evaluates on the UNSW-NB15 test set.
Outputs: Accuracy, Precision, Recall, F1, Confusion Matrix.
Saves:   data/evaluation_report.json

Run from project root:
    py src/evaluate.py
"""

from __future__ import annotations

import json
import sys
import os
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_SRC  = Path(__file__).resolve().parent
_ROOT = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
os.chdir(_ROOT)

from module1_ingestion import run_ingestion, NUMERICAL_FEATURES
from module2_sanitizer import DummyProbe, FGSMAttacker
from module3_stgcn import SpatioTemporalGCN, WindowGraphBuilder

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ---- Load saved model ------------------------------------------------
    weights_path = Path("models/stgcn_weights.pt")
    if not weights_path.exists():
        logger.error("No trained weights found at %s. Run module5_pipeline.py first.", weights_path)
        sys.exit(1)

    model = SpatioTemporalGCN(
        n_features=len(NUMERICAL_FEATURES), gcn_hidden=128, attn_heads=4,
        fc_hidden=64, n_classes=2, dropout=0.0, pool="both",
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    logger.info("Loaded ST-GCN weights from %s", weights_path)

    # ---- Load test set ---------------------------------------------------
    logger.info("Loading test set...")
    windows, labels, _ = run_ingestion(
        csv_path="data/raw/UNSW_NB15_testing-set.csv",
        window_size=50, stride=25,
    )
    logger.info("Test set: %d windows", len(windows))

    # ---- Inference -------------------------------------------------------
    graph_builder = WindowGraphBuilder(method="cosine", threshold=0.60, k=4)
    attacker      = FGSMAttacker(model=DummyProbe(window_size=50, n_features=len(NUMERICAL_FEATURES)), epsilon=0.10)

    y_true, y_pred_clean, y_pred_adv = [], [], []

    logger.info("Running inference...")
    for i, (win_np, label_np) in enumerate(zip(windows, labels)):
        true_label = int(label_np.mean() >= 0.5)
        y_true.append(true_label)

        # Clean inference
        x_clean = torch.tensor(win_np, dtype=torch.float32)
        g_clean = graph_builder.build(x_clean, label=true_label).to(device)
        with torch.no_grad():
            probs = F.softmax(model(g_clean), dim=-1)
        y_pred_clean.append(int(probs.argmax(dim=-1).item()))

        # Adversarial inference (what attacker sees without defense)
        # We need gradients for the FGSM attack
        x_adv  = attacker.attack(x_clean, torch.tensor([true_label], dtype=torch.long))
        g_adv  = graph_builder.build(x_adv, label=true_label).to(device)
        with torch.no_grad():
            probs_adv = F.softmax(model(g_adv), dim=-1)
        y_pred_adv.append(int(probs_adv.argmax(dim=-1).item()))

        if (i + 1) % 1000 == 0:
            logger.info("  Processed %d / %d windows", i + 1, len(windows))

    # ---- Metrics ---------------------------------------------------------
    def compute_metrics(y_t, y_p, label=""):
        y_t, y_p = np.array(y_t), np.array(y_p)
        acc = float((y_t == y_p).mean())

        tp = int(((y_p == 1) & (y_t == 1)).sum())
        fp = int(((y_p == 1) & (y_t == 0)).sum())
        fn = int(((y_p == 0) & (y_t == 1)).sum())
        tn = int(((y_p == 0) & (y_t == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"\n  [{label}]")
        print(f"  Accuracy  : {acc*100:.2f}%")
        print(f"  Precision : {precision:.4f}")
        print(f"  Recall    : {recall:.4f}")
        print(f"  F1 Score  : {f1:.4f}")
        print(f"  Confusion : TP={tp} FP={fp} FN={fn} TN={tn}")

        return {"accuracy": round(acc, 4), "precision": round(precision, 4),
                "recall": round(recall, 4), "f1": round(f1, 4),
                "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}}

    print("\n" + "=" * 60)
    print("  AR-MCDF  |  Evaluation Report")
    print("=" * 60)

    clean_metrics = compute_metrics(y_true, y_pred_clean, "Clean Traffic (No Attack)")
    adv_metrics   = compute_metrics(y_true, y_pred_adv,   "Adversarial Traffic (FGSM Attack, No Defense)")

    # ---- Save report -----------------------------------------------------
    report = {
        "test_windows":          len(windows),
        "clean_inference":       clean_metrics,
        "adversarial_inference": adv_metrics,
        "robustness_drop_f1":    round(clean_metrics["f1"] - adv_metrics["f1"], 4),
        "model_path":            str(weights_path),
    }

    report_path = Path("data/evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  [SAVED] Evaluation report -> {report_path}")
    print(f"\n  Robustness drop (F1): {report['robustness_drop_f1']:.4f}")
    print("  (This is the F1 loss caused by FGSM attack without defense)")
    print("=" * 60)

    return report


if __name__ == "__main__":
    evaluate()
