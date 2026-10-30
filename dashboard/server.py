"""
AR-MCDF Dashboard Server  (Production-Ready)
=============================================
  GET  /              -> main metrics dashboard
  GET  /demo          -> attack & defense animation page
  GET  /api/metrics   -> pipeline training metrics
  GET  /api/evaluation -> model evaluation report
  POST /api/run-demo  -> run inference-only demo (uses pre-trained weights)
  GET  /api/status    -> health check

Run in development:
    py dashboard/server.py

Run in production (via Docker / Gunicorn):
    gunicorn -w 2 -b 0.0.0.0:8050 "dashboard.server:app"
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Paths  & src/ bootstrap
# ---------------------------------------------------------------------------
_HERE         = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_METRICS_FILE = _PROJECT_ROOT / "data" / "pipeline_metrics.json"
_EVAL_FILE    = _PROJECT_ROOT / "data" / "evaluation_report.json"
_MODELS_DIR   = _PROJECT_ROOT / "models"
_STGCN_PATH   = _MODELS_DIR   / "stgcn_weights.pt"
_QTABLE_PATH  = _MODELS_DIR   / "qtable.npy"
_SRC          = _PROJECT_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Pre-load heavy modules & trained weights once at startup
# ---------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn.functional as F

from module1_ingestion import run_ingestion, NUMERICAL_FEATURES
from module2_sanitizer import DummyProbe, FGSMAttacker, WhiteBoxGraphAttacker, ConfidenceThresholdSanitizer
from module3_stgcn import SpatioTemporalGCN, WindowGraphBuilder
from module4_mitigator import (
    QLearningBouncer, StateDiscretiser, CloudMitigationEnv,
    RewardConfig, ACTION_LABELS, N_ACTIONS,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load ST-GCN ---------------------------------------------------------
_stgcn_model: SpatioTemporalGCN | None = None

def _load_stgcn() -> SpatioTemporalGCN:
    global _stgcn_model
    if _stgcn_model is not None:
        return _stgcn_model
    model = SpatioTemporalGCN(
        n_features=len(NUMERICAL_FEATURES), gcn_hidden=128, attn_heads=4,
        fc_hidden=64, n_classes=2, dropout=0.0, pool="both",
    )
    if _STGCN_PATH.exists():
        model.load_state_dict(torch.load(_STGCN_PATH, map_location=DEVICE))
        print(f"[SERVER] Loaded ST-GCN weights from {_STGCN_PATH}")
    else:
        print(f"[SERVER] WARNING: No trained weights at {_STGCN_PATH}. Using random init.")
        print("[SERVER] Run: py src/module5_pipeline.py  to train the model.")
    model = model.to(DEVICE)
    model.eval()
    _stgcn_model = model
    return model

# ---- Load Q-table --------------------------------------------------------
_q_table: np.ndarray | None = None

def _load_qtable() -> np.ndarray | None:
    global _q_table
    if _q_table is not None:
        return _q_table
    if _QTABLE_PATH.exists():
        _q_table = np.load(_QTABLE_PATH)
        print(f"[SERVER] Loaded Q-table from {_QTABLE_PATH}")
    else:
        print(f"[SERVER] WARNING: No Q-table at {_QTABLE_PATH}. Will skip Stage 5.")
    return _q_table


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")

app = Flask(
    __name__,
    template_folder=str(_HERE / "templates"),
    static_folder=str(_HERE / "static"),
)
CORS(app, origins=ALLOWED_ORIGIN)


# ---------------------------------------------------------------------------
# Helper: heatmap normalization
# ---------------------------------------------------------------------------
def to_heatmap(arr, n_feat=10, ref_arr=None):
    if isinstance(arr, torch.Tensor):
        a = arr.detach().cpu().numpy()
    else:
        a = np.array(arr, dtype=float)
    if a.ndim == 3:
        a = a.squeeze(0)
    a = a[:, :n_feat]
    if ref_arr is not None:
        if isinstance(ref_arr, torch.Tensor):
            ref = ref_arr.detach().cpu().numpy()
        else:
            ref = np.array(ref_arr, dtype=float)
        if ref.ndim == 3:
            ref = ref.squeeze(0)
        ref = ref[:, :n_feat]
        lo = ref.min(axis=0, keepdims=True)
        hi = ref.max(axis=0, keepdims=True)
    else:
        lo = a.min(axis=0, keepdims=True)
        hi = a.max(axis=0, keepdims=True)
    rng = np.where(hi - lo > 1e-8, hi - lo, 1.0)
    return np.clip((a - lo) / rng, 0.0, 1.0).tolist()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/demo")
def demo():
    return render_template("demo.html")


@app.route("/api/metrics")
def metrics():
    if not _METRICS_FILE.exists():
        return jsonify({
            "status": "idle",
            "message": "Pipeline not trained yet. Run: py src/module5_pipeline.py",
            "current_epoch": 0, "total_epochs": 0, "windows_total": 0,
            "epoch_history": [], "q_policy": [],
            "n_threat_bins": 4, "n_resource_bins": 4, "last_updated": None,
        })
    try:
        with open(_METRICS_FILE, "r") as f:
            data = json.load(f)
        return jsonify(data)
    except (json.JSONDecodeError, OSError) as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/evaluation")
def evaluation():
    if not _EVAL_FILE.exists():
        return jsonify({
            "status": "not_evaluated",
            "message": "Run: py src/evaluate.py  to generate the evaluation report.",
        })
    try:
        with open(_EVAL_FILE, "r") as f:
            data = json.load(f)
        return jsonify(data)
    except (json.JSONDecodeError, OSError) as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/status")
def status():
    return jsonify({
        "server":              "online",
        "device":              str(DEVICE),
        "weights_loaded":      _STGCN_PATH.exists(),
        "qtable_loaded":       _QTABLE_PATH.exists(),
        "metrics_available":   _METRICS_FILE.exists(),
        "evaluation_available": _EVAL_FILE.exists(),
        "server_time":         datetime.now().isoformat(),
    })


@app.route("/api/run-demo", methods=["POST"])
def run_demo():
    """
    Run 5-stage inference pipeline using pre-loaded trained weights.
    No re-training on every request — weights are loaded at startup.
    """
    try:
        req_data  = request.get_json(silent=True) or {}
        max_stage = min(int(req_data.get("max_stage", 5)), 5)
        res = {"ok": True, "device": str(DEVICE), "weights_loaded": _STGCN_PATH.exists()}

        CSV_PATH = str(_PROJECT_ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv")

        # ---- Stage 1: Ingestion ------------------------------------------
        windows, labels, _ = run_ingestion(csv_path=CSV_PATH, window_size=50, stride=25)

        normal_idx = attack_idx = None
        for i, lbl in enumerate(labels):
            if normal_idx is None and int(lbl.mean() >= 0.5) == 0:
                normal_idx = i
            if attack_idx is None and int(lbl.mean() >= 0.5) == 1:
                attack_idx = i
            if normal_idx is not None and attack_idx is not None:
                break
        if attack_idx is None:
            attack_idx = int(np.argmax([float(l.mean()) for l in labels]))

        normal_win   = windows[normal_idx]
        attack_win   = windows[attack_idx]
        attack_ratio = float(labels[attack_idx].mean() * 100)

        res["stage1"] = {
            "total_windows":  len(windows),
            "attack_ratio":   attack_ratio,
            "normal_heatmap": to_heatmap(normal_win),
            "attack_heatmap": to_heatmap(attack_win),
        }
        if max_stage < 2:
            return jsonify(res)

        # ---- Stage 2: FGSM Attack ----------------------------------------
        stgcn         = _load_stgcn()
        graph_builder = WindowGraphBuilder(method="cosine", threshold=0.60, k=4)
        
        # White-Box Attack directly against the ST-GCN!
        attacker = WhiteBoxGraphAttacker(model=stgcn, epsilon=0.15) # higher epsilon for the robust model

        x_clean = torch.tensor(attack_win, dtype=torch.float32)
        y_true  = torch.tensor([1], dtype=torch.long)
        x_adv   = attacker.attack(x_clean, y_true, graph_builder)

        l2_adv  = float((x_adv - x_clean).pow(2).sum().sqrt().item())
        linf    = float((x_adv - x_clean).abs().max().item())
        pct_alt = float((x_adv != x_clean).float().mean().item() * 100)

        res["stage2"] = {
            "l2_norm":     round(l2_adv, 4),
            "linf":        round(linf, 4),
            "pct_altered": round(pct_alt, 1),
            "adv_heatmap": to_heatmap(x_adv, ref_arr=attack_win),
        }
        if max_stage < 3:
            return jsonify(res)

        # ---- Stage 3: Sanitizer ------------------------------------------
        probe = DummyProbe(window_size=50, n_features=len(NUMERICAL_FEATURES))
        sanitizer = ConfidenceThresholdSanitizer(
            model=probe, tau=0.85, method="gaussian", kernel_size=5, sigma=0.9
        )
        x_san, conf_tensor, flagged_tensor = sanitizer.sanitize(x_adv)
        conf_score  = float(conf_tensor.item())
        was_flagged = bool(flagged_tensor.item())
        l2_after    = float((x_san - x_clean).pow(2).sum().sqrt().item())

        x_san_t = x_san.float() if isinstance(x_san, torch.Tensor) else torch.tensor(x_san, dtype=torch.float32)
        if x_san_t.dim() == 3:
            x_san_t = x_san_t.squeeze(0)

        res["stage3"] = {
            "confidence":  round(conf_score, 4),
            "flagged":     was_flagged,
            "threshold":   0.85,
            "l2_before":   round(l2_adv, 4),
            "l2_after":    round(l2_after, 4),
            "san_heatmap": to_heatmap(x_san_t, ref_arr=attack_win),
        }
        if max_stage < 4:
            return jsonify(res)

        # ---- Stage 4: ST-GCN Detection (uses pre-loaded trained weights) --
        stgcn         = _load_stgcn()
        graph_builder = WindowGraphBuilder(method="cosine", threshold=0.60, k=4)
        x_adv_t       = x_adv.squeeze(0) if x_adv.dim() == 3 else x_adv
        label_int     = int(labels[attack_idx].mean() >= 0.5)

        graph_no_def = graph_builder.build(x_adv_t, label=label_int).to(DEVICE)
        graph_def    = graph_builder.build(x_san_t,  label=label_int).to(DEVICE)

        with torch.no_grad():
            probs_no_def = F.softmax(stgcn(graph_no_def), dim=-1)
            probs_def    = F.softmax(stgcn(graph_def),    dim=-1)

        prob_attack_no_def = float(probs_no_def[0, 1].item())
        prob_attack_def    = float(probs_def[0, 1].item())

        pred_no_def = "ATTACK" if prob_attack_no_def >= 0.5 else "NORMAL"
        pred_def    = "ATTACK" if (was_flagged or prob_attack_def >= 0.5) else "NORMAL"

        res["stage4"] = {
            "without_defense": {
                "prob_normal": round(float(probs_no_def[0, 0].item()), 4),
                "prob_attack": round(prob_attack_no_def, 4),
                "prediction":  pred_no_def,
                "fooled":      pred_no_def == "NORMAL",
            },
            "with_defense": {
                "prob_normal": round(float(probs_def[0, 0].item()), 4),
                "prob_attack": round(prob_attack_def, 4),
                "prediction":  pred_def,
                "caught_by_sanitizer": was_flagged,
            },
        }
        if max_stage < 5:
            return jsonify(res)

        # ---- Stage 5: Q-Learning Bouncer (uses pre-loaded Q-table) -------
        disc    = StateDiscretiser(n_threat_bins=4, n_resource_bins=4)
        qtable  = _load_qtable()

        bouncer = QLearningBouncer(
            n_states=disc.n_states, n_actions=N_ACTIONS,
            alpha=0.10, gamma=0.90, epsilon=0.0, seed=42,
        )
        if qtable is not None:
            bouncer.Q = qtable
        else:
            # Fall back: quick 500-episode train
            env = CloudMitigationEnv(discretiser=disc, reward_cfg=RewardConfig(), seed=42)
            bouncer.train(env, n_episodes=500, log_interval=500)

        resource_value = 0.95

        state_nd  = disc.discretise(prob_attack_no_def, resource_value)
        action_nd = int(bouncer.choose_action(state_nd))
        iso_nd    = round(0.65 * prob_attack_no_def + 0.35 * resource_value, 4)

        state_d   = disc.discretise(prob_attack_def, resource_value)
        action_d  = int(bouncer.choose_action(state_d))
        iso_d     = round(0.65 * prob_attack_def + 0.35 * resource_value, 4)

        res["stage5"] = {
            "resource_value": resource_value,
            "without_defense": {
                "threat_level":    round(prob_attack_no_def, 4),
                "isolation_score": iso_nd,
                "action":          action_nd,
                "action_name":     ACTION_LABELS[action_nd],
            },
            "with_defense": {
                "threat_level":    round(prob_attack_def, 4),
                "isolation_score": iso_d,
                "action":          action_d,
                "action_name":     ACTION_LABELS[action_d],
            },
        }

        return jsonify(res)

    except Exception as e:
        import traceback
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  AR-MCDF Dashboard Server  (Production Mode)")
    print("=" * 60)
    print(f"  Device    : {DEVICE}")
    print(f"  Weights   : {'LOADED [OK]' if _STGCN_PATH.exists() else 'NOT FOUND -- run training first'}")
    print(f"  Q-table   : {'LOADED [OK]' if _QTABLE_PATH.exists() else 'NOT FOUND -- run training first'}")
    print(f"  Dashboard : http://localhost:8050")
    print(f"  Demo      : http://localhost:8050/demo")
    print(f"  Project   : {_PROJECT_ROOT}")
    print("=" * 60)

    # Pre-load on startup
    _load_stgcn()
    _load_qtable()

    app.run(host="0.0.0.0", port=8050, debug=False, threaded=True)
