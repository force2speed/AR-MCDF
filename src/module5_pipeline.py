"""
=============================================================================
AR-MCDF  |  Module 5 -- End-to-End Master Pipeline  (CUDA-accelerated)
=============================================================================
Purpose:
    Integrates Modules 1-4 into a single, reproducible training loop.
    Automatically uses CUDA (GTX 1650) if available, falls back to CPU.

    After training completes:
        models/stgcn_weights.pt  -- saved ST-GCN state dict
        models/qtable.npy        -- saved Q-table array
        data/pipeline_metrics.json  -- live training metrics (written per epoch)

Standalone execution:
    py src/module5_pipeline.py

=============================================================================
"""

from __future__ import annotations

import logging
import os
import sys
import time
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

# ---------------------------------------------------------------------------
# Path setup: add src/ so sibling modules are importable when run directly
# ---------------------------------------------------------------------------
_SRC  = Path(__file__).resolve().parent
_ROOT = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
os.chdir(_ROOT)

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------
from module1_ingestion import run_ingestion, NUMERICAL_FEATURES          # noqa: E402
from module2_sanitizer import (                                           # noqa: E402
    DummyProbe, FGSMAttacker, ConfidenceThresholdSanitizer,
)
from module3_stgcn import (                                               # noqa: E402
    WindowGraphBuilder, SpatioTemporalGCN, FocalLoss,
)
from module4_mitigator import (                                           # noqa: E402
    CloudMitigationEnv, QLearningBouncer, PolicyPrinter,
    RewardConfig, StateDiscretiser, ACTION_LABELS, BETA1, BETA2,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(module)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("module2_sanitizer").setLevel(logging.WARNING)


# ===========================================================================
# Pipeline Configuration  (single source of truth)
# ===========================================================================

class PipelineConfig:
    """
    Central hyperparameter registry for the AR-MCDF pipeline.
    """
    # ---- Ingestion (Module 1) -------------------------------------------
    csv_path:        str   = "data/raw/UNSW_NB15_training-set.csv"
    window_size:     int   = 50
    stride:          int   = 25

    # ---- Sanitizer (Module 2) -------------------------------------------
    fgsm_epsilon:    float = 0.10
    san_tau:         float = 0.85
    san_method:      str   = "gaussian"
    san_kernel:      int   = 5
    san_sigma:       float = 1.5

    # ---- ST-GCN (Module 3) ----------------------------------------------
    n_features:      int   = len(NUMERICAL_FEATURES)   # 27
    gcn_hidden:      int   = 128
    attn_heads:      int   = 4
    fc_hidden:       int   = 64
    n_classes:       int   = 2
    gcn_dropout:     float = 0.4
    pool_mode:       str   = "both"
    edge_method:     str   = "cosine"
    edge_threshold:  float = 0.60
    edge_k:          int   = 4
    focal_alpha:     float = 0.25
    focal_gamma:     float = 2.0
    lr:              float = 0.001

    # ---- Mitigator (Module 4) -------------------------------------------
    n_threat_bins:   int   = 4
    n_resource_bins: int   = 4
    q_alpha:         float = 0.10
    q_gamma:         float = 0.90
    q_epsilon:       float = 1.00
    q_epsilon_min:   float = 0.01
    q_epsilon_decay: float = 0.9995

    # ---- Training Loop --------------------------------------------------
    n_epochs:        int   = 15
    batch_size:      int   = 32    # Mini-batch size for ST-GCN (fits GTX 1650 4GB)
    random_seed:     int   = 42

    # ---- Model output paths ---------------------------------------------
    model_dir:       str   = "models"
    stgcn_path:      str   = "models/stgcn_weights.pt"
    qtable_path:     str   = "models/qtable.npy"


# ===========================================================================
# Epoch Metrics Container
# ===========================================================================

class EpochMetrics:
    """Accumulates per-batch metrics and computes epoch averages."""

    def __init__(self) -> None:
        self.losses:         List[float] = []
        self.correct:        int         = 0
        self.total:          int         = 0
        self.q_rewards:      List[float] = []
        self.threat_levels:  List[float] = []
        self.actions:        List[int]   = []

    def update_clf(self, loss: float, pred: int, label: int) -> None:
        self.losses.append(loss)
        self.correct += int(pred == label)
        self.total   += 1

    def update_rl(self, reward: float, threat: float, action: int) -> None:
        self.q_rewards.append(reward)
        self.threat_levels.append(threat)
        self.actions.append(action)

    @property
    def avg_loss(self) -> float:
        return float(np.mean(self.losses)) if self.losses else 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def avg_q_reward(self) -> float:
        return float(np.mean(self.q_rewards)) if self.q_rewards else 0.0


# ===========================================================================
# AR-MCDF Master Pipeline
# ===========================================================================

class ARMCDFPipeline:
    """
    End-to-end Adversarial-Resilient Multi-Cloud DevSecOps pipeline.
    CUDA-accelerated: automatically uses GPU if available.
    Trains in mini-batches of size cfg.batch_size for VRAM efficiency.
    """

    def __init__(self, cfg: PipelineConfig | None = None) -> None:
        self.cfg = cfg or PipelineConfig()

        # ---- Device selection (CUDA first) --------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("[DEVICE] Using: %s", self.device)
        if self.device.type == "cuda":
            logger.info(
                "[DEVICE] GPU: %s | VRAM: %.1f GB",
                torch.cuda.get_device_name(0),
                torch.cuda.get_device_properties(0).total_memory / 1e9,
            )

        self._build_all()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _build_all(self) -> None:
        cfg = self.cfg
        torch.manual_seed(cfg.random_seed)
        np.random.seed(cfg.random_seed)

        # ---- Module 1: Ingestion ----------------------------------------
        logger.info("[M1] Loading & preprocessing telemetry...")
        self.windows, self.labels, self.scaler = run_ingestion(
            csv_path    = cfg.csv_path,
            window_size = cfg.window_size,
            stride      = cfg.stride,
        )
        self.n_windows = len(self.windows)
        logger.info(
            "[M1] Ingestion complete | windows=%d | shape=%s",
            self.n_windows, self.windows[0].shape,
        )

        # ---- Module 2: Sanitizer components -----------------------------
        logger.info("[M2] Building FGSM Attacker + Confidence Sanitizer...")
        self.probe = DummyProbe(window_size=cfg.window_size, n_features=cfg.n_features)
        self.attacker  = FGSMAttacker(model=self.probe, epsilon=cfg.fgsm_epsilon)
        self.sanitizer = ConfidenceThresholdSanitizer(
            model       = self.probe,
            tau         = cfg.san_tau,
            method      = cfg.san_method,
            kernel_size = cfg.san_kernel,
            sigma       = cfg.san_sigma,
        )

        # ---- Module 3: ST-GCN + Focal Loss + Optimizer ------------------
        logger.info("[M3] Building SpatioTemporalGCN on %s...", self.device)
        self.graph_builder = WindowGraphBuilder(
            method    = cfg.edge_method,
            threshold = cfg.edge_threshold,
            k         = cfg.edge_k,
        )
        self.model = SpatioTemporalGCN(
            n_features = cfg.n_features,
            gcn_hidden = cfg.gcn_hidden,
            attn_heads = cfg.attn_heads,
            fc_hidden  = cfg.fc_hidden,
            n_classes  = cfg.n_classes,
            dropout    = cfg.gcn_dropout,
            pool       = cfg.pool_mode,
        ).to(self.device)

        self.criterion = FocalLoss(
            alpha     = cfg.focal_alpha,
            gamma     = cfg.focal_gamma,
            reduction = "mean",
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info("[M3] SpatioTemporalGCN | params=%d | device=%s", total_params, self.device)

        # ---- Module 4: Q-Learning Bouncer + Environment -----------------
        logger.info("[M4] Building Q-Learning Bouncer + CloudMitigationEnv...")
        self.disc    = StateDiscretiser(n_threat_bins=cfg.n_threat_bins, n_resource_bins=cfg.n_resource_bins)
        self.env     = CloudMitigationEnv(discretiser=self.disc, reward_cfg=RewardConfig(), seed=cfg.random_seed)
        self.bouncer = QLearningBouncer(
            n_states      = self.disc.n_states,
            n_actions      = 4,
            alpha          = cfg.q_alpha,
            gamma          = cfg.q_gamma,
            epsilon        = cfg.q_epsilon,
            epsilon_min    = cfg.q_epsilon_min,
            epsilon_decay  = cfg.q_epsilon_decay,
            seed           = cfg.random_seed,
        )

        # History across all epochs
        self.epoch_history: List[dict] = []

    # ------------------------------------------------------------------
    # Mini-batch processing (CUDA-optimised)
    # ------------------------------------------------------------------

    def _process_batch(
        self,
        batch_windows: List[np.ndarray],
        batch_labels:  List[np.ndarray],
        rng: np.random.Generator,
    ) -> Tuple[float, List[int], List[int], List[float], List[int], List[float]]:
        """
        Process a mini-batch of windows together on GPU.
        Returns lists of: losses, preds, trues, threats, actions, q_rewards.
        """
        cfg   = self.cfg
        graphs = []
        trues  = []

        for win_np, label_np in zip(batch_windows, batch_labels):
            true_label = int(label_np.mean() >= 0.5)
            trues.append(true_label)

            # FGSM attack on CPU (probe runs on CPU)
            x_clean = torch.tensor(win_np, dtype=torch.float32)
            y_lbl   = torch.tensor([true_label], dtype=torch.long)
            x_adv   = self.attacker.attack(x_clean, y_lbl)

            # Sanitize
            x_san, _, _ = self.sanitizer.sanitize(x_adv)

            # Build graph (on CPU) — will be moved to device via Batch
            graph = self.graph_builder.build(x_san, label=true_label)
            graphs.append(graph)

        # Collate into PyG batch and move to GPU
        from torch_geometric.data import Batch as PyGBatch
        batch = PyGBatch.from_data_list(graphs).to(self.device)
        targets = torch.tensor(trues, dtype=torch.long, device=self.device)

        # Forward pass on GPU
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(batch)                         # (B, 2)
        loss   = self.criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        probs        = F.softmax(logits.detach(), dim=1)   # (B, 2)
        threat_levels = probs[:, 1].cpu().tolist()
        preds         = probs.argmax(dim=1).cpu().tolist()
        batch_loss    = float(loss.item())

        # Q-Learning (CPU tabular — no GPU needed)
        actions   = []
        q_rewards = []
        for threat_level in threat_levels:
            resource_value = float(rng.uniform(0.0, 1.0))
            state_idx, _, _ = self.env.reset()
            state_idx       = self.disc.discretise(threat_level, resource_value)
            action          = self.bouncer.choose_action(state_idx)
            next_state, q_reward, _ = self.env.step(action, threat=threat_level, resource=resource_value)
            self.bouncer.update(state_idx, action, q_reward, next_state)
            actions.append(action)
            q_rewards.append(q_reward)

        return batch_loss, preds, trues, threat_levels, actions, q_rewards

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.random_seed)

        # Create models directory
        Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)

        SEP  = "=" * 72
        SEP2 = "-" * 72

        print(f"\n{SEP}")
        print("  AR-MCDF  |  Master Pipeline Training")
        print(f"  Device   : {self.device}")
        print(f"  Epochs   : {cfg.n_epochs} | Batch Size: {cfg.batch_size}")
        print(f"  Windows  : {self.n_windows:,} | Steps/epoch: {self.n_windows // cfg.batch_size:,}")
        print(SEP)

        for epoch in range(1, cfg.n_epochs + 1):
            t0      = time.perf_counter()
            metrics = EpochMetrics()

            # Shuffle windows each epoch
            indices = np.random.permutation(self.n_windows)

            # Mini-batch loop
            for batch_start in range(0, self.n_windows, cfg.batch_size):
                batch_idx = indices[batch_start:batch_start + cfg.batch_size]
                batch_wins   = [self.windows[i] for i in batch_idx]
                batch_labels = [self.labels[i]  for i in batch_idx]

                batch_loss, preds, trues, threats, actions, q_rews = self._process_batch(
                    batch_wins, batch_labels, rng
                )

                for i in range(len(preds)):
                    metrics.update_clf(batch_loss, preds[i], trues[i])
                    metrics.update_rl(q_rews[i], threats[i], actions[i])

            # Decay epsilon once per epoch
            self.bouncer.decay_epsilon()

            elapsed = time.perf_counter() - t0
            epoch_stats = {
                "epoch":      epoch,
                "focal_loss": metrics.avg_loss,
                "accuracy":   metrics.accuracy,
                "q_reward":   metrics.avg_q_reward,
                "epsilon":    self.bouncer.epsilon,
                "elapsed_s":  elapsed,
            }
            self.epoch_history.append(epoch_stats)
            self._write_metrics()

            print(
                f"  Epoch {epoch:>2}/{cfg.n_epochs} | "
                f"FocalLoss={metrics.avg_loss:.5f} | "
                f"Acc={metrics.accuracy*100:5.1f}% | "
                f"Q-Reward={metrics.avg_q_reward:+6.3f} | "
                f"eps={self.bouncer.epsilon:.4f} | "
                f"[{elapsed:.1f}s]"
            )

        # ------------------------------------------------------------------
        # Save trained artifacts
        # ------------------------------------------------------------------
        self._save_models()
        self._print_final_summary()

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def _save_models(self) -> None:
        cfg = self.cfg

        # Save ST-GCN weights (move to CPU first for portability)
        stgcn_path = Path(cfg.stgcn_path)
        torch.save(self.model.cpu().state_dict(), stgcn_path)
        self.model.to(self.device)  # move back to device
        logger.info("[SAVE] ST-GCN weights -> %s", stgcn_path)

        # Save Q-table
        qtable_path = Path(cfg.qtable_path)
        np.save(qtable_path, self.bouncer.Q)
        logger.info("[SAVE] Q-table -> %s", qtable_path)

        # Mark metrics as done
        self._write_metrics(status="done")

        print(f"\n  [SAVED] ST-GCN weights  -> {stgcn_path}")
        print(f"  [SAVED] Q-table         -> {qtable_path}")

    # ------------------------------------------------------------------
    # Dashboard Metrics Export
    # ------------------------------------------------------------------

    def _write_metrics(self, status: str = "training") -> None:
        metrics_file = Path("data/pipeline_metrics.json")
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "status":          status,
            "current_epoch":   len(self.epoch_history),
            "total_epochs":    self.cfg.n_epochs,
            "windows_total":   self.n_windows,
            "fgsm_epsilon":    self.cfg.fgsm_epsilon,
            "n_threat_bins":   self.cfg.n_threat_bins,
            "n_resource_bins": self.cfg.n_resource_bins,
            "epoch_history":   self.epoch_history,
            "q_policy":        self.bouncer.greedy_policy().tolist(),
            "last_updated":    time.time(),
        }
        try:
            with open(metrics_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning("Failed to write metrics: %s", e)

    # ------------------------------------------------------------------
    # Final Report
    # ------------------------------------------------------------------

    def _print_final_summary(self) -> None:
        cfg  = self.cfg
        last = self.epoch_history[-1]
        best = min(self.epoch_history, key=lambda d: d["focal_loss"])
        SEP  = "=" * 72

        total_params     = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"\n{SEP}")
        print("  AR-MCDF  |  FINAL PIPELINE SUMMARY")
        print(SEP)
        print(f"\n  [Module 1 - Ingestion]")
        print(f"  Windows produced        : {self.n_windows:,}")
        print(f"  Window shape            : ({cfg.window_size}, {cfg.n_features})")
        print(f"\n  [Module 2 - Sanitizer]")
        print(f"  FGSM epsilon            : {cfg.fgsm_epsilon}")
        print(f"  Confidence threshold t  : {cfg.san_tau}")
        print(f"  Smoothing method        : {cfg.san_method}")
        print(f"\n  [Module 3 - ST-GCN Classifier]")
        print(f"  Device                  : {self.device}")
        print(f"  Total parameters        : {total_params:,}")
        print(f"  Trainable parameters    : {trainable_params:,}")
        print(f"  Final epoch accuracy    : {last['accuracy']*100:.2f}%")
        print(f"  Best epoch focal loss   : {best['focal_loss']:.5f} (epoch {best['epoch']})")
        print(f"\n  [Epoch Training Curve]")
        print(f"  {'Epoch':>5}  {'FocalLoss':>10}  {'Accuracy':>9}  {'Q-Reward':>9}  {'Epsilon':>8}  {'Time':>6}")
        print(f"  {'-----':>5}  {'----------':>10}  {'---------':>9}  {'---------':>9}  {'-------':>8}  {'------':>6}")
        for h in self.epoch_history:
            print(
                f"  {h['epoch']:>5}  {h['focal_loss']:>10.5f}  "
                f"{h['accuracy']*100:>8.2f}%  "
                f"{h['q_reward']:>+9.3f}  "
                f"{h['epsilon']:>8.4f}  "
                f"{h['elapsed_s']:>5.1f}s"
            )
        print(f"\n  [Module 4 - Q-Learning Bouncer]")
        print(f"  Final epsilon           : {self.bouncer.epsilon:.6f}")
        print(f"  alpha / gamma           : {cfg.q_alpha} / {cfg.q_gamma}")
        printer = PolicyPrinter(self.bouncer, self.disc)
        printer.print_policy_grid()
        print(f"\n{SEP}")
        print("  [OK]  AR-MCDF Pipeline completed successfully.")
        print(f"{SEP}\n")


# ===========================================================================
# Entry Point
# ===========================================================================

if __name__ == "__main__":
    logger.info("=" * 72)
    logger.info("AR-MCDF  |  Module 5 -- CUDA-accelerated Pipeline")
    logger.info("=" * 72)

    cfg = PipelineConfig()
    pipeline = ARMCDFPipeline(cfg=cfg)
    pipeline.train()
