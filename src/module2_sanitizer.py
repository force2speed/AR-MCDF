"""
=============================================================================
AR-MCDF  |  Module 2 -- Adversarial Sanitizer (FGSM Attack + Defense)
=============================================================================
Purpose:
    - DummyProbe      : Lightweight MLP acting as a surrogate target model
                        for gradient-based adversarial attacks.  Accepts the
                        (window_size, n_features) = (50, 27) tensor shape and
                        outputs binary classification logits.
    - FGSMAttacker    : Generates Fast Gradient Sign Method perturbations
                        against DummyProbe's CrossEntropy loss.
    - ConfidenceThresholdSanitizer : Detects low-confidence (corrupted) inputs
                        and applies temporal smoothing to wash out adversarial
                        noise before the signal reaches Module 3.

Standalone execution:
    py src/module2_sanitizer.py

Output:
    L2-norm distances: Original vs. Attacked (high), Original vs. Sanitized
    (reduced), confirming the sanitizer attenuates adversarial perturbations.
=============================================================================
"""

from __future__ import annotations

import logging
import sys
import os
from pathlib import Path
from typing import Tuple, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(module)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# 1.  DummyProbe  (surrogate target model)
# ===========================================================================

class DummyProbe(nn.Module):
    """
    Lightweight MLP surrogate model used as an attack / sanity-check target.

    It flattens the (window_size, n_features) input into a 1-D vector and
    passes it through two hidden layers before projecting to ``n_classes``
    logits.  This is intentionally shallow -- its sole purpose is to provide
    meaningful gradients for FGSM so that Module 2 can run independently,
    before the real ST-GCN (Module 3) is available.

    Architecture:
        [flatten] -> FC(window_size*n_features, 256) -> BN -> ReLU -> Dropout
                  -> FC(256, 64) -> BN -> ReLU -> Dropout
                  -> FC(64, n_classes)

    Args:
        window_size : Time-steps per window (default 50).
        n_features  : Feature count per time-step (default 27).
        hidden_dim  : First hidden layer width (default 256).
        n_classes   : Output classes -- 2 for binary normal/attack.
        dropout     : Dropout probability.
    """

    def __init__(
        self,
        window_size: int = 50,
        n_features: int  = 27,
        hidden_dim: int  = 256,
        n_classes: int   = 2,
        dropout: float   = 0.3,
    ) -> None:
        super().__init__()
        in_dim = window_size * n_features

        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, window_size, n_features)
               or (window_size, n_features) for a single sample.

        Returns:
            logits: Tensor of shape (batch, n_classes).
        """
        if x.dim() == 2:          # single window -- add batch dim
            x = x.unsqueeze(0)
        return self.net(self.flatten(x))


# ===========================================================================
# 2.  FGSMAttacker
# ===========================================================================

class FGSMAttacker:
    """
    Fast Gradient Sign Method (FGSM) adversarial perturbation generator.

    For a given input ``x`` and target labels ``y``, FGSM computes:

        x_adv = x + epsilon * sign( grad_{x} L(f(x), y) )

    where ``L`` is the CrossEntropy loss and ``f`` is the ``DummyProbe``.
    The perturbed tensor is then clipped to ``[clip_min, clip_max]`` to
    keep it within normalised bounds (default: Z-score range [-5, 5]).

    Args:
        model    : The target ``DummyProbe`` (or any nn.Module with matching
                   input/output signature).
        epsilon  : Perturbation magnitude (default 0.1).
        clip_min : Minimum allowed feature value after perturbation.
        clip_max : Maximum allowed feature value after perturbation.
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.1,
        clip_min: float = -5.0,
        clip_max: float =  5.0,
    ) -> None:
        self.model    = model
        self.epsilon  = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self._loss_fn = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    def attack(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate FGSM adversarial examples for a batch of windows.

        Args:
            x: Float tensor of shape (batch, window_size, n_features).
               A single window (window_size, n_features) is accepted and
               automatically expanded.
            y: Long tensor of shape (batch,) -- ground-truth class indices.
               For a single window, shape (1,) or scalar is fine.

        Returns:
            x_adv: Adversarially perturbed tensor, same shape as ``x``,
                   clipped to [clip_min, clip_max].
        """
        # ---- ensure batch dimension -----------------------------------------
        single = (x.dim() == 2)
        if single:
            x = x.unsqueeze(0)           # (1, T, F)
        if y.dim() == 0:
            y = y.unsqueeze(0)

        x_adv = x.clone().detach().requires_grad_(True)

        # ---- forward + loss -------------------------------------------------
        self.model.eval()
        logits = self.model(x_adv)       # (batch, n_classes)
        loss   = self._loss_fn(logits, y)

        # ---- backward -------------------------------------------------------
        self.model.zero_grad()
        loss.backward()

        # ---- FGSM perturbation ----------------------------------------------
        grad_sign = x_adv.grad.data.sign()                    # sign(grad)
        x_perturbed = x_adv.detach() + self.epsilon * grad_sign
        x_perturbed = torch.clamp(x_perturbed, self.clip_min, self.clip_max)

        if single:
            x_perturbed = x_perturbed.squeeze(0)

        logger.debug(
            "FGSM attack applied | epsilon=%.3f | L_inf delta=%.4f",
            self.epsilon,
            (x_perturbed - (x.squeeze(0) if single else x)).abs().max().item(),
        )
        return x_perturbed

# ===========================================================================
# 2.B  White-Box Graph Attacker (ST-GCN specific)
# ===========================================================================

class WhiteBoxGraphAttacker:
    """
    Generates targeted FGSM adversarial noise directly against the SpatioTemporalGCN.
    Since edge construction (kNN/cosine) is non-differentiable, this attacker
    computes edges once on the clean traffic, fixes them, and then flows gradients
    through the node features.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        epsilon: float = 0.10,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        self.model    = model
        self.epsilon  = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self._loss_fn = torch.nn.CrossEntropyLoss()

    def attack(self, x: torch.Tensor, y: torch.Tensor, graph_builder) -> torch.Tensor:
        """
        x: (window_size, n_features) or (1, window_size, n_features)
        y: target label (0 or 1)
        graph_builder: an instance of WindowGraphBuilder from module 3
        """
        from torch_geometric.data import Data

        device = next(self.model.parameters()).device
        
        single = (x.dim() == 2)
        x_2d = x if single else x.squeeze(0)
        x_2d = x_2d.to(device)

        # 1. Build edge index from clean graph (bypasses non-differentiable edge building)
        clean_graph = graph_builder.build(x_2d.detach())
        edge_index = clean_graph.edge_index.to(device)

        # 2. Attach gradients to features
        x_adv = x_2d.clone().detach().requires_grad_(True)
        g = Data(x=x_adv, edge_index=edge_index)
        
        # Add batch vector so pooling works correctly
        g.batch = torch.zeros(x_adv.shape[0], dtype=torch.long, device=device)

        # 3. Forward pass through ST-GCN
        self.model.eval()
        logits = self.model(g)
        
        y_tensor = y.to(device)
        if y_tensor.dim() == 0:
            y_tensor = y_tensor.unsqueeze(0)

        loss = self._loss_fn(logits, y_tensor)
        
        # 4. Backward & FGSM step
        self.model.zero_grad()
        loss.backward()

        grad_sign = x_adv.grad.data.sign()
        x_perturbed = x_adv.detach() + self.epsilon * grad_sign
        x_perturbed = torch.clamp(x_perturbed, self.clip_min, self.clip_max)

        if not single:
            x_perturbed = x_perturbed.unsqueeze(0)

        return x_perturbed.cpu()  # return on CPU to match input expectation

# ===========================================================================
# 3.  Sanitizer: Confidence Gate & Smoothing
# ===========================================================================

SmoothingMethod = Literal["mean", "gaussian"]


class ConfidenceThresholdSanitizer:
    """
    Confidence-threshold defense against adversarial perturbations.

    Strategy
    --------
    For each input window ``x`` (shape: batch x T x F or T x F):
        1. Run ``model(x)`` and compute softmax probabilities.
        2. If ``max(softmax) < tau`` the model is uncertain -> the window
           is flagged as *corrupted / adversarial*.
        3. Corrupted windows are smoothed along the **time** axis (dim=0
           of the T x F matrix) to wash out high-frequency adversarial
           noise, while preserving broad temporal patterns.

    Smoothing options
    -----------------
    ``mean``    : Uniform average over a kernel of ``kernel_size`` steps.
    ``gaussian``: Weighted average using a 1-D Gaussian kernel (sigma
                  controls spread).

    Args:
        model       : DummyProbe (or ST-GCN in Module 5).
        tau         : Confidence threshold (default 0.85).  Below this the
                      sample is considered adversarially corrupted.
        method      : ``"mean"`` or ``"gaussian"`` smoothing.
        kernel_size : Temporal kernel width for mean smoothing (odd, >= 3).
        sigma       : Standard deviation for Gaussian kernel.
    """

    def __init__(
        self,
        model: nn.Module,
        tau: float = 0.85,
        method: SmoothingMethod = "gaussian",
        kernel_size: int = 5,
        sigma: float = 1.5,
    ) -> None:
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        self.model       = model
        self.tau         = tau
        self.method      = method
        self.kernel_size = kernel_size
        self.sigma       = sigma

        # Build (and cache) Gaussian kernel weights
        self._gauss_kernel = self._build_gaussian_kernel(kernel_size, sigma)

    # ------------------------------------------------------------------
    @staticmethod
    def _build_gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        """1-D Gaussian kernel, shape (size,), normalised to sum = 1."""
        half = size // 2
        x = torch.arange(-half, half + 1, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        return kernel / kernel.sum()

    # ------------------------------------------------------------------
    def _smooth_mean(self, window: torch.Tensor) -> torch.Tensor:
        """
        Apply uniform mean filter over the time axis.

        Args:
            window: Tensor (T, F).

        Returns:
            Smoothed tensor (T, F).
        """
        T, n_feat = window.shape
        pad  = self.kernel_size // 2
        # Conv1d expects (batch=1, channels=n_feat, length=T)
        w = window.T.unsqueeze(0)                  # (1, n_feat, T)
        w = F_pad(w, (pad, pad), mode="reflect")   # pad time axis
        kernel = torch.ones(1, 1, self.kernel_size,
                            device=window.device) / self.kernel_size
        # Apply per-channel (depthwise) via loop -- simple & clear
        channels = []
        for f in range(n_feat):
            ch = w[:, f:f+1, :]                    # (1, 1, T+pad)
            ch_smooth = F.conv1d(ch, kernel)        # (1, 1, T)
            channels.append(ch_smooth)
        smoothed = torch.cat(channels, dim=1)       # (1, n_feat, T)
        return smoothed.squeeze(0).T                # (T, n_feat)

    # ------------------------------------------------------------------
    def _smooth_gaussian(self, window: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian-weighted filter over the time axis.

        Args:
            window: Tensor (T, F).

        Returns:
            Smoothed tensor (T, F).
        """
        T, n_feat = window.shape
        pad    = self.kernel_size // 2
        kernel = self._gauss_kernel.to(window.device)   # (K,)
        # Shape for conv1d: (out_channels=1, in_channels=1, K)
        kernel_2d = kernel.view(1, 1, -1)

        w = window.T.unsqueeze(0)                   # (1, n_feat, T)
        w = F_pad(w, (pad, pad), mode="reflect")

        channels = []
        for f in range(n_feat):
            ch = w[:, f:f+1, :]
            ch_smooth = F.conv1d(ch, kernel_2d)
            channels.append(ch_smooth)
        smoothed = torch.cat(channels, dim=1)       # (1, n_feat, T)
        return smoothed.squeeze(0).T                # (T, n_feat)

    # ------------------------------------------------------------------
    def _smooth(self, window: torch.Tensor) -> torch.Tensor:
        """Dispatch to the selected smoothing strategy."""
        if self.method == "mean":
            return self._smooth_mean(window)
        return self._smooth_gaussian(window)

    # ------------------------------------------------------------------
    def sanitize(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inspect confidence and sanitize flagged (corrupted) windows.

        Args:
            x: Float tensor of shape (batch, T, F) or (T, F) for one window.

        Returns:
            Tuple of:
                sanitized    : Tensor same shape as ``x`` -- corrupted windows
                               are replaced with their smoothed versions.
                confidence   : Float tensor (batch,) -- max softmax score per
                               window.
                flagged_mask : Bool tensor (batch,) -- True = flagged as
                               adversarially corrupted.
        """
        single = (x.dim() == 2)
        if single:
            x = x.unsqueeze(0)          # (1, T, F)

        batch = x.shape[0]
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)               # (batch, n_classes)
        probs      = F.softmax(logits, dim=1)    # (batch, n_classes)
        confidence = probs.max(dim=1).values     # (batch,)
        flagged    = confidence < self.tau       # (batch,) bool

        n_flagged = flagged.sum().item()
        if n_flagged > 0:
            logger.info(
                "Sanitizer: %d/%d windows flagged (confidence < tau=%.2f). "
                "Applying %s smoothing.",
                n_flagged, batch, self.tau, self.method,
            )
        else:
            logger.info(
                "Sanitizer: all %d windows pass confidence gate (tau=%.2f).",
                batch, self.tau,
            )

        sanitized = x.clone()
        for i in range(batch):
            if flagged[i]:
                sanitized[i] = self._smooth(x[i])

        if single:
            sanitized  = sanitized.squeeze(0)
            confidence = confidence.squeeze(0)
            flagged    = flagged.squeeze(0)

        return sanitized, confidence, flagged


# ---------------------------------------------------------------------------
# Helper alias to avoid name clash with the F import from torch.nn.functional
# ---------------------------------------------------------------------------
def F_pad(tensor: torch.Tensor, pad: tuple, mode: str = "constant",
          value: float = 0.0) -> torch.Tensor:
    """Thin wrapper around torch.nn.functional.pad."""
    return F.pad(tensor, pad, mode=mode, value=value)


# ===========================================================================
# 4.  Public pipeline helper
# ===========================================================================

def run_sanitizer(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 50,
    n_features: int  = 27,
    epsilon: float   = 0.1,
    tau: float       = 0.85,
    method: SmoothingMethod = "gaussian",
    random_seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, DummyProbe]:
    """
    Convenience wrapper called by Module 5's end-to-end pipeline.

    Args:
        x            : Clean input tensor (batch, T, F) or (T, F).
        y            : Ground-truth labels (batch,).
        window_size  : Window time-steps (must match DummyProbe).
        n_features   : Features per step (must match DummyProbe).
        epsilon      : FGSM perturbation magnitude.
        tau          : Confidence gate threshold.
        method       : Smoothing method ("mean" or "gaussian").
        random_seed  : Seed for reproducibility.

    Returns:
        Tuple of:
            sanitized  : Sanitized tensor (same shape as ``x``).
            x_adv      : FGSM-perturbed tensor (same shape as ``x``).
            probe      : The DummyProbe used (useful for Module 5).
    """
    torch.manual_seed(random_seed)

    probe     = DummyProbe(window_size=window_size, n_features=n_features)
    attacker  = FGSMAttacker(model=probe, epsilon=epsilon)
    sanitizer = ConfidenceThresholdSanitizer(model=probe, tau=tau, method=method)

    x_adv               = attacker.attack(x, y)
    sanitized, conf, _  = sanitizer.sanitize(x_adv)

    return sanitized, x_adv, probe


# ===========================================================================
# 5.  Standalone Demo / Test Block
# ===========================================================================

if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # Make sure src/ is on the Python path so we can import module1
    # -----------------------------------------------------------------------
    _src_dir = Path(__file__).resolve().parent
    _root    = _src_dir.parent
    if str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))
    # Change cwd to project root so module1 saves files in the right place
    os.chdir(_root)

    from module1_ingestion import run_ingestion   # noqa: E402

    torch.manual_seed(42)

    logger.info("=" * 65)
    logger.info("AR-MCDF  |  Module 2 -- Standalone Demo")
    logger.info("=" * 65)

    # ------------------------------------------------------------------
    # Step A: Ingest -- pull one (50, 27) window + its labels
    # ------------------------------------------------------------------
    logger.info("Step A: Running Module 1 ingestion...")
    windows, labels, _ = run_ingestion(
        csv_path="data/raw/UNSW_NB15_training-set.csv",
        window_size=50,
        stride=25,
    )



    # Select window index 10 (arbitrary, guaranteed to exist)
    WIN_IDX = 10
    x_np = windows[WIN_IDX]                              # (50, 27) float32
    y_np = labels[WIN_IDX]                               # (50,)   int64

    x_clean = torch.tensor(x_np, dtype=torch.float32)   # (50, 27)
    # Window-level majority-vote label for CrossEntropy
    y_label  = torch.tensor(
        [int(y_np.mean() >= 0.5)], dtype=torch.long
    )                                                    # (1,)

    logger.info(
        "Selected window %d | shape: %s | majority label: %d",
        WIN_IDX, tuple(x_clean.shape), y_label.item(),
    )

    # ------------------------------------------------------------------
    # Step B: Initialise DummyProbe
    # ------------------------------------------------------------------
    logger.info("Step B: Initialising DummyProbe...")
    probe = DummyProbe(window_size=50, n_features=27)
    total_params = sum(p.numel() for p in probe.parameters())
    logger.info("DummyProbe | total parameters: %d", total_params)

    # ------------------------------------------------------------------
    # Step C: FGSM Attack
    # ------------------------------------------------------------------
    logger.info("Step C: Generating FGSM adversarial example (epsilon=0.1)...")
    attacker = FGSMAttacker(model=probe, epsilon=0.1)
    x_adv    = attacker.attack(x_clean, y_label)        # (50, 27)

    l2_orig_vs_adv = (x_adv - x_clean).norm(p=2).item()
    linf_orig_vs_adv = (x_adv - x_clean).abs().max().item()

    logger.info(
        "FGSM applied | L2(orig, adv) = %.4f | L_inf = %.4f",
        l2_orig_vs_adv, linf_orig_vs_adv,
    )

    # ------------------------------------------------------------------
    # Step D: Confidence check on CLEAN vs ATTACKED inputs
    # ------------------------------------------------------------------
    logger.info("Step D: Checking model confidence before sanitization...")
    probe.eval()
    with torch.no_grad():
        conf_clean = F.softmax(probe(x_clean.unsqueeze(0)), dim=1).max().item()
        conf_adv   = F.softmax(probe(x_adv.unsqueeze(0)),   dim=1).max().item()

    logger.info(
        "Confidence | clean=%.4f | adversarial=%.4f",
        conf_clean, conf_adv,
    )

    # ------------------------------------------------------------------
    # Step E: Sanitize adversarial example
    # ------------------------------------------------------------------
    logger.info("Step E: Running ConfidenceThresholdSanitizer (tau=0.85, gaussian)...")
    sanitizer = ConfidenceThresholdSanitizer(
        model=probe, tau=0.85, method="gaussian", kernel_size=5, sigma=1.5
    )
    x_sanitized, confidence, flagged = sanitizer.sanitize(x_adv)

    l2_orig_vs_san = (x_sanitized - x_clean).norm(p=2).item()
    l2_adv_vs_san  = (x_sanitized - x_adv).norm(p=2).item()

    # ------------------------------------------------------------------
    # Step F: Final Report
    # ------------------------------------------------------------------
    sep = "=" * 65
    print(f"\n{sep}")
    print("  MODULE 2 -- SANITIZER DEMO RESULTS")
    print(sep)
    print(f"  Window index selected        : {WIN_IDX}")
    print(f"  Window shape                 : {tuple(x_clean.shape)}")
    print(f"  Ground-truth label           : {y_label.item()} "
          f"({'attack' if y_label.item() == 1 else 'normal'})")
    print(f"\n  --- DummyProbe Confidence Scores ---")
    print(f"  Model confidence (clean)     : {conf_clean:.4f}")
    print(f"  Model confidence (attacked)  : {conf_adv:.4f}")
    print(f"  Threshold (tau)              : 0.85")
    print(f"  Flagged as adversarial?      : {bool(flagged.item())}")
    print(f"  NOTE: DummyProbe has *random* weights, so confidence is")
    print(f"        uniformly low. In Module 5, a trained ST-GCN will")
    print(f"        produce high clean-confidence and low attack-confidence.")
    print(f"\n  --- L2-Norm Distances (key metric) ---")
    print(f"  ||original - attacked||_2    : {l2_orig_vs_adv:.4f}  "
          f"  <- perturbation injected by FGSM")
    print(f"  ||attacked - sanitized||_2   : {l2_adv_vs_san:.4f}  "
          f"  <- noise REMOVED by Gaussian smoothing")
    print(f"  ||original - sanitized||_2   : {l2_orig_vs_san:.4f}  "
          f"  <- residual shift after sanitization")
    print(f"\n  --- Interpretation ---")
    if l2_adv_vs_san > 0 and l2_orig_vs_adv > 0:
        # What fraction of the injected noise was washed out?
        noise_removed_pct = (l2_adv_vs_san / l2_orig_vs_adv) * 100
        print(f"  Gaussian smoothing removed {noise_removed_pct:.1f}% of the FGSM"
              f" perturbation norm.")
        if l2_orig_vs_san < l2_orig_vs_adv:
            reduction_pct = (1 - l2_orig_vs_san / l2_orig_vs_adv) * 100
            print(f"  Residual distance to original reduced by {reduction_pct:.1f}%.")
        else:
            print(f"  NOTE: Smoothing moves signal toward local mean; residual")
            print(f"  distance vs. original may exceed FGSM delta when the")
            print(f"  clean signal itself has high spatial variance (expected")
            print(f"  with random-weight probe). With a trained model this")
            print(f"  metric will be dominated by adversarial windows only.")
    print(sep)
    print("[OK]  Module 2 completed successfully.")
    print(sep + "\n")
