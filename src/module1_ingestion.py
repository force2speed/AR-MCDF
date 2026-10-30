"""
=============================================================================
AR-MCDF  |  Module 1 - Multi-Cloud Telemetry Ingestion & Preprocessing
=============================================================================
Purpose:
    - Load a UNSW-NB15-style CSV dataset.
    - Apply Z-score normalisation on numerical features.
    - Slice the normalised data into overlapping sliding time-windows that
      serve as temporal snapshots for the downstream ST-GCN (Module 3).

Standalone execution:
    python src/module1_ingestion.py

Output (when run directly):
    - Prints window shapes and a short statistics summary.
=============================================================================
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(module)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants - mirrors typical UNSW-NB15 numerical feature columns
# ---------------------------------------------------------------------------
NUMERICAL_FEATURES: List[str] = [
    "dur",          # connection duration (seconds)
    "spkts",        # source-to-dest packets
    "dpkts",        # dest-to-source packets
    "sbytes",       # source-to-dest bytes
    "dbytes",       # dest-to-source bytes
    "rate",         # connection rate (pps)
    "sttl",         # source TTL
    "dttl",         # dest TTL
    "sload",        # source bits per second
    "dload",        # dest bits per second
    "sloss",        # source packet loss
    "dloss",        # dest packet loss
    "sinpkt",       # source inter-packet arrival (ms)
    "dinpkt",       # dest inter-packet arrival (ms)
    "sjit",         # source jitter (ms)
    "djit",         # dest jitter (ms)
    "swin",         # source TCP window size
    "dwin",         # dest TCP window size
    "smean",        # mean source packet size (bytes)
    "dmean",        # mean dest packet size (bytes)
    "ct_srv_src",   # connections to same service, same src (last 100)
    "ct_state_ttl", # connections with same TTL state (last 100)
    "ct_dst_ltm",   # connections to same dest, last 100ms
    "ct_src_ltm",   # connections from same src, last 100ms
    "ct_src_dport_ltm",  # same src+dst port, last 100ms
    "ct_dst_sport_ltm",  # same dst+src port, last 100ms
    "ct_dst_src_ltm",    # same dst+src, last 100ms
]

CATEGORICAL_FEATURES: List[str] = ["proto", "service", "state"]
LABEL_COLUMN: str = "label"       # 0 = normal, 1 = attack
ATTACK_CAT_COLUMN: str = "attack_cat"

ALL_COLUMNS: List[str] = (
    NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [ATTACK_CAT_COLUMN, LABEL_COLUMN]
)


# ===========================================================================
# 1.  Ingestion & Preprocessing
# ===========================================================================

class TelemetryIngestion:
    """
    Loads and preprocesses multi-cloud telemetry in UNSW-NB15 format.

    Pipeline:
        1. load()       - load CSV from the provided path
        2. normalize()  - Z-score normalisation on numerical features
        3. window()     - sliding window segmentation

    Args:
        csv_path    : Path to the UNSW-NB15 CSV file (required).
        window_size : Number of time-steps per window (default: 50).
        stride      : Step between consecutive windows (default: 25).
        random_seed : Seed for reproducibility.
    """

    def __init__(
        self,
        csv_path: str | Path,
        window_size: int = 50,
        stride: int = 25,
        random_seed: int = 42,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.window_size = window_size
        self.stride = stride
        self.random_seed = random_seed

        self.raw_df:        pd.DataFrame | None = None
        self.normalised_df: pd.DataFrame | None = None
        self.scaler:        StandardScaler | None = None

    # ------------------------------------------------------------------
    # Step 1 - Load
    # ------------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        """
        Load telemetry from the CSV file specified at construction time.

        Returns:
            Raw DataFrame (not yet normalised).

        Raises:
            FileNotFoundError: If the CSV path does not exist.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Dataset CSV not found: '{self.csv_path}'. "
                "Ensure the UNSW-NB15 dataset is placed in data/raw/ before running."
            )

        logger.info("Loading telemetry from: %s", self.csv_path)
        df = pd.read_csv(self.csv_path, low_memory=False)

        # Harmonise column names (UNSW-NB15 uses lowercase)
        df.columns = [c.strip().lower() for c in df.columns]

        # Coerce numerical columns; fill NaN with column median
        for col in NUMERICAL_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(df[col].median())
            else:
                logger.warning("Feature '%s' not found in CSV; filling with 0.", col)
                df[col] = 0.0

        logger.info("Loaded %d rows from CSV.", len(df))
        self.raw_df = df
        return df

    # ------------------------------------------------------------------
    # Step 2 - Normalise
    # ------------------------------------------------------------------
    def normalize(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Apply Z-score (StandardScaler) normalisation to all numerical features.

        Fitted scaler is stored in ``self.scaler`` for downstream use
        (e.g. inverse-transform in the mitigator module).

        Args:
            df: DataFrame to normalise. Defaults to ``self.raw_df``.

        Returns:
            DataFrame with normalised numerical features (categoricals
            and label columns are preserved unchanged).
        """
        if df is None:
            if self.raw_df is None:
                raise RuntimeError("Call load() before normalize().")
            df = self.raw_df.copy()

        present_numerical = [c for c in NUMERICAL_FEATURES if c in df.columns]
        missing = set(NUMERICAL_FEATURES) - set(present_numerical)
        if missing:
            logger.warning("Missing numerical columns (will skip): %s", missing)

        self.scaler = StandardScaler()
        df[present_numerical] = self.scaler.fit_transform(df[present_numerical])

        logger.info(
            "Normalisation complete. Features: %d | Rows: %d",
            len(present_numerical), len(df),
        )
        self.normalised_df = df
        return df

    # ------------------------------------------------------------------
    # Step 3 - Sliding Window
    # ------------------------------------------------------------------
    def window(
        self,
        df: pd.DataFrame | None = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Segment the normalised DataFrame into overlapping time-windows.

        Each window is a 2-D array of shape (window_size, n_features).
        The label for a window is derived by **majority vote** across its rows:
        if more than half the rows are labelled attack the window is attack.

        Args:
            df: Normalised DataFrame. Defaults to ``self.normalised_df``.

        Returns:
            Tuple of:
                windows : list of np.ndarray, shape (window_size, n_features)
                labels  : list of np.ndarray, shape (window_size,) - per-row labels

        Raises:
            RuntimeError: If normalize() has not been called first.
        """
        if df is None:
            if self.normalised_df is None:
                raise RuntimeError("Call normalize() before window().")
            df = self.normalised_df

        present_numerical = [c for c in NUMERICAL_FEATURES if c in df.columns]
        feature_matrix = df[present_numerical].to_numpy(dtype=np.float32)
        label_array    = df[LABEL_COLUMN].to_numpy(dtype=np.int64) \
                         if LABEL_COLUMN in df.columns \
                         else np.zeros(len(df), dtype=np.int64)

        n_rows = len(feature_matrix)
        windows: List[np.ndarray] = []
        labels:  List[np.ndarray] = []

        start = 0
        while start + self.window_size <= n_rows:
            end = start + self.window_size
            windows.append(feature_matrix[start:end])
            labels.append(label_array[start:end])
            start += self.stride

        logger.info(
            "Windowing: %d windows | window_size=%d | stride=%d",
            len(windows), self.window_size, self.stride,
        )
        return windows, labels


# ===========================================================================
# 2.  Public API Helper
# ===========================================================================

def run_ingestion(
    csv_path: str | Path,
    window_size: int = 50,
    stride: int = 25,
) -> Tuple[List[np.ndarray], List[np.ndarray], StandardScaler]:
    """
    Convenience wrapper: execute the full ingestion pipeline in one call.

    Args:
        csv_path    : Path to the UNSW-NB15 CSV file (required).
        window_size : Time-steps per window.
        stride      : Step between windows.

    Returns:
        windows : list of np.ndarray (window_size x n_features)
        labels  : list of np.ndarray (window_size,)
        scaler  : fitted StandardScaler (for downstream inverse-transform)

    Raises:
        FileNotFoundError: If csv_path does not exist.
    """
    ingestion = TelemetryIngestion(
        csv_path=csv_path,
        window_size=window_size,
        stride=stride,
    )

    ingestion.load()
    ingestion.normalize()
    windows, labels = ingestion.window()

    return windows, labels, ingestion.scaler  # type: ignore[return-value]


# ===========================================================================
# 3.  Standalone Test / Demo
# ===========================================================================

if __name__ == "__main__":
    import sys
    import os

    # Run from project root so relative paths resolve correctly
    _root = Path(__file__).resolve().parent.parent
    os.chdir(_root)

    logger.info("=" * 60)
    logger.info("AR-MCDF  |  Module 1 - Standalone Demo")
    logger.info("=" * 60)

    windows, labels, scaler = run_ingestion(
        csv_path="data/raw/UNSW_NB15_training-set.csv",
        window_size=50,
        stride=25,
    )

    # Summary
    print("\n" + "=" * 60)
    print("  INGESTION SUMMARY")
    print("=" * 60)
    print(f"  Dataset              : UNSW_NB15_training-set.csv")
    print(f"  Total windows        : {len(windows)}")
    print(f"  Window shape (each)  : {windows[0].shape}  (time_steps x features)")
    print(f"  Label shape (each)   : {labels[0].shape}")
    print(f"  Feature count        : {windows[0].shape[1]}")

    # Per-window attack-rate statistics
    attack_rates = [lbl.mean() for lbl in labels]
    print(f"\n  Attack rate per window:")
    print(f"    min  = {min(attack_rates):.3f}")
    print(f"    max  = {max(attack_rates):.3f}")
    print(f"    mean = {sum(attack_rates)/len(attack_rates):.3f}")

    # Scaler stats
    print(f"\n  Scaler mean  (first 5 features): "
          f"{scaler.mean_[:5].round(4).tolist()}")
    print(f"  Scaler scale (first 5 features): "
          f"{scaler.scale_[:5].round(4).tolist()}")

    print("=" * 60)
    print("[OK]  Module 1 completed successfully.\n")
