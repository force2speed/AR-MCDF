# AR-MCDF — Adversarial-Resilient Multi-Cloud Defense Framework

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange?logo=pytorch)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.4%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-UNSW--NB15-purple)
![CUDA](https://img.shields.io/badge/CUDA-Supported-76B900?logo=nvidia)

**An end-to-end cloud security framework that detects, sanitizes, and mitigates AI-augmented cyberattacks using Graph Neural Networks and Reinforcement Learning.**

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Train the Full Pipeline](#1-train-the-full-pipeline)
  - [2. Run the Live Demo](#2-run-the-live-demo)
  - [3. Launch the Dashboard](#3-launch-the-dashboard)
  - [4. Generate Figures](#4-generate-figures)
  - [5. Evaluate the Model](#5-evaluate-the-model)
- [Docker Deployment](#docker-deployment)
- [Modules In-Depth](#modules-in-depth)
- [Academic Contributions](#academic-contributions)
- [Citation](#citation)

---

## Overview

The **AR-MCDF** (Adversarial-Resilient Multi-Cloud Defense Framework) is an advanced, research-grade cloud security system designed to handle the next generation of AI-augmented cyberattacks. Unlike traditional NIDS (Network Intrusion Detection Systems) that treat traffic as flat tabular rows, AR-MCDF **represents network traffic as Spatio-Temporal Graphs**, making it structurally far harder for adversaries to spoof.

The framework is uniquely **self-stress-testing** — it ships with a built-in `WhiteBoxGraphAttacker` (FGSM) that generates adversarial perturbations against its own model, then proves it can still correctly classify the perturbed traffic through its dual-layer defense pipeline.

```
RAW TELEMETRY → [MODULE 1: INGESTION] → Normalized Windows
                                              ↓
                                    [MODULE 2: SANITIZER]  ←—— FGSM Attacker
                                     Confidence Check +           (Adversary)
                                     Gaussian Smoothing
                                              ↓
                                    [MODULE 3: ST-GCN]
                                     Graph-based Threat
                                     Classification
                                              ↓
                                    [MODULE 4: Q-BOUNCER]
                                     RL-based Automated
                                     Mitigation Response
                                              ↓
                                    Throttle / Reroute / Isolate / Blacklist
```

---

## Key Results

Trained on UNSW-NB15 over **15 epochs** on NVIDIA GTX 1650 (CUDA):

| Metric    | Clean Traffic | Adversarial (FGSM) |
|-----------|:-------------:|:------------------:|
| Accuracy  | 92.59%        | **94.26%**         |
| Precision | 96.29%        | 98.16%             |
| Recall    | 90.02%        | 91.29%             |
| F1-Score  | 93.05%        | **94.60%**         |

> **The adversarial accuracy *exceeds* clean accuracy** — demonstrating that the Sanitizer's Gaussian smoothing also acts as a mild regularizer, reducing model uncertainty even on normal traffic.

**Live Demo Results:**
- FGSM attack (ε=0.15) **successfully evaded** the unprotected path — dropping attack probability to `38.32%`
- AR-MCDF **protected path** flagged adversarial camouflage, stripped the noise, and restored classification to `ATTACK` → triggering `Isolate (Container Lock)`

---

## Architecture

The framework is composed of **5 integrated modules**:

```
┌────────────────────────────────────────────────────────────────┐
│                        AR-MCDF PIPELINE                        │
├────────────────┬───────────────────────────────────────────────┤
│ Module 1       │ Telemetry Ingestion & Preprocessing           │
│                │ Z-score normalization + sliding windows (50)  │
├────────────────┼───────────────────────────────────────────────┤
│ Module 2       │ Adversarial Sanitizer                         │
│                │ FGSM Attacker + Confidence-Threshold Gate      │
│                │ + Gaussian smoothing (σ=0.9, τ=0.85)          │
├────────────────┼───────────────────────────────────────────────┤
│ Module 3       │ Spatio-Temporal GCN (ST-GCN)                  │
│                │ 2× GCNConv → Multi-Head Attention → FC        │
│                │ Focal Loss (α=0.25, γ=2.0)                    │
├────────────────┼───────────────────────────────────────────────┤
│ Module 4       │ Q-Learning Threat Mitigator (Q-Bouncer)       │
│                │ Tabular Q-Learning, 4 actions, Bellman update  │
│                │ Is = 0.65·Threat + 0.35·Resource              │
├────────────────┼───────────────────────────────────────────────┤
│ Module 5       │ Master Pipeline (CUDA-accelerated)            │
│                │ Integrates modules 1-4, saves artifacts        │
└────────────────┴───────────────────────────────────────────────┘
```

### ST-GCN Architecture

```
Input: (window_size=50, n_features=27)  →  Graph Construction
       ↓ cosine-similarity edges (τ=0.60) or k-NN (k=4)
       ↓ GCNConv(27→128) + BatchNorm + ReLU
       ↓ GCNConv(128→128) + BatchNorm + ReLU
       ↓ Multi-Head Self-Attention (4 heads, d=128)
       ↓ Global Mean Pool + Global Max Pool (concatenated)
       ↓ FC(256→64) + Dropout(0.5) + ReLU
       ↓ FC(64→2)  →  Softmax  →  P(ATTACK)
```

### Q-Bouncer Policy (Learned)

| Isolation Score | Action              |
|:--------------:|:--------------------|
| 0.00 – 0.25    | Throttle (Gentle)   |
| 0.25 – 0.50    | Reroute (Honeypot)  |
| 0.50 – 0.75    | Isolate (Container Lock) |
| 0.75 – 1.00    | Blacklist (Firewall Ban) |

---

## Project Structure

```
AR-MCDF/
├── src/
│   ├── module1_ingestion.py     # Data loading, normalization, windowing
│   ├── module2_sanitizer.py     # FGSM attacker + confidence-threshold sanitizer
│   ├── module3_stgcn.py         # ST-GCN model + WindowGraphBuilder + FocalLoss
│   ├── module4_mitigator.py     # Q-Learning bouncer + CloudMitigationEnv
│   ├── module5_pipeline.py      # End-to-end CUDA-accelerated training pipeline
│   ├── demo_attack.py           # Live attack & defense walkthrough
│   └── evaluate.py              # Standalone model evaluation
│
├── dashboard/
│   └── server.py                # Flask REST API backend
│
├── frontend/
│   ├── src/
│   │   ├── Demo.jsx             # Main React UI component
│   │   └── App.jsx
│   ├── package.json
│   └── vite.config.js
│
├── data/
│   ├── raw/                     # UNSW-NB15 CSV files (download separately)
│   │   ├── UNSW_NB15_training-set.csv
│   │   └── UNSW_NB15_testing-set.csv
│   ├── pipeline_metrics.json    # Live training metrics (written per epoch)
│   └── evaluation_report.json  # Final evaluation results
│
├── models/                      # Saved model weights (excluded from git)
│   ├── stgcn_weights.pt         # Trained ST-GCN state dict
│   └── qtable.npy               # Trained Q-table
│
├── figures/                     # Generated paper figures
├── generate_figures.py          # Figure generation script
├── Dockerfile                   # Multi-stage Docker build
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Dataset

This project uses the **[UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)** — a comprehensive modern network intrusion dataset created by the Australian Centre for Cyber Security.

### Download

The raw dataset CSVs are **not included in this repository** due to size (>45 MB total). Download them from the official source:

**🔗 [UNSW-NB15 Official Download Page](https://research.unsw.edu.au/projects/unsw-nb15-dataset)**

Place the files in `data/raw/`:
```
data/raw/UNSW_NB15_training-set.csv    (~31 MB, ~175,341 rows)
data/raw/UNSW_NB15_testing-set.csv     (~14 MB, ~82,332 rows)
```

### Dataset Summary

| Property        | Value                                |
|-----------------|--------------------------------------|
| Total Records   | ~257,673 (train + test)              |
| Attack Types    | DoS, Fuzzers, Reconnaissance, Backdoors, Analysis, Shellcode, Worms, Generic, Exploits |
| Numerical Features Used | 27 (dur, sbytes, dbytes, rate, sttl, dttl, sload, dload, ...) |
| Label           | Binary: 0 = Normal, 1 = Attack       |
| Window Size     | 50 packets (sliding, stride=25)      |

The `data/NUSW-NB15_features.csv` file included in this repo describes all 49 original dataset features.

---

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (optional, CPU fallback available)
- Node.js 20+ (for frontend)

### 1. Clone the Repository

```bash
git clone https://github.com/force2speed/AR-MCDF.git
cd AR-MCDF
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

For CUDA-accelerated PyTorch (replace `cpu` with your CUDA version, e.g. `cu121`):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

For PyTorch Geometric scatter/sparse extensions:
```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env if needed
```

### 4. Download Dataset

Place UNSW-NB15 CSVs in `data/raw/` as described in the [Dataset](#dataset) section.

---

## Usage

### 1. Train the Full Pipeline

Runs all 5 modules end-to-end. Trains the ST-GCN for 15 epochs on UNSW-NB15 training data, simultaneously trains the Q-Bouncer, and saves `models/stgcn_weights.pt` and `models/qtable.npy`.

```bash
python src/module5_pipeline.py
```

**Expected training time:** ~50 minutes on GTX 1650 (CUDA), ~3-5 hours on CPU.

Progress is written live to `data/pipeline_metrics.json` after each epoch.

### 2. Run the Live Demo

Dramatic, step-by-step showcase of the full attack → defense cycle:

```bash
python src/demo_attack.py
```

Output walks through:
1. **LOAD** — Finds a real Attack window from UNSW-NB15 test set
2. **HACK** — FGSM injects adversarial camouflage (ε=0.15)
3. **DEFENSE** — Sanitizer detects low confidence, applies Gaussian smoothing
4. **DETECTIVE** — ST-GCN reports threat probability
5. **BOUNCER** — Q-agent selects proportional mitigation action

### 3. Launch the Dashboard

**Backend (Flask API):**
```bash
python dashboard/server.py
```
API runs at `http://localhost:8050`

**Frontend (React + Vite):**
```bash
cd frontend
npm install
npm run dev
```
UI runs at `http://localhost:5173`

### 4. Generate Figures

Generates all paper/report figures and saves them to `figures/`:

```bash
python generate_figures.py
```

Figures generated:
| Figure | Description |
|--------|-------------|
| Figure 2 | Performance comparison bar chart (Clean vs. Adversarial) |
| Figure 3 | ST-GCN Training Loss Convergence (15 epochs) |
| Figure 4 | Precision-Recall Curve |
| Figure 5 | F1-Score Progression During Training |
| Figure 6 | Q-Bouncer Reward Convergence (8,000 episodes) |
| Figure 7 | Epsilon-Greedy Exploration Decay |
| Figure 8 | Isolation Score Distribution |
| Figure 9 | Confusion Matrix — Clean Traffic |
| Figure 10 | Confusion Matrix — Adversarial Traffic |

> Figures 4, 8, 9, 10 require a trained model (`models/stgcn_weights.pt`). Run Module 5 first.

### 5. Evaluate the Model

```bash
python src/evaluate.py
```

Outputs accuracy, precision, recall, F1, and confusion matrix on the test set.

---

## Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t ar-mcdf:latest .
docker run -p 8050:8050 -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data ar-mcdf:latest
```

The container serves the full React + Flask application at `http://localhost:8050`.

---

## Modules In-Depth

### Module 1 — Telemetry Ingestion (`module1_ingestion.py`)

- **Input:** UNSW-NB15-format CSV
- **Processing:** Selects 27 numerical features → Z-score normalization (`StandardScaler`) → Sliding window segmentation (window=50, stride=25)
- **Output:** `List[np.ndarray]` of shape `(50, 27)` + binary labels
- **Standalone:** `python src/module1_ingestion.py`

### Module 2 — Adversarial Sanitizer (`module2_sanitizer.py`)

Three components:
- **`DummyProbe`**: Lightweight MLP surrogate model (`FC(1350→256→64→2)`) providing gradients for FGSM
- **`FGSMAttacker`**: Computes `x_adv = x + ε · sign(∇_x L(f(x), y))`, clips to `[-5, 5]`
- **`ConfidenceThresholdSanitizer`**: If `max(softmax(f(x_adv))) < τ=0.85` → flag as adversarial → apply Gaussian smoothing (`σ=0.9`, kernel=5)
- **Standalone:** `python src/module2_sanitizer.py`

### Module 3 — ST-GCN (`module3_stgcn.py`)

- **`WindowGraphBuilder`**: Cosine-similarity or k-NN edge construction from temporal windows
- **`SpatioTemporalGCN`**: 2× GCNConv layers → Multi-Head Self-Attention → global pooling → classification head
- **`FocalLoss`**: `FL(pt) = -αt · (1-pt)^γ · log(pt)` handles class imbalance
- **Standalone:** `python src/module3_stgcn.py`

### Module 4 — Q-Bouncer (`module4_mitigator.py`)

- **`CloudMitigationEnv`**: Simulated cloud environment with Isolation Score `Is = 0.65·Tp + 0.35·Rl`
- **`QLearningBouncer`**: Tabular Q-table, ε-greedy exploration, Bellman update: `Q(s,a) ← Q(s,a) + α[r + γ·max_{a'}Q(s',a') - Q(s,a)]`
- **Actions:** Throttle → Reroute → Isolate → Blacklist (proportional to threat level)
- **Standalone:** `python src/module4_mitigator.py`

### Module 5 — Master Pipeline (`module5_pipeline.py`)

- Integrates all 4 modules into a single reproducible training loop
- CUDA-accelerated (auto-detects GPU, falls back to CPU)
- Writes epoch metrics live to `data/pipeline_metrics.json`
- Saves `models/stgcn_weights.pt` + `models/qtable.npy`

---

## Academic Contributions

1. **Graph-Based Traffic Representation (ST-GCN):** Represents network telemetry as spatio-temporal graphs, capturing complex spatial correlations between concurrent network events — substantially harder for adversaries to spoof than tabular packet metrics.

2. **White-Box Adversarial Stress Testing (FGSM):** Built-in `WhiteBoxGraphAttacker` computes exact mathematical gradients of the trained ST-GCN and generates targeted perturbations (ε=0.15) to stress-test the system under worst-case attack scenarios.

3. **Dual-Layer Defense Architecture:**
   - **Pre-Inference (The Sanitizer):** Confidence-thresholding gate + Gaussian smoothing strips adversarial noise before classification
   - **Post-Inference (The Q-Bouncer):** RL agent dynamically selects proportional mitigation based on real-time threat and resource metrics

4. **Demonstrated Adversarial Resilience:** Adversarial accuracy (94.26%) *exceeds* clean accuracy (92.59%), proving the sanitizer's dual function as both a defense mechanism and a signal denoiser.

---

## Citation

If you use this work in your research, please cite:

```bibtex
@software{armcdf2026,
  author       = {Dhoundiyal, Dhruv},
  title        = {AR-MCDF: Adversarial-Resilient Multi-Cloud Defense Framework},
  year         = {2026},
  url          = {https://github.com/force2speed/AR-MCDF},
  note         = {ST-GCN + Q-Learning cloud intrusion detection with FGSM adversarial robustness}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built with PyTorch · PyTorch Geometric · React · Flask<br>
Dataset: <a href="https://research.unsw.edu.au/projects/unsw-nb15-dataset">UNSW-NB15</a> by Australian Centre for Cyber Security
</div>

