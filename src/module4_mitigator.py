"""
=============================================================================
AR-MCDF  |  Module 4 -- Q-Learning Threat Mitigator
=============================================================================
Purpose:
    - CloudMitigationEnv   : Simulated cloud security environment.
                            Computes the Isolation Score
                            Is = beta1 * Threat + beta2 * Resource
                            and returns shaped rewards based on whether the
                            chosen mitigation action is proportional to the
                            combined threat/resource severity.
    - QLearningBouncer     : Tabular Q-Learning agent with epsilon-greedy
                            exploration and the standard Bellman update:
                            Q(s,a) <- Q(s,a) + alpha[r + gamma*max_a'Q(s',a') - Q(s,a)]
    - PolicyPrinter        : Pretty-prints the learned Q-table as a 2-D
                            policy grid (Threat bins x Resource bins).

Standalone execution:
    py src/module4_mitigator.py

Output:
    Training progress log, per-episode reward curve stats, and a printed
    policy table proving the agent learned proportional response:
        Low Threat              -> Throttle
        Medium Threat           -> Reroute
        High Threat + Low  Res  -> Isolate
        High Threat + High Res  -> Blacklist
=============================================================================
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

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
# 1.  Constants & Action Definitions
# ===========================================================================

# Action index -> human-readable label + severity level
ACTION_LABELS: Dict[int, str] = {
    0: "Throttle   (Gentle)",
    1: "Reroute    (Honeypot)",
    2: "Isolate    (Container Lock)",
    3: "Blacklist  (Firewall Ban)",
}

ACTION_SHORT: Dict[int, str] = {
    0: "THROTTLE",
    1: "REROUTE ",
    2: "ISOLATE ",
    3: "BLACKLST",
}

N_ACTIONS = 4

# Isolation Score coefficients  (beta1 weights Threat higher than Resource)
BETA1 = 0.65   # Threat weight
BETA2 = 0.35   # Resource (asset value) weight


# ===========================================================================
# 2.  State Discretisation Helper
# ===========================================================================

@dataclass
class StateDiscretiser:
    """
    Maps continuous (Threat_Level, Resource_Value) in [0,1] x [0,1] into a
    flat integer state index for the tabular Q-table.

    The 2-D grid has ``n_threat_bins`` rows and ``n_resource_bins`` columns.
    State index = threat_bin * n_resource_bins + resource_bin.

    Args:
        n_threat_bins   : Number of Threat_Level discretisation bins.
        n_resource_bins : Number of Resource_Value discretisation bins.
    """
    n_threat_bins:   int = 4
    n_resource_bins: int = 4

    @property
    def n_states(self) -> int:
        return self.n_threat_bins * self.n_resource_bins

    def discretise(self, threat: float, resource: float) -> int:
        """
        Convert continuous (threat, resource) to a flat state index.

        Args:
            threat   : Threat level in [0.0, 1.0] (e.g. ST-GCN attack prob).
            resource : Resource/asset value in [0.0, 1.0].

        Returns:
            state_idx: Integer in [0, n_states - 1].
        """
        t_bin = min(int(threat   * self.n_threat_bins),   self.n_threat_bins   - 1)
        r_bin = min(int(resource * self.n_resource_bins), self.n_resource_bins - 1)
        return t_bin * self.n_resource_bins + r_bin

    def bin_labels(self) -> Tuple[List[str], List[str]]:
        """Return human-readable bin labels for policy printing."""
        threat_labels   = [
            f"T{i}[{i/self.n_threat_bins:.2f}-{(i+1)/self.n_threat_bins:.2f})"
            for i in range(self.n_threat_bins)
        ]
        resource_labels = [
            f"R{j}[{j/self.n_resource_bins:.2f}-{(j+1)/self.n_resource_bins:.2f})"
            for j in range(self.n_resource_bins)
        ]
        return threat_labels, resource_labels

    def state_to_bins(self, state_idx: int) -> Tuple[int, int]:
        """Inverse of discretise: state_idx -> (threat_bin, resource_bin)."""
        t_bin = state_idx // self.n_resource_bins
        r_bin = state_idx  % self.n_resource_bins
        return t_bin, r_bin

    def bin_midpoints(self, state_idx: int) -> Tuple[float, float]:
        """Return (threat_mid, resource_mid) for a given state index."""
        t_bin, r_bin = self.state_to_bins(state_idx)
        t_mid = (t_bin + 0.5) / self.n_threat_bins
        r_mid = (r_bin + 0.5) / self.n_resource_bins
        return t_mid, r_mid


# ===========================================================================
# 3.  Reward Logic
# ===========================================================================

@dataclass
class RewardConfig:
    """
    Reward shaping configuration for the CloudMitigationEnv.

    Proportional response boundaries
    ---------------------------------
    The Isolation Score  Is = beta1*Threat + beta2*Resource  maps to an
    expected action severity:
        Is < low_thresh                     -> Action 0 (Throttle)   is correct
        low_thresh  <= Is < mid_thresh      -> Action 1 (Reroute)    is correct
        mid_thresh  <= Is < high_thresh     -> Action 2 (Isolate)    is correct
        Is >= high_thresh                   -> Action 3 (Blacklist)  is correct

    Reward schedule
    ---------------
    Exact match                 : +10   (proportional response)
    One step away from optimal  : +3    (slight over/under-reaction)
    Two steps from optimal      : -5    (significant mismatch)
    Three steps from optimal    : -10   (complete over/under-reaction)
    """
    low_thresh:  float = 0.25    # Is boundary Throttle  -> Reroute
    mid_thresh:  float = 0.50    # Is boundary Reroute   -> Isolate
    high_thresh: float = 0.75    # Is boundary Isolate   -> Blacklist

    # Reward table indexed by abs(action - optimal_action)
    reward_by_delta: List[float] = field(
        default_factory=lambda: [10.0, 3.0, -5.0, -10.0]
    )

    def optimal_action(self, isolation_score: float) -> int:
        """Return the ideal action index for a given Isolation Score."""
        if isolation_score < self.low_thresh:
            return 0   # Throttle
        elif isolation_score < self.mid_thresh:
            return 1   # Reroute
        elif isolation_score < self.high_thresh:
            return 2   # Isolate
        else:
            return 3   # Blacklist

    def compute_reward(self, action: int, isolation_score: float) -> float:
        """
        Compute shaped reward.

        Args:
            action          : Agent's chosen action index.
            isolation_score : Is = beta1*Threat + beta2*Resource.

        Returns:
            reward: Float reward value.
        """
        optimal = self.optimal_action(isolation_score)
        delta   = abs(action - optimal)
        return self.reward_by_delta[delta]


# ===========================================================================
# 4.  Cloud Mitigation Environment
# ===========================================================================

class CloudMitigationEnv:
    """
    Simulation environment for cloud threat mitigation episodes.

    Each episode
    ------------
    1. ``reset()``  : Sample random (Threat, Resource) pair; return initial state.
    2. ``step(a)``  : Compute Is, compute reward, sample next state, return
                    (next_state, reward, done).

    The environment is episodic with a single decision step per episode
    (stateless Markov Bandit).  Multi-step episodes can be enabled by
    setting ``max_steps > 1``; the state then transitions stochastically.

    Args:
        discretiser : StateDiscretiser for the state space.
        reward_cfg  : RewardConfig for shaping rewards.
        beta1       : Threat weight in Isolation Score.
        beta2       : Resource weight in Isolation Score.
        max_steps   : Steps per episode (default 1 for bandit-style).
        seed        : Optional RNG seed.
    """

    def __init__(
        self,
        discretiser: StateDiscretiser,
        reward_cfg: Optional[RewardConfig] = None,
        beta1: float = BETA1,
        beta2: float = BETA2,
        max_steps: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        self.disc       = discretiser
        self.reward_cfg = reward_cfg or RewardConfig()
        self.beta1      = beta1
        self.beta2      = beta2
        self.max_steps  = max_steps
        self.rng        = np.random.default_rng(seed)

        self._threat:   float = 0.0
        self._resource: float = 0.0
        self._step_num: int   = 0

    # ------------------------------------------------------------------
    def reset(self) -> Tuple[int, float, float]:
        """
        Sample a new (Threat, Resource) scenario and return the initial state.

        Returns:
            state_idx : Discretised state index.
            threat    : Continuous Threat_Level in [0, 1].
            resource  : Continuous Resource_Value in [0, 1].
        """
        self._threat   = float(self.rng.uniform(0.0, 1.0))
        self._resource = float(self.rng.uniform(0.0, 1.0))
        self._step_num = 0
        state = self.disc.discretise(self._threat, self._resource)
        return state, self._threat, self._resource

    # ------------------------------------------------------------------
    def step(
        self,
        action: int,
        threat: Optional[float] = None,
        resource: Optional[float] = None,
    ) -> Tuple[int, float, bool]:
        """
        Apply an action and observe the next state and reward.

        Args:
            action   : Integer action index in {0, 1, 2, 3}.
            threat   : Override current threat (useful in pipeline, default None).
            resource : Override current resource (default None).

        Returns:
            next_state : Discretised next-state index.
            reward     : Shaped float reward.
            done       : True when episode ends.
        """
        if threat   is not None: self._threat   = float(threat)
        if resource is not None: self._resource = float(resource)

        isolation_score = self.beta1 * self._threat + self.beta2 * self._resource
        reward          = self.reward_cfg.compute_reward(action, isolation_score)

        self._step_num += 1
        done = self._step_num >= self.max_steps

        # Next state: resample slightly perturbed values (Markov transition)
        next_threat   = float(np.clip(
            self._threat   + self.rng.normal(0, 0.05), 0.0, 1.0
        ))
        next_resource = float(np.clip(
            self._resource + self.rng.normal(0, 0.05), 0.0, 1.0
        ))
        next_state = self.disc.discretise(next_threat, next_resource)

        self._threat   = next_threat
        self._resource = next_resource

        return next_state, reward, done

    # ------------------------------------------------------------------
    def isolation_score(self) -> float:
        """Return Is for the current threat/resource values."""
        return self.beta1 * self._threat + self.beta2 * self._resource


# ===========================================================================
# 5.  Q-Learning Agent
# ===========================================================================

class QLearningBouncer:
    """
    Tabular Q-Learning agent for discrete state-action cloud mitigation.

    Q-Update (Bellman equation):
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_{a'} Q(s', a') - Q(s, a)]

    Exploration: epsilon-greedy with exponential decay
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    Args:
        n_states     : Total number of discrete states.
        n_actions    : Total number of discrete actions (default 4).
        alpha        : Learning rate (default 0.1).
        gamma        : Discount factor (default 0.9).
        epsilon      : Initial exploration rate (default 1.0).
        epsilon_min  : Minimum exploration floor (default 0.01).
        epsilon_decay: Multiplicative decay per episode (default 0.9995).
        seed         : Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        n_states:      int   = 16,
        n_actions:     int   = N_ACTIONS,
        alpha:         float = 0.10,
        gamma:         float = 0.90,
        epsilon:       float = 1.0,
        epsilon_min:   float = 0.01,
        epsilon_decay: float = 0.9995,
        seed:          Optional[int] = None,
    ) -> None:
        self.n_states      = n_states
        self.n_actions     = n_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng           = random.Random(seed)

        # Q-table: initialised to small random values to break ties
        rng_np = np.random.default_rng(seed)
        self.Q: np.ndarray = rng_np.uniform(
            low=-0.01, high=0.01, size=(n_states, n_actions)
        ).astype(np.float64)

        # Training history
        self.episode_rewards: List[float] = []
        self.epsilon_history: List[float] = []

    # ------------------------------------------------------------------
    def choose_action(self, state: int) -> int:
        """
        Epsilon-greedy action selection.

        Args:
            state: Current discrete state index.

        Returns:
            action: Chosen action index in {0, ..., n_actions-1}.
        """
        if self.rng.random() < self.epsilon:
            return self.rng.randint(0, self.n_actions - 1)   # explore
        return int(np.argmax(self.Q[state]))                  # exploit

    # ------------------------------------------------------------------
    def update(
        self,
        state:      int,
        action:     int,
        reward:     float,
        next_state: int,
    ) -> float:
        """
        Apply the Bellman Q-update equation.

        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_{a'} Q(s',a') - Q(s,a)]

        Args:
            state      : Current state index.
            action     : Action taken.
            reward     : Reward received.
            next_state : Resulting next-state index.

        Returns:
            td_error: Temporal-difference error (useful for diagnostics).
        """
        best_next_q = np.max(self.Q[next_state])
        td_target   = reward + self.gamma * best_next_q
        td_error    = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        return float(td_error)

    # ------------------------------------------------------------------
    def decay_epsilon(self) -> None:
        """Apply one step of epsilon decay (called once per episode)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    def greedy_policy(self) -> np.ndarray:
        """
        Return the greedy policy (argmax Q per state).

        Returns:
            policy: ndarray of shape (n_states,) with action indices.
        """
        return np.argmax(self.Q, axis=1)

    # ------------------------------------------------------------------
    def train(
        self,
        env: CloudMitigationEnv,
        n_episodes: int = 8_000,
        log_interval: int = 1_000,
    ) -> List[float]:
        """
        Full training loop.

        Args:
            env          : CloudMitigationEnv instance.
            n_episodes   : Number of episodes to train for.
            log_interval : Log progress every N episodes.

        Returns:
            episode_rewards: List of total rewards per episode.
        """
        logger.info(
            "Q-Learning training | episodes=%d | alpha=%.3f | gamma=%.3f | "
            "epsilon_start=%.2f | epsilon_decay=%.5f",
            n_episodes, self.alpha, self.gamma,
            self.epsilon, self.epsilon_decay,
        )

        for ep in range(1, n_episodes + 1):
            state, _, _ = env.reset()
            total_reward = 0.0
            done = False

            while not done:
                action               = self.choose_action(state)
                next_state, reward, done = env.step(action)
                self.update(state, action, reward, next_state)
                state        = next_state
                total_reward += reward

            self.decay_epsilon()
            self.episode_rewards.append(total_reward)
            self.epsilon_history.append(self.epsilon)

            if ep % log_interval == 0:
                window = self.episode_rewards[-log_interval:]
                logger.info(
                    "Episode %6d/%d | avg_reward=%.3f | epsilon=%.4f",
                    ep, n_episodes,
                    float(np.mean(window)),
                    self.epsilon,
                )

        logger.info(
            "Training complete | final epsilon=%.4f | "
            "last-%d avg reward=%.3f",
            self.epsilon, log_interval,
            float(np.mean(self.episode_rewards[-log_interval:])),
        )
        return self.episode_rewards


# ===========================================================================
# 6.  Policy Printer
# ===========================================================================

class PolicyPrinter:
    """
    Pretty-prints the learned Q-table as a 2-D policy grid and
    per-state Q-value table.
    """

    def __init__(self, agent: QLearningBouncer, disc: StateDiscretiser) -> None:
        self.agent = agent
        self.disc  = disc

    # ------------------------------------------------------------------
    def print_policy_grid(self) -> None:
        """Print the greedy policy as a Threat x Resource matrix."""
        policy        = self.agent.greedy_policy()
        t_labels, r_labels = self.disc.bin_labels()
        n_t = self.disc.n_threat_bins
        n_r = self.disc.n_resource_bins

        sep = "=" * (14 + 14 * n_r)
        print(f"\n{sep}")
        print("  LEARNED POLICY GRID  (rows=Threat, cols=Resource Value)")
        print(sep)

        # Header row
        row_header = "Threat \\ Resource"
        header = f"  {row_header:15s}"
        for rl in r_labels:
            header += f"  {rl[:11]:11s}"
        print(header)
        print("-" * (14 + 14 * n_r))

        for t_bin in range(n_t):
            row = f"  {t_labels[t_bin][:15]:15s}"
            for r_bin in range(n_r):
                s_idx  = t_bin * n_r + r_bin
                action = policy[s_idx]
                row   += f"  {ACTION_SHORT[action]}"
            print(row)

        print(sep)

    # ------------------------------------------------------------------
    def print_q_table(self) -> None:
        """Print the full Q-table with values for each state."""
        policy        = self.agent.greedy_policy()
        t_labels, r_labels = self.disc.bin_labels()
        n_t = self.disc.n_threat_bins
        n_r = self.disc.n_resource_bins

        col_w = 10
        sep   = "=" * (30 + col_w * N_ACTIONS + 14)
        print(f"\n{sep}")
        print("  Q-TABLE  (Q-values per state x action)")
        print(sep)

        # Action header
        hdr = f"  {'State':28s}"
        for a in range(N_ACTIONS):
            hdr += f"  {ACTION_SHORT[a]:>{col_w}}"
        hdr += f"  {'Best':>10}"
        print(hdr)
        print("-" * (30 + col_w * N_ACTIONS + 14))

        for s in range(self.disc.n_states):
            t_bin, r_bin = self.disc.state_to_bins(s)
            state_label  = f"T{t_bin}-R{r_bin} Is~{self.disc.bin_midpoints(s)[0]*BETA1 + self.disc.bin_midpoints(s)[1]*BETA2:.2f}"
            row = f"  {state_label:28s}"
            for a in range(N_ACTIONS):
                row += f"  {self.agent.Q[s, a]:>{col_w}.4f}"
            row += f"  {ACTION_SHORT[policy[s]]:>10}"
            print(row)

        print(sep)

    # ------------------------------------------------------------------
    def print_reward_stats(self) -> None:
        """Print summary statistics on training rewards."""
        rewards = np.array(self.agent.episode_rewards)
        n       = len(rewards)
        w       = min(1000, n // 5)

        print(f"\n  --- Training Reward Statistics ---")
        print(f"  Total episodes          : {n:,}")
        print(f"  First {w}-ep avg reward  : {rewards[:w].mean():.3f}")
        print(f"  Last  {w}-ep avg reward  : {rewards[-w:].mean():.3f}")
        print(f"  Global min / max        : {rewards.min():.1f} / {rewards.max():.1f}")
        print(f"  Final epsilon           : {self.agent.epsilon:.6f}")


# ===========================================================================
# 7.  Public Pipeline Helper
# ===========================================================================

def run_mitigator(
    threat_level:   float,
    resource_value: float,
    agent:          Optional[QLearningBouncer] = None,
    disc:           Optional[StateDiscretiser] = None,
) -> Tuple[int, str, float]:
    """
    Convenience function called by Module 5.

    Given a (threat_level, resource_value) pair from the ST-GCN output,
    return the mitigator's recommended action.

    Args:
        threat_level   : Float in [0, 1] from ST-GCN softmax.
        resource_value : Float in [0, 1] from cloud node metadata.
        agent          : Pre-trained QLearningBouncer (required in pipeline).
        disc           : StateDiscretiser matching the agent's table size.

    Returns:
        action_idx  : Integer action index.
        action_name : Human-readable action label.
        iso_score   : Computed Isolation Score.
    """
    if agent is None or disc is None:
        raise ValueError(
            "run_mitigator() requires a trained agent and matching discretiser. "
            "Call train() first in Module 5."
        )
    state     = disc.discretise(threat_level, resource_value)
    action    = int(np.argmax(agent.Q[state]))
    iso_score = BETA1 * threat_level + BETA2 * resource_value
    return action, ACTION_LABELS[action], iso_score


# ===========================================================================
# 8.  Standalone Demo / Training Block
# ===========================================================================

if __name__ == "__main__":
    import sys

    SEP = "=" * 68

    logger.info(SEP)
    logger.info("AR-MCDF  |  Module 4 -- Q-Learning Mitigator Demo")
    logger.info(SEP)

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------
    N_THREAT_BINS   = 4
    N_RESOURCE_BINS = 4
    N_EPISODES      = 8_000
    ALPHA           = 0.10
    GAMMA           = 0.90
    EPSILON_START   = 1.00
    EPSILON_MIN     = 0.01
    EPSILON_DECAY   = 0.9994   # reaches ~0.02 by ep 8000

    # ------------------------------------------------------------------
    # Step A: Initialise components
    # ------------------------------------------------------------------
    logger.info("Step A: Initialising State Space, Environment, and Agent...")

    disc    = StateDiscretiser(n_threat_bins=N_THREAT_BINS, n_resource_bins=N_RESOURCE_BINS)
    r_cfg   = RewardConfig(low_thresh=0.25, mid_thresh=0.50, high_thresh=0.75)
    env     = CloudMitigationEnv(discretiser=disc, reward_cfg=r_cfg, seed=42)
    agent   = QLearningBouncer(
        n_states      = disc.n_states,
        n_actions     = N_ACTIONS,
        alpha         = ALPHA,
        gamma         = GAMMA,
        epsilon       = EPSILON_START,
        epsilon_min   = EPSILON_MIN,
        epsilon_decay = EPSILON_DECAY,
        seed          = 42,
    )

    logger.info(
        "State space: %d bins (%dx%d) | Action space: %d",
        disc.n_states, N_THREAT_BINS, N_RESOURCE_BINS, N_ACTIONS,
    )
    logger.info("Actions:")
    for idx, label in ACTION_LABELS.items():
        logger.info("  [%d] %s", idx, label)

    # ------------------------------------------------------------------
    # Step B: Training loop
    # ------------------------------------------------------------------
    logger.info("Step B: Training Q-Learning agent for %d episodes...", N_EPISODES)
    rewards = agent.train(env, n_episodes=N_EPISODES, log_interval=1_000)

    # ------------------------------------------------------------------
    # Step C: Print learned policy
    # ------------------------------------------------------------------
    printer = PolicyPrinter(agent, disc)

    print(f"\n{SEP}")
    print("  MODULE 4 -- Q-LEARNING MITIGATOR RESULTS")
    print(SEP)

    printer.print_reward_stats()
    printer.print_policy_grid()
    printer.print_q_table()

    # ------------------------------------------------------------------
    # Step D: Spot-check specific (Threat, Resource) scenarios
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("  SPOT-CHECK: Policy Decisions on Key Scenarios")
    print(SEP)
    print(f"  {'Scenario':<35} {'Threat':>7}  {'Resource':>8}  {'Is':>6}  {'Action'}")
    print(f"  {'-'*35} {'-'*7}  {'-'*8}  {'-'*6}  {'-'*28}")

    scenarios = [
        ("Low threat,  low  resource",  0.10, 0.10),
        ("Low threat,  high resource",  0.10, 0.90),
        ("Med threat,  low  resource",  0.40, 0.20),
        ("Med threat,  high resource",  0.50, 0.80),
        ("High threat, low  resource",  0.80, 0.15),
        ("High threat, high resource",  0.90, 0.95),
        ("Max threat,  max  resource",  1.00, 1.00),
    ]

    for name, threat, resource in scenarios:
        state     = disc.discretise(threat, resource)
        action    = int(np.argmax(agent.Q[state]))
        iso       = BETA1 * threat + BETA2 * resource
        print(f"  {name:<35} {threat:>7.2f}  {resource:>8.2f}  "
            f"{iso:>6.3f}  {ACTION_LABELS[action]}")

    print(SEP)
    print("[OK]  Module 4 completed successfully.")
    print(SEP + "\n")