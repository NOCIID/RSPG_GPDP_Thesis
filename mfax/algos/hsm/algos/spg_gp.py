"""
Structural Policy Gradient with a Gaussian-process critic.

Rollouts are truncated after `horizon` periods and the GP values the rest of the future.
See mfax/algos/hsm/utils/gp_train.py for the method and experiments/gp_critic/SUMMARY.md
for the results.

    python -m mfax.algos.hsm.algos.spg_gp                                 # time-aware GP critic, H = 32
    python -m mfax.algos.hsm.algos.spg_gp --horizon 25 --no-time-input   # first version (infinite horizon)
    python -m mfax.algos.hsm.algos.spg_gp --no-use-gp --horizon 128 \
        --phase-counts 8                                                  # identical to spg.py
"""

from dataclasses import dataclass
import os
import sys
from pathlib import Path

import tyro

# --- make repo root importable even when file is executed by path ---
_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "mfax").is_dir()), None
)
if _REPO_ROOT is not None:
    sys.path.insert(0, str(_REPO_ROOT))
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"
os.environ.pop("LD_LIBRARY_PATH", None)
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

from mfax.algos.hsm.utils.gp_train import train  # noqa: E402


@dataclass
class args:
    # --- logging ---
    debug: bool = False
    evaluate: bool = True
    log: bool = False
    save: bool = True
    wandb_project: str = "mfax"
    wandb_team: str = ""
    wandb_group: str = "mfax"
    run_name: str = ""

    # --- environment (as in configs/endogenous_spg.yaml) ---
    task: str = "endogenous"
    state_type: str = "states"
    discount_factor: float = 0.95
    normalize_obs: bool = True
    normalize_states: bool = True
    common_noise: bool = True

    # --- hsm hyperparameters ---
    algo: str = "hsm_spg_gp"
    seed: int = 0
    num_envs: int = 8
    num_iterations: int = 500
    lr: float = 0.01
    anneal_lr: bool = True
    max_grad_norm: float = 1.0
    activation: str = "relu"

    # --- truncation and restarts ---
    use_gp: bool = True
    horizon: int = 32  # 32 x 4 phases = the whole 128-period episode
    phase_counts: tuple[int, ...] = (4, 2, 1, 1)  # economy slots per window phase; sums to num_envs

    # --- GP critic ---
    time_input: bool = True  # time-aware, finite-horizon critic (time left in the episode as an input)
    n_gp: int = 1000  # training points per fit
    n_lags: int = 0  # lagged (z, K) inputs
    refit_every: int = 10
    fast_refit_steps: int = 0  # refit after every step for the first N steps (critic lags a fast-moving policy)
    fast_refit_window: int = 5  # ... on the targets of the last this-many steps
    gp_lr: float = 0.05
    gp_steps_first: int = 400
    gp_steps_refit: int = 20
    warmstart_envs: int = 32
    warmstart_keep: int = 128
    warmstart_extra: int = 128

    # --- evaluation ---
    checkpoint_every: int = 0  # save agent + critic every N steps (0 = off), for validating the critic
    eval_frequency: int = 50
    eval_envs: int = 32
    eval_seed: int = 12345


if __name__ == "__main__":
    train(tyro.cli(args), recurrent=False)
