"""
Gaussian-process dynamic programming (GPDP) for the endogenous mean-field game: no gradient descent
and no policy network. Each iteration simulates the economy under the population's policy, computes
best responses backward with one value GP per period (averaging over tomorrow's shock, continuous
consumption shares), and fits a policy GP on a blend of the population's and the best shares.
See mfax/algos/hsm/utils/gpdp_train.py and FINDINGS.md ("Design: GPDP instead of gradient descent").

    python -m mfax.algos.hsm.algos.gpdp --run-name my_run
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
os.environ.pop("LD_LIBRARY_PATH", None)
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

from mfax.algos.hsm.utils.gpdp_train import train  # noqa: E402


@dataclass
class args:
    # --- logging ---
    save: bool = True
    run_name: str = ""

    # --- environment ---
    task: str = "endogenous"
    discount_factor: float = 0.95
    common_noise: bool = True

    # --- GPDP ---
    algo: str = "gpdp"
    seed: int = 0
    num_iterations: int = 30
    num_stations: int = 8  # economies simulated per iteration (the value GPs' training points per period)
    alpha: float = 0.5  # blending: policy <- (1 - alpha) * policy + alpha * best response
    start_share: float = 0.15  # starting policy: every state consumes this share of cash on hand
    n_nodes: int = 5  # Gauss-Hermite points for tomorrow's shock
    search_iter: int = 30  # golden-section iterations per best share
    policy_time: bool = True  # policy GP inputs (z, K, T - t); --no-policy-time: (z, K) only (time ablation;
    #                            the value GPs stay one per period, as finite-horizon DP requires)
    policy_memory_run: str = ""  # RSPG run whose GRU (frozen) feeds the policy GP: inputs (z, K, hidden state);
    #                               use with --no-policy-time (RSPG's information set: no explicit date)
    policy_memory_zk: bool = True  # with memory: inputs (z, K, h); --no-policy-memory-zk: h only (as RSPG's network)
    grid_actions: bool = False  # train on MFAX's 20 grid actions (best responses among them; the
    #                             population plays its shares on the grid through MFAX's env.mf_step)
    law_of_motion: bool = False  # tomorrow's K from a GP law of motion (z, K, T - t) -> K' - K learned from
    #                              each forward pass; expectation over (z', K') in closed form (GPDP paper,
    #                              section 4 and Appendix A; gpdp_lom.py). With --single-vgp only.
    lom_support_paths: int = 0  # with --law-of-motion: 0 solves the Bellman equation at the stations' (z, K, t);
    #                             n > 0 at n support paths simulated with the law of motion itself (the
    #                             stations then only teach the law of motion; needs --alpha 1)

    # --- GPs (bounded hyperparameters, see gpdp_backward.fit_gp_bounded) ---
    gp_lr: float = 0.05
    vgp_steps_first: int = 300  # value GP of the last period, each iteration
    vgp_steps: int = 50  # value GPs of earlier periods (warm started from the next period's)
    single_vgp: bool = False  # ONE value GP over (z, K, T - t) for all periods instead of one per period
    vgp_hyp_steps: int = 30  # single value GP: Adam steps per re-fit (warm started)
    vgp_hyp_cols: int = 125  # single value GP: output columns used to fit the hyperparameters
    pgp_steps_first: int = 300  # policy GP, first iteration
    pgp_steps: int = 100  # policy GP, later iterations (warm started)
    lom_steps: int = 300  # law-of-motion GP, every iteration (cold started)

    # --- evaluation (as gp_train.py: fixed economies and key) ---
    eval_frequency: int = 1
    eval_envs: int = 32
    eval_seed: int = 12345


if __name__ == "__main__":
    train(tyro.cli(args))
