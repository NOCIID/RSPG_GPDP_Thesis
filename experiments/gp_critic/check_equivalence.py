"""
Check 0: with the GP switched off, full-length windows and every economy reset at each step
(--no-use-gp --horizon 128 --phase-counts 8), the new train step must reproduce the original
spg.py / rspg.py train step exactly: same random numbers, same gradient, same parameters.

    python -m experiments.gp_critic.check_equivalence
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mfax.algos.hsm.algos import rspg as orig_rspg  # noqa: E402
from mfax.algos.hsm.algos import spg as orig_spg  # noqa: E402
from mfax.algos.hsm.algos.rspg_gp import args as RSPGGPArgs  # noqa: E402
from mfax.algos.hsm.algos.spg_gp import args as SPGGPArgs  # noqa: E402
from mfax.algos.hsm.utils.gp_critic import empty_gp  # noqa: E402
from mfax.algos.hsm.utils.gp_rollouts import MFPolicyFn  # noqa: E402
from mfax.algos.hsm.utils.gp_train import (  # noqa: E402
    build_policy,
    init_actor,
    initial_econ,
    make_train_step,
    phase_layout,
)
from mfax.envs import make_env  # noqa: E402

N_STEPS = 3


def check(name, orig_module, new_args, recurrent):
    env = make_env("pushforward/endogenous", common_noise=True)
    individual_states = env.params.states
    orig_args = orig_module.args(
        task="endogenous",
        state_type="states",
        discount_factor=new_args.discount_factor,
        normalize_obs=True,
        normalize_states=True,
        num_envs=new_args.num_envs,
        num_iterations=new_args.num_iterations,
        lr=new_args.lr,
    )
    net = build_policy(new_args, env, recurrent)
    policy_fn = MFPolicyFn(env, net, individual_states, True, True, recurrent)
    layout = phase_layout(new_args.phase_counts, new_args.horizon, new_args.discount_factor)
    actor_ts, rng = init_actor(new_args, env, net, individual_states, recurrent)

    orig_step = jax.jit(orig_module.make_train_step(orig_args, env, net, individual_states))
    new_step = jax.jit(make_train_step(new_args, env, policy_fn, layout))

    gp = empty_gp(new_args.n_gp, 2, env.n_states)
    econ = initial_econ(new_args, env, policy_fn, layout, jax.random.PRNGKey(0))

    orig_state = (actor_ts, rng)
    new_state = (actor_ts, gp, econ, rng)
    results = []
    for step in range(N_STEPS):
        orig_state, _ = orig_step(orig_state, None)
        new_state, (loss,) = new_step(new_state, None)
        diffs = jax.tree.map(
            lambda a, b: jnp.max(jnp.abs(a - b)), orig_state[0].params, new_state[0].params
        )
        scale = jax.tree.map(lambda a: jnp.max(jnp.abs(a)), orig_state[0].params)
        max_diff = float(max(jax.tree.leaves(diffs)))
        max_scale = float(max(jax.tree.leaves(scale)))
        moved = jax.tree.map(
            lambda a, b: jnp.max(jnp.abs(a - b)), orig_state[0].params, actor_ts.params
        )
        results.append(
            dict(
                step=step + 1,
                max_abs_param_diff=max_diff,
                max_abs_param=max_scale,
                max_abs_param_change_from_init=float(max(jax.tree.leaves(moved))),
                loss=float(loss),
                rng_equal=bool(jnp.all(orig_state[1] == new_state[3])),
            )
        )
        print(name, results[-1], flush=True)

    # --- raw gradients: with plain SGD (lr = 1, no clipping) the parameter change is the gradient ---
    sgd_ts = TrainState.create(apply_fn=(), params=actor_ts.params, tx=optax.sgd(1.0))
    (orig_sgd, _), _ = orig_step((sgd_ts, rng), None)
    (new_sgd, _, _, _), _ = new_step((sgd_ts, gp, econ, rng), None)
    g_orig = jax.flatten_util.ravel_pytree(
        jax.tree.map(lambda a, b: a - b, actor_ts.params, orig_sgd.params)
    )[0]
    g_new = jax.flatten_util.ravel_pytree(
        jax.tree.map(lambda a, b: a - b, actor_ts.params, new_sgd.params)
    )[0]
    g_orig, g_new = np.asarray(g_orig, np.float64), np.asarray(g_new, np.float64)
    grad_check = dict(
        grad_norm=float(np.linalg.norm(g_orig)),
        rel_grad_diff=float(np.linalg.norm(g_new - g_orig) / np.linalg.norm(g_orig)),
        cosine=float(g_new @ g_orig / (np.linalg.norm(g_new) * np.linalg.norm(g_orig))),
    )
    print(name, grad_check, flush=True)
    return dict(adam_steps=results, gradient=grad_check)


if __name__ == "__main__":
    common = dict(use_gp=False, horizon=128, phase_counts=(8,), num_envs=8, time_input=False)
    out = dict(
        spg=check("SPG", orig_spg, replace(SPGGPArgs(), **common), recurrent=False),
        rspg=check("RSPG", orig_rspg, replace(RSPGGPArgs(), **common), recurrent=True),
    )
    path = Path(__file__).parent / "results" / "equivalence.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"Saved {path}")
