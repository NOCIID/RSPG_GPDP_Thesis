# Findings journal

A dated log of what was tried on the GP-critic SPG/RSPG work in MFAX (endogenous model), and what
worked and what didn't. Newest entry first. The full method description and every earlier number
are in [experiments/gp_critic/SUMMARY.md](experiments/gp_critic/SUMMARY.md); this file only logs
decisions and outcomes.

---

## 2026-10-14

### GP law of motion + closed-form expectation (GPDP paper, section 4) — ✅ works, same accuracy, +4.5 s per iteration

**Question (requested).** Can the most distinctive part of GPDP work here? That part is a GP model of
the dynamics, with the expectation of the value GP over an uncertain next state computed in closed
form (Appendix A). The goal is to see whether it works, not whether it's necessary.

**What it is.**
- **Law of motion:** a GP over (z_t, K_t, T − t) → K_{t+1} − K_t, learned from each forward pass
  (8 stations × 127 transitions).
  - z' = ρz + νε is known in distribution, and the time left drops by one.
  - So tomorrow's value-GP input x' = (z', K', T − t − 1) is Gaussian, with mean (ρz, K + m, T − t − 1)
    and variance diag(ν², s², 0).
  - The household action is not an input: a household takes prices as given.
- **Closed form:** for an SE kernel and a Gaussian input, E[f(x')] = y_mean + y_std · lᵀα, exactly.
  - Only the input has to be Gaussian; the output doesn't.
  - One l-vector serves all 1000 outputs.
  - It replaces the 5 Gauss–Hermite points over z' and the exact K'.
- **log c̄ vs V:** the report's value GP predicts log c̄, and V = D·u(exp(log c̄)) is curved in it.
  - So E[V] = D·u(exp(E[log c̄]))·exp((1−σ)²·Var[log c̄]/2), the lognormal moment.
  - E[log c̄] comes from the closed form; Var[log c̄] from a first-order expansion.
- **The terminal period:** its continuation is the known terminal utility, not a GP, so 5 × 5
  Gauss–Hermite points over (z', K') are used there.
- **No mean field needed:** a household's reward and next state depend on the aggregate state only
  through (r, w) and the date, and (r, w) is a function of (z, K).
  - So with the law of motion, the backward pass needs only (z, K, t) at each support point.
  - This makes a second variant possible: support points simulated by the law of motion itself
    (`--lom-support-paths 8`).

Code: `mfax/algos/hsm/utils/gpdp_lom.py` (new); flags `--law-of-motion` and `--lom-support-paths`
in `gpdp.py` / `gpdp_train.py` (off by default; the report runs are unchanged). Checks:
`experiments/gpdp/lom_checks.py` → `results/lom_checks.json` (on the final policy of
`report_singlevgp_a1_s0`).

**Checks** (law of motion fitted on 8 stations, tested on 64 fresh economies):

| Check | Result |
|---|---|
| Law of motion: R² of K_{t+1} − K_t | **0.999996**; RMSE 0.001 on K ≈ 25.6; worst error 0.08% of K |
| Krusell–Smith log-linear rule (log K' on log K, z, T − t), for reference | R² −0.32: a single log-linear rule can't capture the start-up and end-of-horizon dynamics |
| Calibration of its variance | z-score std 1.26; 89% inside the 95% band (slightly overconfident) |
| Closed form vs 40-point Gauss–Hermite, K' exact | max 0.015% of V (the current 5-point rule: 0.016%) |
| The same without the lognormal correction | max 0.049%: the correction matters at this level |
| Closed form vs Monte Carlo (20,000 draws), K' uncertain | max 0.026% (about the Monte Carlo noise) |
| Whole backward pass at the stations: law of motion vs exact K' | shares differ by 0.0001 on average (max 0.016); values by 0.002% CE (max 0.14%); same time (1.8 s) |

**What it took to make the law of motion accurate.**
1. **Double precision.** Its targets are almost noiseless.
   - In float32 the hyperparameter fit hit a failed Cholesky factorization on its first steps,
     and silently kept its starting values (length scales 1, R² 0.984, worst error 43%).
   - The predictive variance (s_f² minus a nearly equal term) is also lost to rounding in float32.
   - In float64 it fits (noise ≈ 0.001 in standardized units).
2. **log(T − t) as input instead of T − t.** K falls steeply in the last periods, from ≈27.5 to
   ≈5.6, as households run down their wealth. R² 0.9989 → 0.999996; worst error 1.2% → 0.08%.
3. **Cold start every iteration.** Warm-starting from the previous iteration's hyperparameters got
   stuck.
   - Iteration 0's constant share has no end-of-horizon fall in K, so the time length scale
     sits at its bound (100).
   - 100 Adam steps did not bring it back: NLML 1210 vs −4.2 cold.
   - On fresh economies that meant a worst error of 219% (R² 0.87), instead of 0.39% with a cold start.
4. **Simulating support paths with the law of motion compounds its errors over 127 periods.** With
   the bad warm-started law of motion, the dreamt paths left the data region, K went to 0 and the
   value GP returned NaN. After fix 3 it converges (smoke test: clairvoyant gain 0.057% at
   iteration 2).

5. **A crash, and a fix.** Run (b) seed 1 crashed at iteration 12, after converging normally (gains
   0.06–0.11%), because the law of motion's Cholesky returned NaN.
   - The fit's safeguard skipped steps from non-finite hyperparameters, but never checked the last
     accepted update.
   - Fixed: the fit now returns the last hyperparameters whose NLML was finite, and as a last resort
     raises the noise tenfold until the matrix factorizes.
   - On a normal fit nothing changes (same hyperparameters to every printed digit), so the 5 runs of (a) and run (b)-seed 0
     are kept. (b) seeds 1–4 were rerun.
   - **The rerun crashed again at the same iteration, and the real cause was elsewhere.** At
     iteration 12, all the simulated economies were NaN after period 0: the **policy GP** fitted at
     iteration 11 had NaN weights.
     - Its Gram matrix has a condition number of about 1.6e7, borderline for a float32 Cholesky,
       which failed.
     - `fit_gp_bounded` checks the hyperparameters for finiteness but not the weights, and the
       training loop only checked the targets.
   - Fixed in `gpdp_train.py`: a policy GP with non-finite weights never becomes the population's
     rule, and the previous rule is kept (logged as `policy_kept`).
   - This is not specific to the law of motion: the report's policy GP has the same conditioning
     and was lucky. None of the finished runs (report, (a), (b)-seed 0) has a non-finite row, so
     the guard can't have changed them.
6. **Occasional poor law-of-motion fit (variant a).** In a typical iteration its worst error on the
   fresh economies is 0.08–0.15% of K. Two iterations out of 90 were worse: R² 0.975 and worst
   error 7.4% (seed 2, iteration 19), and 2.0% (seed 3, iteration 15). The gain from deviating
   stayed normal in both (0.11%, 0.09%). Likely a worse local optimum of the hyperparameter fit.
   Possible fix, not applied: fit from both a cold start and the previous hyperparameters, and keep
   the better NLML.

**Full runs:** `experiments/gpdp/run_lom.sh` (5 seeds × {stations, law-of-motion paths}, the
report's settings; it skips finished runs). Summary: `experiments/gpdp/summarize_lom.py` →
`results/summary_lom.json`, from `results/lom_evaluate.json` and `results/relative_errors_lom.json`.

**Results** (64 held-out economies, 5 runs each; mean ± sample std):

| | Report (K' from the simulation) | (a) law of motion, stations | (b) law of motion, its own paths |
|---|---|---|---|
| MFAX exploitability (nearest grid share) | 0.027 ± 0.008 | 0.027 ± 0.008 | 0.027 ± 0.003 |
| Gain from deviating (vs a clairvoyant best response allowed any share) | 0.09% | 0.09% | 0.09% |
| CE relative error, t = 0 / all periods | 0.08% / 0.044% | 0.08% / 0.044% | 0.08% / 0.044% |
| CE relative error by block (0–31 … 120–126) | 0.059 · 0.052 · 0.043 · 0.026 · 0.020 · 0.011% | 0.057 · 0.054 · 0.045 · 0.026 · 0.021 · 0.011% | 0.057 · 0.052 · 0.044 · 0.027 · 0.021 · 0.012% |
| Policy relative error | 1.64% | 1.63% | 1.64% |
| Iterations / time to converge (gain < 0.1%) | 2 / 22 s | 2 / 32 s | 2 / 32 s |
| Time per iteration | 3.3 s | 7.8 s | 7.9 s |
| Law of motion on fresh economies: largest error, median over iterations | — | 0.11% | 0.11% |
| Iterations with the previous rule kept (NaN guard) | 0 | 0 | 1 / 100 |

**Conclusions.**
- **✅ It works**, in both variants: the same accuracy on every metric, in the same 2 iterations.
  Even (b) works, where the backward pass never sees a simulated distribution and the support
  points are 127-step paths imagined by the law of motion (a "world model" in RL terms).
- **The cost is time: +4.5 s per iteration.** Almost all of it is the law of motion's own
  hyperparameter fit: 300 float64 Adam steps on 1016 points take 4.4 s. 100 steps take 1.7 s, with
  a similar worst error on fresh economies (0.16% vs 0.18%, one check, not run in full).
- **Single precision caused every failure** met here (law of motion; policy GP in (b) seed 1).
  User's request: try float64 for all GPDP GPs in the future.
- Added to the report as a new section, "Extension: a learned law of motion" (before
  Limitations), with the checks table and this table.

### Report configuration: one value GP, no blending — ✅ converges in 2 iterations, beats RSPG

**Why.** `reports/gpdp_report.tex` describes one implementation only:
- a single value GP over (z, K, T − t), predicting log c̄;
- no blending (α = 1);
- a policy GP over (z, K, T − t);
- continuous shares and a 15% start.

That exact combination hadn't been run (earlier runs had a single GP *with* blending, or no blending
with per-period GPs). It's run here with 5 seeds, 20 iterations, evaluated every iteration.

Script: `experiments/gpdp/run_report_config.sh`. Results: `results/report_evaluate.json` and
`results/relative_errors_report.json`. Runs: `runs/endogenous/gpdp/report_singlevgp_a1_s*`.

**Results** (64 held-out economies; vs the original RSPG, 5 runs each):

| | RSPG | GPDP (report configuration) |
|---|---|---|
| MFAX exploitability (GPDP: nearest grid share) | 0.039 ± 0.007 | **0.027 ± 0.008** (0.020–0.041; p = 0.05) |
| Gain from deviating vs a best response allowed any share | 1.06% | **0.09%** ± 0.03% |
| CE relative error, t = 0 / all periods | 1.54% / 8.7% | **0.08% / 0.04%** |
| CE relative error by block (0–31 … 120–126) | 1.7 · 2.3 · 5.2 · 14.6 · 29.4 · 50.4% | **0.06 · 0.05 · 0.04 · 0.03 · 0.02 · 0.01%** |
| Policy relative error | 14.2% | 1.6% |
| Time to converge | ~1080 s | **22 s, after 2 iterations** (including about 12 s of one-off compilation) |
| Full budget | 1445 s | 82 s (20 iterations, 3.3 s each) |
| Capital at periods 100 / 127 | 27.5 / 26.7 | 27.8 / 2.9 |
| Numerical breakdowns | 0/5 | **0/5** |

**Notes.**
- **Without blending it converges in 2 iterations.** After that the gain fluctuates between 0.06%
  and 0.27% from iteration to iteration: the population adopts each new rule in full, and the single
  value GP is refitted every iteration.
- **It's slightly less accurate than the per-period GPs with blending** (0.027 vs 0.021 MFAX; 0.09%
  vs 0.06%), but it's the simplest to describe, and it still clearly beats RSPG.

### GPDP with RSPG's information set (8-dim GRU memory, no date) — ❌ worse than RSPG, unstable

**Question (requested).** Give the policy GP the same information as RSPG's network, and compare
with RSPG directly.
- RSPG's network sees the household state (2 dims) and a 128-dim embedding of its GRU, which
  summarizes the price history (today's prices included). It never sees z, K or the date
  directly: 130 inputs in total.
- The policy GP's household state is its output index (1000 outputs), so only the economy part
  needs to be an input.

**Setup.**
- **RSPG with an 8-dim GRU:** a new `--gru-hidden-size` flag in `rspg_gp.py` / `gp_train.py`,
  default 128, so existing runs are unchanged. 5 seeds of RSPG + GP critic (`gpfix_h8_s*`).
  - These provide the GRU (frozen afterwards) and the closest baseline: the same information as
    the GP.
  - **Shrinking the memory costs RSPG nothing:** MFAX exploitability 0.043 ± 0.007, vs 0.039 ± 0.005
    with 128 dims.
- **GPDP:** single value GP over (z, K, T − t) (the planning tool keeps the date). The **policy GP
  has no date**, and the GRU of the RSPG run with the same seed is run, frozen, on each simulated
  economy's prices. Two input sets:
  - **(z, K, h₁…h₈)**: today's prices made explicit, plus memory (10 inputs);
  - **h₁…h₈ only**: the closest to RSPG's network (8 inputs).
- New code (GPDP files only): `GRUFeatures`, `RecurrentPolicyGP`, `GPDPRecurrentAgent` (MFAX's
  recurrent agent interface; it matches GPDP's own simulation to 3e-7), and the flags
  `--policy-memory-run` and `--no-policy-memory-zk`.
- **New safeguards:** the policy is never updated from non-finite targets, and the single value GP
  keeps its previous fit when a solve is non-finite. No run hung.
- Scripts: `run_check_memory.sh`, `run_check_memory_h.sh`. Results: `results/check_memory*_evaluate.json`
  and `results/relative_errors_memory*.json`.

**Results** (5 runs each, 64 held-out economies):

| | RSPG-8 | GPDP (z, K, h₈) | GPDP h₈ only | for reference: GPDP (z, K, T − t) |
|---|---|---|---|---|
| MFAX exploitability, nearest grid | **0.034–0.054** | 0.048–0.077, one run 0.74 | 0.043–0.171, 2 runs NaN | 0.021 |
| Gain from any share (% of consumption) | 1.09% | 0.17–0.57% (one run NaN) | 0.38–1.24% (3 runs NaN) | 0.06% |
| Capital at period 100 | 28.1 | 21–26 | 19–27 | 27.9 |
| Converges? | yes | **no** (policy keeps changing, as with no date) | **no** | yes |
| Runs that broke down (NaN policy or values) | 0/5 | 1/5 | 2/5 | 0/5 |

**Findings.**
1. **With RSPG's information, RSPG wins** on MFAX's metric (p = 0.02 for (z, K, h₈)). GPDP doesn't
   converge, saves too little, and breaks down in some runs. The h-only version is worse still.
2. **The frozen GRU only partly replaces the date.** Some runs learned to dissave at the end (share
   0.21–0.24 at period 120), so the GP did read some time from the memory. Others didn't (0.14).
3. **On the continuous metric, the working GPDP runs beat RSPG-8** (0.17–0.57% vs 1.09%), but that
   reflects RSPG's 20-share grid more than GPDP being better.
4. **Likely reasons:**
   - The GRU was trained jointly with RSPG's network, on RSPG's own economies. As frozen inputs, its
     states are partly out of distribution in GPDP's economies (especially early on, from the 15%
     start), and they move as GPDP's policy changes.
   - With 8–10 learned inputs and about 1000 points, the GP extrapolates badly in unvisited
     hidden-state regions. That produced extreme shares (consumption near 0, utility −∞, NaN).
5. **Fairness caveat.** RSPG learns its memory *for* its policy; GPDP here borrows a memory learned
   for another model. A fair GPDP-with-memory needs memory features suited to a GP: a learned
   clock, past prices, or a memory trained for the GP.

**Conclusion.** GPDP's strength here depends on a low-dimensional input that a GP can use directly
(the date). RSPG's learned memory doesn't transfer. With RSPG's information set, gradient-trained
RSPG is better and more robust.

**To do.**
- Add a fallback when the *policy* GP's fit itself returns non-finite weights; currently only its
  targets are checked.
- Try simple memory features: past prices (lags), or the number of periods elapsed.

---

## 2026-10-13

### One value GP for all periods instead of one per period — works, but no better: keep one per period

**Question (requested).** GPDP fits **126 value GPs per iteration**, one per period (1–126), each
on only 8 simulated economies.
- Is that overfitting?
- Would **one** value GP over (z, K, T − t), like the policy GP, do as well or better?

**What was built.** A `--single-vgp` flag, in GPDP files only; the per-period design stays the
default.
- **One GP whose training data grows during the backward pass.** Period t uses the GP trained on
  periods t + 1 … 126, reaching 8 × 126 = 1008 points.
- **Growing data without recompiling:** padded to a few sizes, with padded rows masked (zero kernel
  rows, unit diagonal, zero targets). Tested exact: predictions match an unpadded fit to 5e-7.
- **Hyperparameters are re-fitted after 4, 8, 16, 32, 64 and 96 periods,** on 125 of the 1000
  output columns, and carried across GPDP iterations.
- Regression: the per-period default still reproduces its earlier numbers exactly.

**Two fixes were needed before it worked.**
1. **Fit consumption equivalents, not raw values.** The targets are log c̄ with V = D(T − t) u(c̄),
   where D is the remaining discount sum.
   - Raw values grow about 20× from the last period to the first, just because more periods
     remain. That trend pushed the hyperparameters to signal scale 8.
   - With about 256 points the kernel matrix's condition number then reached 7.6e7, beyond
     float32. Everything was NaN from period 94 on, and the population collapsed to consuming
     everything.
2. **Don't fit hyperparameters on the first period alone,** whose points all share the same time
   left, and tighten the bounds for about 1000 points (signal ≤ 2, noise ≥ 3%).

After the fixes, on the same economies, its values match the per-period GPs within 0.1–0.8% (CE)
and its shares within 0.002–0.016.

**Results.** 5 runs started; 4 completed (see robustness below). 64 held-out economies; errors
against the exact continuous best response. Scripts: `run_check_single_vgp.sh`. Results:
`results/check_singlevgp_evaluate.json` and `results/relative_errors_singlevgp.json`.

| | **One value GP** (4 runs) | 126 value GPs (5 runs) |
|---|---|---|
| MFAX exploitability, nearest grid action | 0.0211 ± 0.0017 | 0.0206 ± 0.0001 (p = 0.75) |
| Exploitability, % of consumption | 0.071% ± 0.006% | **0.062%** (p = 0.008) |
| Value relative error, t = 0 / all periods | 0.061% / 0.041% | **0.053%** / 0.040% |
| Value relative error by block (0–31 · … · 120–126) | 0.049 · 0.051 · 0.043 · 0.026 · 0.022 · 0.014% | 0.045 · 0.050 · 0.042 · 0.027 · 0.024 · 0.020% |
| Policy relative error | 1.60% | 1.57% |
| Capital at period 100 | 27.7–27.8 | 27.9 |
| Training time (30 iterations), per iteration | 112–115 s, 3.3 s | **78 s, 2.4 s** |
| Runs that broke down | **1 of 5** | 0 of 5 |

**Findings.**
1. **One value GP works:** it reaches essentially the same equilibrium. On MFAX's metric it's
   indistinguishable, and its errors are the same in every period block.
2. **But it's slightly less accurate.** Exploitability is 0.071% vs 0.062%; all 4 runs are above all
   5 per-period runs, though the difference is tiny. During training it wobbles (0.057–0.112%)
   where the per-period version stays flat (0.055%).
3. **It's slower:** about 35–45% more time per iteration, from re-solving a GP with up to 1000
   points at every period instead of 8.
4. **It's less robust.** Seed 4 converged normally to iteration 20, then the single GP produced
   **NaN** between iterations 20 and 25. The run then **hung**, instead of failing, until the
   2-hour job limit killed it, so its evaluation is missing. The cause of the hang itself is not
   understood. The per-period design never produced NaN in any run.
5. **Answer to the overfitting question:** if the 126 per-period GPs (8 points each) were
   overfitting, pooling all 1008 points into one GP should have helped. It didn't. The per-period
   GPs generalize at least as well. The single GP, if anything, slightly under-fits: one smooth
   function across all periods, with a noise floor needed for numerical stability.

**Decision: keep one value GP per period.** If the single GP is revisited, it would need a fallback
when a fit returns NaN (keep the previous GP), and the reason for the hang should be found.

---

## 2026-10-12

### Time-input ablation: GPDP's policy GP without the date — ❌ much worse; the date is essential

**Question (requested).** Does giving the policy GP the time left (T − t) actually help? RSPG can
only infer the date through its GRU, and SPG can't see it at all.

**Setup.**
- `--no-policy-time`: the policy GP's inputs are (z, K) instead of (z, K, T − t). The value GPs stay
  one per period, as finite-horizon DP requires.
- Otherwise identical to step 4: start at 15%, α = 0.5, 30 iterations, seeds 0–4.
- Regression check: the with-date version reproduces its earlier numbers exactly.
- Scripts: `run_check_time.sh`. Results: `results/check_time_evaluate.json` and
  `results/relative_errors_time.json`. Runs: `runs/endogenous/gpdp/notime_*`.

**Results** (5 runs each, 64 held-out economies; relative errors are in consumption-equivalent
units, against the independent clairvoyant continuous best response):

| | GPDP with date | **GPDP without date** | Original RSPG |
|---|---|---|---|
| MFAX exploitability, nearest grid action | 0.0206 ± 0.0001 | **0.172 ± 0.150** (0.079–0.438) | 0.039 ± 0.007 |
| Gain from deviating, continuous clairvoyant | 0.06% | **1.17% ± 0.90%** (0.65–2.77%) | 1.06% |
| Value relative error, t = 0 (μ₀) | 0.053% | **1.91%** | 1.54% |
| Value relative error, all periods | 0.040% | **7.3%** | 8.7% |
| Value relative error by block (0–31 · 32–63 · 64–95 · 96–111 · 112–119 · 120–126) | 0.04 · 0.05 · 0.04 · 0.03 · 0.02 · 0.02% | **2.3 · 2.4 · 3.4 · 10.4 · 24.1 · 45.5%** | (seed 0, vs honest grid BR: 0.2 · 0.7 · 2.9 · 10.6 · 23.8 · 44.4%) |
| Policy relative error | 1.6% | 14.5% | 14.2% |
| Capital at periods 100 / 127 | 27.9 / 2.9 | **19.8 / 19.7** | 27.5 / 26.7 |
| Mean share at periods 0 / 64 / 100 / 120 / 126 | 0.26 / 0.13 / 0.13 / 0.22 / 0.58 | 0.27 / **0.16 / 0.15 / 0.15 / 0.15** | 0.26 / 0.14 / 0.13 / 0.14 / 0.14 |

**Findings.**
1. **Without the date, GPDP doesn't converge.**
   - The honest gain plateaus at 0.6–1.1% and keeps bouncing.
   - The policy keeps changing: the share change stays at about 0.012 per iteration, against 0.0003
     with the date.
   - Each seed stops at a different point of that bouncing. Seed 3 is the worst (0.44 MFAX).
2. **It's worse than RSPG on every measure,** on MFAX's metric (all 5 runs above all 5 RSPG runs)
   and on the relative errors. It saves too little (capital 19.8 vs about 28) and doesn't dissave at
   the end (like RSPG).
3. **The errors are everywhere, not just at the end:** 2.3% already in the first 32 periods,
   against 0.04% with the date. At the end, 45%, the same pattern as RSPG.
4. **Why (hypothesis, consistent with the shares):**
   - The best shares depend strongly on the date: 0.26 → 0.13 → 0.58. (z, K) can't stand in for it.
   - The policy GP's regression weights **all 127 periods equally**, so the late "consume a lot"
     targets pull the single date-blind rule up in mid-episode (0.16 instead of 0.13). That means
     too little saving, low capital, then different best responses, and no fixed point.
   - The objective, by contrast, is discounted. SPG, also date-blind but trained by gradient on the
     discounted objective, reaches capital of about 27 and MFAX 0.048.
5. **Implication.** GPDP's advantage relies on a policy class that can represent the best response,
   here meaning it needs the date. When it can't, regressing on best-response targets is worse than
   following the gradient of the true objective.
   - Giving the date is legitimate: it's public information in a finite-horizon game.
   - But the comparison with RSPG and SPG must state it.

**To try next.**
- Weight the policy GP's training points by γᵗ (the objective's weighting) or drop the last periods.
  This might let a date-blind GPDP converge, which would test hypothesis 4.
- The reverse ablation: RSPG with the date as an input.

### Check 1: GPDP trained on the 20-action grid, and CE relative errors for all methods

**What was built.** GPDP files only:
- **A `--grid-actions` flag:**
  - the backward pass picks the best of the 20 grid actions;
  - the population plays the policy GP's shares on the grid through **MFAX's own `env.mf_step`**;
  - the policy's value comes from MFAX's own `evaluate_given_sequence`, and the clairvoyant best
    response from MFAX's own `br()`.
  - Regression check: the continuous version reproduces its earlier numbers exactly.
- **A new metric, the honest grid gain:** a best response on the 20 actions that doesn't see future
  shocks. It is MFAX's exploitability without the hindsight.
- **`experiments/gpdp/relative_errors.py`: consumption-equivalent (CE) relative errors**, reported
  from now on next to exploitability (requested by the supervisor).
  - Each value V is turned into the constant consumption c̄ with the same lifetime utility.
  - c̄_a / c̄_b = (V_a / V_b)^(1/(1−σ)) (σ = 2 gives V_b / V_a), so ratios don't explode near V = 0
    and have an economic scale.
  - **Value relative error** |c̄_policy / c̄_BR − 1|: at t = 0 weighted by μ₀ (exploitability's
    weighting), over all periods weighted by μ_t, and the max over visited states.
  - **Policy relative error** |c_policy / c_BR − 1|, i.e. the share ratio; mixed policies use their
    mean share.
  - Safety check: for mixed policies, the values reproduce MFAX's `evaluate_given_sequence` to 4e-7.
- Scripts: `run_check1_grid.sh`. Results: `results/check1_evaluate.json` and
  `results/relative_errors.json`. Runs: `runs/endogenous/gpdp/grid_*`.

**Results** (5 runs each, 64 held-out economies, key 777; "BR" = best response):

| | Original RSPG | RSPG + GP critic | GPDP continuous | **GPDP trained on grid** |
|---|---|---|---|---|
| MFAX exploitability, as played | 0.039 ± 0.007 | 0.039 ± 0.005 | 0.303 (split) | **0.048 ± 0.009** (p = 0.12 vs original) |
| MFAX exploitability, nearest grid action | — | — | 0.0206 | **0.0205 ± 0.004** (p = 0.008 vs original) |
| Honest grid gain (no hindsight) | 0.16% | 0.17% | — | 0.21% |
| Time to MFAX ≤ 0.05 (training evaluation) | ~1083 s | ~276 s | — | **23–30 s** |
| **Value rel. error vs honest grid BR, t = 0 (μ₀)** | 0.19% | 0.19% | — | 0.26% |
| **Value rel. error vs honest grid BR, all periods** | **7.5%** | **7.9%** | — | **1.2%** |
| **Policy rel. error vs honest grid BR** | 9.3% | 9.7% | — | 2.4% |
| **Value rel. error vs honest continuous BR, t = 0** | 1.47% | 1.48% | **0.003%** | 1.55% |
| **Value rel. error vs honest continuous BR, all periods** | 8.7% | 9.1% | **0.004%** | 2.4% |
| **Policy rel. error vs honest continuous BR** | 14.0% | 14.4% | **0.25%** | 9.0% |

**Findings.**
1. **On the same 20 actions, GPDP matches RSPG (0.048 vs 0.039, not significant), about 40× faster**
   (MFAX exploitability ≤ 0.05 after about 25 s vs about 1083 s).
2. **Training on the grid adds nothing.** Snapped to the nearest grid action, grid-trained and
   continuous GPDP are identical (0.0205 vs 0.0206, p = 0.97), and both beat RSPG (p = 0.008).
   - The grid's staircase targets make the policy GP's job harder: it smooths the steps into costly
     two-action mixtures, so the as-trained policy is worse than its own snapped version.
   - **Best recipe: learn continuous shares, snap to the grid at the end.**
3. **The relative errors show what exploitability hides: RSPG is badly off at the end of the
   episode.**
   - Its value relative error vs the honest grid BR, by period block (seed 0):
     0.23% (t 0–31) · 0.69% · 2.9% · 10.6% (t 96–111) · 23.8% (t 112–119) · **44.4% (t 120–126)**.
   - Near the end it consumes 16% of cash on hand where the BR consumes 24%: it doesn't dissave.
   - Exploitability, which weights from t = 0 with discounting, barely sees this.
   - GPDP is accurate everywhere: grid GPDP stays at 0.4–1.8% in every block, and continuous GPDP
     at 0.00–0.02% (policy error 0.2–0.9%).
4. **Small policy errors, smaller value errors.** RSPG's consumption rule is 9% off the best grid
   response on average, but its value only 0.19% (at t = 0). Values are flat near the optimum, a
   second-order loss. Part of the 9% is also mixing: a mixed policy's mean share vs a pure grid action.
5. **Max errors are noisy and dominated by odd states** (up to 70–350%, poor households or the last
   periods). They are reported in the JSON but shouldn't be headlined.

**⚠ Caveat (raised by the user): the honest best response is partly circular for GPDP.**
- The honest BR is computed with **GPDP's own machinery**: the same value GPs, lookahead and search.
  For GPDP, "0.003% vs the honest BR" therefore mainly shows that GPDP has converged to what its own
  machinery thinks is best.
- If the V-GPs had a systematic bias (for example from describing the economy by (z, K) only),
  both sides would share it.
- Using the best continuous action in the reference is not the problem; the reference should be the
  optimum. The circular part is the shared GPs.
- **The independent reference is the clairvoyant BR.** It uses no GP: exact backward induction on
  the realized economy, with an exact search per household. It knows the future shocks, so it's at
  least as good as any real deviation. **The true error lies between the honest and clairvoyant
  numbers.**

| Value relative error at t = 0 (μ₀) | Original RSPG | GPDP continuous | GPDP grid |
|---|---|---|---|
| vs clairvoyant **continuous** BR (independent) | 1.54% | **0.053%** | 1.61% |
| vs clairvoyant **grid** BR (independent; MFAX's `br()` logic) | 0.25% | GPDP is better than it | 0.33% |

- So even against a reference that shares nothing with GPDP, its error is **at most about 0.05%**,
  against about 1.5% for RSPG. MFAX's exploitability (MFAX's own code) is independent too:
  0.021 (GPDP snapped) vs 0.039 (RSPG).
- **The machinery isn't just agreeing with itself:**
  - applied to RSPG, the same honest machinery finds a 1.5% improvement;
  - in step 2 it captured 95% of the clairvoyant gain against RSPG.
- **Reporting from now on:** the **clairvoyant BR is the headline reference** for relative errors
  (independent and exact). The honest BR is a secondary "without hindsight" number. Still to do
  (robustness list): a Monte Carlo accuracy check of the V-GPs against brute-force simulation, the
  fully independent test of the GP machinery.

**For the supervisor.** The CE relative errors stay well-behaved: no explosions, unlike raw value
ratios. They're the clearest single argument for GPDP:
- on continuous shares, **0.003% value error and 0.25% policy error, uniformly over the episode**;
- for RSPG, **1.5% / 14%**, rising to 44% value error in the last periods.

### GPDP build step 4: the full loop (no gradient descent, no network) — ✅ converges, beats RSPG

**What was built.** New files only:
- `mfax/algos/hsm/algos/gpdp.py`: the entry point, `python -m mfax.algos.hsm.algos.gpdp`.
- `mfax/algos/hsm/utils/gpdp_train.py`: the forward pass with continuous shares, the loop and the
  logging.
- `GPDPAgent` in `gpdp_policy.py`: an adapter that runs a policy GP through MFAX's *unchanged*
  evaluation code.
  - Its hidden state is a clock, and prices are turned back into (z, K).
  - Shares go onto the grid by `split` (two neighbours, same mean share) or `nearest`.
- `experiments/gpdp/step4_evaluate.py` and `experiments/gpdp/run_step4.sh`.
- Runs are in `runs/endogenous/gpdp/`; results are in `experiments/gpdp/results/step4_evaluate.json`.

**Setup.**
- Start from "every household consumes 15%".
- Each iteration: a forward pass (8 stations), a backward pass (V-GPs, 5 shock points, continuous
  search), then a policy-GP fit on blended targets with α = 0.5.
- 30 iterations, seeds 0–4, plus one run with α = 1 (no blending).
- All runs are scored on 64 held-out economies (key 777, the same as SUMMARY.md) against the 5
  original RSPG runs (`base_s*`) and the 5 RSPG + GP critic runs (`gpfix_s*`).
- Three metrics; gains are in % of consumption:
  - **MFAX's exploitability:** the grid best response, which sees the realized shocks;
  - **continuous clairvoyant:** any share, sees the shocks;
  - **continuous honest:** any share, follows V-GPs that don't see the shocks.

**Results** (mean ± sd over 5 runs):

| | Original RSPG | RSPG + GP critic | **GPDP** |
|---|---|---|---|
| MFAX exploitability | 0.039 ± 0.007 | 0.039 ± 0.005 | split: 0.303 · **nearest: 0.0206 ± 0.0001** |
| … as % of consumption | 0.24% | 0.24% | split: 1.84% · **nearest: 0.12%** |
| Continuous, clairvoyant | 1.06% | 1.06% | **0.062%** |
| Continuous, honest | 0.99% | 0.99% | **0.0016%** |
| Training time | 1445 s | 304 s | **78 s** (30 iterations; 2.4 s per iteration) |
| Time to convergence | ~1083 s (MFAX ≤ 0.05) | ~276 s | **24–29 s** with α = 0.5 (honest gain < 0.01%); **9.7 s with α = 1** |
| Capital at period 100 | 27.5 | 28.2 | 27.9 |
| Capital at period 127 | 26.7 | 28.3 | **2.9** |
| Mean share at periods 64 / 120 / 126 | 0.136 / 0.136 / 0.137 | 0.134 / 0.128 / 0.128 | 0.134 / **0.222 / 0.580** |

**Findings.**
1. **GPDP converges, fast, with no gradient descent.**
   - The honest gain falls 21.5% → 6.7% → 2.4% → 0.8% → … → below 0.01% by iteration 8.
   - It then stays at 0.0016% for the remaining iterations, in all 5 seeds.
   - That's about **40× less training time than the original RSPG** and about 10× less than RSPG
     + GP critic, to their respective convergence.
2. **On MFAX's own grid metric, GPDP beats RSPG once its shares are snapped to the nearest grid
   action:** 0.021 vs 0.039. All 5 GPDP runs are below all 10 RSPG runs (permutation p = 0.008, the
   smallest possible with 5 vs 5), even though GPDP never trained on the grid.
   - The *split* mapping (0.303) is a bad way to put a continuous policy on the grid. A share of
     0.088 becomes a 74/26 lottery between 0.075 and 0.125, which is costly with σ = 2.
3. **On continuous shares the gap is about 17×** (clairvoyant 0.06% vs 1.06%). What's left for GPDP
   is just the hindsight floor: clairvoyant − honest ≈ 0.06%, the same as in step 2.
4. **No cobweb.** With α = 1 the population jumps straight to each best response. Capital at
   period 100 overshoots once (12.8 → 31.2 → 27.4) and then settles. It converges in 2 iterations
   (9.7 s) to the same equilibrium. Blending isn't needed here (from this start), and α = 1 is about
   2.5× faster.
5. **GPDP's equilibrium dissaves at the end of the episode.** Shares rise to 0.22 at period 120 and
   0.58 at 126, and capital collapses from 27.9 (period 100) to 2.9 (period 127). This is the
   rational response to a known end date, and it's what the finite-horizon game as defined asks
   for. RSPG mostly didn't learn it (1 of 5 original seeds did; SUMMARY 3.3).
   - It barely matters for exploitability (γ¹²⁰ ≈ 0.002).
   - But it matters for interpretation, and for plots: is the 128-period episode the model, or a
     truncation of an infinite-horizon economy? (SUMMARY 6 raised the same question.)

**Caveats.**
- **GPDP's seed spread is tiny (sd 0.0001)** because its seeds only change the stations' shock
  draws, not a network's initialization. It isn't comparable to RSPG's seed spread.
- **GPDP's policy uses the date (T − t) directly.** RSPG's GRU can only infer it, and SPG can't
  see it at all. Exploitability compares policies, so the comparison stays valid, but the policy
  classes differ.
- **The nearest-grid mapping is applied after training.** A GPDP trained *on* the grid (best
  responses restricted to the 20 actions, forward pass through MFAX's own `env.mf_step`) would be
  the strictly like-for-like comparison. It's designed but not yet built; it might lower 0.021
  further.
- **Timings:** same laptop CPU, evaluation excluded for both. The RSPG times are from 2026-10-06;
  today's rerun of `gpfix_s0` matched (311 vs 306 s).
- **Only one starting policy** ("consume 15%") was tried.

**Next options.**
- **Time-input ablation (requested):** a policy GP on (z, K) only, without T − t. Does giving the
  date actually help? RSPG can only infer the date with its GRU, and SPG can't see it.
- GPDP trained on the 20-action grid (the strict comparison on MFAX's metric).
- An α = 1 multi-seed confirmation.
- Other starting policies.
- Lags as GP inputs (the paper's memory idea).
- Monte Carlo accuracy of the V-GPs.

### GPDP build step 3: the policy GP — ✅ works

**What was built.** New files only:
- `mfax/algos/hsm/utils/gpdp_policy.py`: a policy GP with inputs (z, K, T − t), 1000 outputs and
  one shared kernel, fitted on log(a / (1 − a)) with `fit_gp_bounded`.
- `experiments/gpdp/step3_policy_gp.py`, with results in `results/step3_policy_gp.json`.

**Setup.**
- Step 2's backward pass with 8 stations against `gpfix_s0` gives the best shares at
  8 stations × 127 periods = 1016 training points.
- One policy GP is fitted on them (4.5 s).
- On 32 fresh economies, a household following the policy GP is compared with the step-2 household,
  which computes its shares exactly by lookahead.

**Results.**

| | Lookahead (step 2) | Policy GP |
|---|---|---|
| Gain over `gpfix_s0` (% of consumption) | 1.027% | **1.025%** |
| Value lost by using the policy GP instead | — | **0.002%** |
| Time to produce shares (32 economies × 128 periods) | 3.2 s | **0.006 s** (500× faster) |

- **Share errors** (population-weighted, relative to the lookahead): 0.2% in mid-episode and 0.5%
  at period 120. The worst visited state is 2–6% off.
- **Learned settings:** the length scales are z: 26, K: 14 and time left: 0.12 (standardized). The
  best shares vary strongly with **time** and hardly with (z, K) within the stations' range.
- **The GPDP policy anticipates the end of the episode**, as predicted. Its mean share is 0.26 at
  t = 0, 0.13 at t = 64, 0.22 at t = 120 and 0.55 at t = 126.

**⚠ Watch point for step 4.**
- In the stations, K follows nearly the same path in every economy, so K is almost a function of
  time, and the policy GP can't separate a K effect from a time effect.
- When the population's policy changes in step 4, K paths shift, and the policy GP must
  generalize to new K at a given t.
- The fresh stations at each iteration retrain it, but blending ("old policy's share at the new
  points") asks the old policy GP to extrapolate.
- To check in step 4: the share error at off-path K.

**Next:** step 4, the outer loop. The population adopts the (blended) policy GP, then forward,
backward, refit, repeat. Exploitability is measured with MFAX's metric (via an adapter) and with the
continuous best response.

### GPDP build step 2: the backward pass with V-GPs, against a fixed population — ✅ works

**What was built.** New files only:
- `mfax/algos/hsm/utils/gpdp_backward.py`:
  - Gauss–Hermite shock points;
  - the one-period lookahead (average over tomorrow's shock, then a continuous search);
  - the backward pass with one V-GP per period (exact terminal continuation at period 127);
  - the GP-implied best-response shares on any path.
- `path_evaluate` added to `gpdp_continuous.py`.
- The script `experiments/gpdp/step2_backward_pass.py`, with results in
  `experiments/gpdp/results/step2_backward_pass.json`.

**Setup.**
- The population is fixed: `gpfix_s0`.
- The stations are N ∈ {8, 16, 32} economies (a nested batch).
- The test uses 32 fresh economies: a household follows the shares implied by the GPs (which never
  see future shocks), and its value is measured along the realized paths.

**Bug found and fixed: GP fits diverged to NaN.**
- The DP targets are exact, so they're nearly noiseless. The marginal likelihood then keeps
  shrinking the noise and growing the signal and length scales, until the kernel matrix is singular
  in float32. The critic's noisy targets never triggered this.
- **Fix:** `fit_gp_bounded`, a projected Adam with the noise at least 1% of the outputs' spread,
  capped signal and length scales, and a fallback to the previous settings.
- `gp_critic.py` is unchanged.

**Results** (μ₀-weighted, 32 fresh economies; gains are % of consumption relative to the policy):

| Deviating household | Sees future shocks? | Gain over `gpfix_s0` |
|---|---|---|
| Best on the grid (MFAX's metric) | yes | 0.26% |
| Best continuous (step 1) | yes | 1.09% |
| **GP-guided (GPDP backward pass)** | **no** | **1.03%** |

- **The GPDP backward pass works.** An honest, non-clairvoyant household guided by the V-GPs
  captures **95%** of the clairvoyant continuous gain.
- **The hindsight advantage is small but real:** 0.060% ± 0.006% of consumption, positive in all 32
  economies (0.016–0.14%). It's an upper bound on the hindsight floor in MFAX's exploitability,
  because the GP-guided household is also slightly suboptimal from GP error. So the floor is at most
  about a quarter of `gpfix_s0`'s measured 0.26%.
- **The number of stations doesn't matter:** N = 8, 16 and 32 give the same realized value to 4
  digits. **8 is enough.**
- **Fast:** the backward pass takes 1.9 s with N = 8 (4.2 s with N = 32), including compilation.
- **GP accuracy:**
  - the value predicted at t = 0 is −0.24% ± 0.62% off the realized one, so no detectable bias;
  - the per-period check (one realized future per economy) can only rule out biases above about
    1% (signed errors −0.6% to +0.3%, s.e. about 0.5%);
  - a Monte Carlo check like `check_critic.py` would be needed for finer accuracy.
- **Mean share at period 64:** 0.1300 (GP-guided) vs 0.1303 (clairvoyant) vs 0.1295 (policy).

**Next:** step 3, fit the policy GP on the stations' best shares, and check it reproduces the
GP-guided household's shares and value on fresh economies.

### GPDP build step 1: what the 20-action grid costs — ✅ done

**What was built.** Two new files; no existing file was changed:
- `mfax/algos/hsm/utils/gpdp_continuous.py` (continuous shares):
  - a push-forward step, rewards and expected values with one share per household state, calling
    the env's per-state functions;
  - a golden-section search;
  - a path-wise best response.
- `experiments/gpdp/step1_continuous_br.py`, with results in
  `experiments/gpdp/results/step1_continuous_br.json`.

**Safety net (new code against MFAX's own code): passed.**
- **A.** Shares equal to grid actions reproduce `env.mf_step`. The μ difference is 6e-8, z is
  identical, prices differ by 7e-9 and rewards by 2e-6 relative: float32 rounding.
- **B.** The search restricted to the grid reproduces `br()`. Values differ by 5e-7 relative, and
  the same action is chosen in 99.997% of cases (the rest are ties).

**Result.** 8 economy paths of the trained `gpfix_s0` policy (RSPG + GP critic). Both best
responses use the realized paths, like MFAX's `br()`.

| | Grid best response (MFAX) | Continuous best response |
|---|---|---|
| Value, μ₀-weighted | −16.970 | **−16.833** |
| Exploitability of `gpfix_s0` | 0.043 | **0.180** |
| Gain from deviating (% of consumption) | 0.25% | **1.07%** |

- **The grid costs a household about 1.3% of consumption on average (μ₀-weighted), and up to
  8.3%.** That's about **5× the measured exploitability**.
  - So `gpfix_s0` is near-optimal *within the grid*.
  - The grid itself is the larger gap.
- **The average saving rate is right:** the mean share at period 64 is 0.129 (policy), 0.130 (grid
  best response) and 0.130 (continuous). The loss comes from **per-household precision**: 53% of the
  mass at period 64 wants a share more than 0.01 away from any grid point. Examples at period 64:

  | Household | Policy | Grid best | Continuous best | Gain from continuous shares (value at t = 0) |
  |---|---|---|---|---|
  | Poor | 0.42 | 0.425 | 0.437 | 0.2% |
  | Middle (wealth 26) | 0.075 | 0.075 | 0.088 | 1.5% |
  | Rich (wealth 90) | 0.075 | 0.075 | 0.075 | 2.5% (from other periods) |

- **Speed:** the whole continuous best response (8 paths × 128 periods × 1000 states, with 30
  search steps plus the 20 grid checks) takes **0.67 s**. That is encouraging for GPDP's backward
  pass.

**Caveats.**
- One policy, one seed, 8 paths.
- Both best responses are clairvoyant (realized paths), so the comparison is like-for-like, but the
  absolute exploitabilities include the unmeasured hindsight floor.
- It measures a *single* deviating household against the grid policy's economy. If everyone used
  continuous shares, the equilibrium (K, prices) would shift.

**Side finding.** `common_noise` is a parameter of the endogenous env, but the push-forward step
never reads it: z is always random (`mf_step_env`, pushforward/macro/endogenous.py:114).

**Next:** step 2, the backward pass with V-GPs and 5 shock points for a fixed population, as a
non-clairvoyant best-response oracle.

### Design: GPDP instead of gradient descent — v0, before any implementation

**Goal.** Stop learning the policy by gradient descent (MLP → Beta → `prob_a`). Learn it the
GPDP way instead (Deisenroth, Rasmussen & Peters 2009, `Gaussian-process-dynamic-programming_2009_Neurocomputing.pdf`):
dynamic programming computes the best actions, and a **policy GP** learns them. Drop the 20-action
grid in favour of continuous consumption shares. This entry records the starting design, so later
changes can be traced back to it.

#### Where we started

The first proposal, worked out with ChatGPT, was:
- **three GPs** as in the paper: value V, action-value Q (one per state, over actions), and policy;
- backward optimization along simulated economy paths, with the population held fixed;
- an outer loop in which the population adopts the improved policy.

#### Facts from the code that shaped the design

- **MFAX already solves the household problem exactly on the grid.**
  [best_response.py:40-70](mfax/algos/hsm/best_response.py#L40-L70) computes Q for all 1000 states ×
  20 actions and takes the argmax. These are the paper's inner loops, done exactly.
- **MFAX's best response sees the future.** `br()` runs backward induction on each *realized* path,
  so it knows the future shocks z. The exploitability metric is therefore measured against a
  clairvoyant best response.
  - Consequence: even the exact equilibrium has exploitability above zero, a floor under every
    method's number, RSPG included.
  - Its size is unmeasured, probably small (the innovation sd of z is 0.03).
- **The environment already accepts continuous actions.**
  [pushforward/macro/endogenous.py:156-159](mfax/envs/pushforward/macro/endogenous.py#L156-L159)
  uses a vector action directly as the share. Only the generic loops (`mf_transition`,
  `mf_expected_value`, `mf_reward`) loop over the 20 action indices.
- **The action grid is coarse where it matters.**
  - Trained policies consume 11–19% of cash on hand. The neighbouring grid actions there, 0.125
    and 0.175, differ by +40% in consumption.
  - Rich households' optimum (about 4–5%) falls between 0.025 and 0.075.

#### Design v0

**Two passes per iteration.**

```
FORWARD   t = 0 → 127   simulate N economies under the current population policy (one share per
                        state); save (z, K, μ) at every t. These are the "stations".
BACKWARD  t = 127 → 0   compute exact best values and best shares at the stations, period by
                        period, using the V-GP of the next period for the continuation.
```

**The backward pass at one period t, for one economy (station):**
1. **Today's prices** r, w come from (z, K), exactly.
2. **Tomorrow's capital K'** is known from the forward pass: tomorrow's μ depends on today's choices,
   not on tomorrow's shock.
3. **Tomorrow's shock:** the realized z' is deliberately ignored, to avoid hindsight. Use 5
   Gauss–Hermite points instead: z' = ρz + νε with ε ∈ {0, ±1.356, ±2.857} and probabilities
   {0.533, 0.222, 0.011}.
4. **One averaged continuation:** ask V-GP_{t+1} at the 5 values of (z', K'), and average the 5
   lists of 1000 values into one list W. This works because a household's next state doesn't
   depend on z'; only its value does.
5. **Search per household state** (all 1000 at once, vectorised):
   - Q(a) = u(a · cash) + γ · Σ over the 6 next cells of prob · W[cell], where the 6 cells are the
     2 wealth neighbours × 3 income moves;
   - a golden-section search over a ∈ (0, 1), about 20 evaluations to 1e-4 precision;
   - Q is concave in a, so the search can't get stuck;
   - output: V_t (1000 values) and a\*_t (1000 shares).
6. **Fit the V-GP for period t** on the N stations: inputs (z, K), targets the 1000 exact values.
   Then go to t − 1.

At **t = 127** there's no search and no GP: V = u(cash on hand), which is exact at any (z, K), since
prices follow from (z, K) by formula.

**The policy GP.** Its three ingredients work together; they are not alternatives:
1. **Inputs (z, K, T − t):** one GP for all periods, sharing information across neighbouring
   periods.
2. **1000 outputs with one shared kernel:** the same structure as `gp_critic.py`, with different
   targets.
3. **Fit log(a / (1 − a))** and map predictions back, so shares always stay in (0, 1). This also
   gives the right resolution near 0, where 0.01 vs 0.02 means twice the consumption. Clip a\* at
   0.999 first.

It is fitted once per iteration on the stored (z, K, T − t) → a\* pairs, about N × 128 of them,
subsampled.

**The outer loop.**
```
π₀ → repeat: forward (π_k) → backward (V-GPs, a*) → fit policy GP on (1 − α)·a_old + α·a*  → π_{k+1}
```
- **Start:** the trained RSPG policy for debugging (GPDP should barely move it), and a simple rule
  such as "consume 15%" for the fair race against RSPG.
- **Blending α:** start at 0.5 and lower it if capital or exploitability zig-zags. Blending
  *actions* keeps a single GP, and is standard for decision rules in heterogeneous-agent macro.
- **Stop:** when GPDP's own exploitability stops shrinking, or the largest change in any share is
  below 0.001, or after a cap of about 30 iterations.
  - GPDP's own exploitability is nearly free: the best-response value at t = 0 comes from the
    backward pass, and the policy's own value takes one more backward pass with the policy's shares
    and no search.
  - The final score uses MFAX's exploitability on held-out economies.

#### Decisions, and why

| Decision | Why |
|---|---|
| **No Q-GP** | In the paper the Q-GP interpolates between 25 fixed actions. Here Q is exact and cheap at *any* share, so a direct search replaces it. A Q-GP would need about 4 million tiny fits per iteration and add error. |
| **No H = 32 windows and no Monte Carlo warm start** | GPDP goes backward from an exactly known end (t = 127), so nothing needs guessing. H = 32 saved *gradient* computation, and GPDP has none. Every period already uses the GP for its continuation, so each period is solved once and nothing is redundant. The policy also needs targets for all 128 periods. |
| **5 shock points, not one sampled z per path** | The max must see the *average* over tomorrows before choosing (E[max] ≥ max E). One path has only one tomorrow. Shocks are treated like income shocks, whose 3 outcomes are already averaged inside Q. Sampling remains in the forward pass, where there's no max. |
| **Keep the state grid, drop the action grid** | Continuous shares are cheap: the env supports them, and the transition per state goes from 20 × 6 to 6 cells. Dropping the state grid would need a new representation of μ (simulated agents, or a parametric density, which the README lists as a wanted contribution) plus GPs with wealth as an input. That's a separate, bigger project. |
| **N economies is a setting to tune** | RSPG's 8 per step works because it repeats over 500 steps. Here the economies are the GP's training points per period. Try 8, 16 and 32; pooling stations over neighbouring periods (time as input) may make 8 enough. |
| **Backward from the terminal period, not fixed-point sweeps** | The (6, 2) experiment above showed that a GP refitted only on its own predictions drifts (median error about 40%). Backward induction ties every value to the exact end. |

#### Risks

1. **The outer loop may cycle (cobweb):** high r → save a lot → high K → low r → save little → …
   Gradient descent damped this implicitly through small steps. → Blend (α). The number of
   iterations is the main unknown for speed.
2. **Hindsight:** a max along one realized path uses future shocks, which makes values
   over-optimistic, makes the action targets contradictory, and removes the precautionary motive
   for aggregate risk. → Use the 5 shock points (design step 3).
3. **Fitting the policy GP:**
   - grid argmax targets would form a staircase; continuous search removes this;
   - corners near a = 1 for poor households: clip, and accept slight rounding;
   - blending creates mixtures that a one-share GP can't represent: blend actions instead; the
     formal fictitious-play guarantees then don't strictly apply.
4. **Approximate aggregation:** the GPs see (z, K), not the full μ. This is not discussed in detail
   yet. The critic's R² of 0.98 suggests it's adequate here, but the policy is a different function.
5. **Value drift:** handled by backward induction from the exact terminal value. Still check the
   V-GP against brute-force Monte Carlo.

#### Evaluation plan

- **MFAX's discrete exploitability:** split each share between its two neighbouring grid actions.
- **A continuous-action best response for fairness:** a continuous policy could otherwise beat
  MFAX's 20-action best response, which would make exploitability meaningless. Score RSPG against it
  too; the difference shows what the action grid costs the original method.

#### Build order (each step produces numbers to check)

1. **One economy path:** the continuous best response for every household, against MFAX's
   20-action best response on the same path. Measures what the grid costs, with no GP involved.
2. **Backward pass with V-GPs and 5 shock points** for a *fixed* population (the RSPG policy), as a
   best-response oracle. Compare it with MFAX's clairvoyant best response; this also measures the
   hindsight floor.
3. **Fit the policy GP** on those best shares, and check its error on held-out economies.
4. **The full outer loop:** blending, stopping, then exploitability and wall-clock time against
   RSPG and RSPG-GP.

#### Open questions

- How many outer iterations does it take, and so is it faster than RSPG-GP (304 s)?
- How many economies N per pass?
- Do GP settings fitted once transfer across periods?
- Is (z, K, T − t) enough to describe the economy for the *policy*?

### Experiment: phase layout (6, 2) instead of (4, 2, 1, 1) — ❌ worse, keep (4, 2, 1, 1)

**Question.** Does moving the economies of the late windows to the early ones reduce the error and
improve the metrics?

**Answer: no.**
- (6, 2) learns slightly faster early on but ends worse and less reliable.
- Its critic drifts far off: 2–85% consumption-equivalent error (median about 40%, and one seed
  diverged), against about 1% for (4, 2, 1, 1).
- Near the end of training, its gradient points the wrong way.
- The cause is the critic losing its anchor, not the windowed estimator. With a good critic, the
  (6, 2) gradient is fine.
- The last window carries only 0.6% of the objective, but it is what keeps the critic honest.

| | Default (4, 2, 1, 1) | Tested (6, 2) |
|---|---|---|
| Economies per window (periods) | 4 (0–31), 2 (32–63), 1 (64–95), 1 (96–127) | 6 (0–31), 2 (32–63) |
| Periods that get a gradient | 0–127 | 0–63 |
| Where the last window ends | true end of the episode (terminal reward, no GP) | period 64, on a GP estimate |
| Compute per step | 8 economies × 32 periods | same |

Share of the objective (and of exploitability) carried by each window, γ = 0.95:
0–31: 81% · 32–63: 16% · 64–95: 3% · 96–127: 0.6%.

**Expected.**
- *For (6, 2):* 50% more economies on the window that carries 81% of the objective, so a less noisy
  gradient there. Slightly cheaper RSPG steps (shorter GRU replay, one pipeline-fill step instead of 3).
- *Against (6, 2):*
  1. Periods 64–127 (3.6% of the objective) never get a gradient. The estimator is then no longer
     exactly the full gradient in expectation, so the report's claim would weaken.
  2. No window reaches the true end, so the critic loses its anchor. Each refit replaces the GP's
     data with the latest windows' targets, which all bootstrap from the GP itself. The critic's
     knowledge of the finite horizon then comes only from the first fit. It may drift towards an
     infinite-horizon value at period 64, i.e. back to the horizon bias that the time-aware critic
     removed (3.4% on average, 7.8% at period 75 for the first version).

**Setup.**
- Same protocol as SUMMARY.md: RSPG with 5 seeds and SPG with 3 seeds, 500 steps, 8 economies,
  lr 0.01, CPU.
- Flags: `--use-gp --horizon 32 --phase-counts 6 2 --time-input`.
- Compared against the existing original (`base_s*`) and (4, 2, 1, 1) (`gpfix_s*`) runs.
- Runs are in `runs/endogenous/hsm_{rspg,spg}_gp/gp62_s*`. Outputs are in
  `experiments/gp_critic/results/*gp62*.json` and `results/logs/*gp62*`.
- Reproduce with `PY=python caffeinate -i bash experiments/gp_critic/run_gp62.sh` (about 1h45).

#### Results

**1. Exploitability and speed** (held-out, 64 economies; mean ± sd over seeds; `seeds_{rspg,spg}_gp62.json`)

| | Original | (4, 2, 1, 1) | (6, 2) |
|---|---|---|---|
| RSPG exploitability | 0.039 ± 0.007 | 0.039 ± 0.005 | **0.046 ± 0.013** |
| RSPG range over seeds | 0.033–0.051 | 0.031–0.046 | 0.034–0.067 |
| RSPG p vs original / vs (4, 2, 1, 1) | — | 0.92 / — | 0.37 / 0.39 |
| SPG exploitability | 0.048 ± 0.004 | 0.049 ± 0.002 | **0.065 ± 0.014** |
| SPG p vs original / vs (4, 2, 1, 1) | — | 0.7 / — | 0.2 / 0.1 (the minimum possible with 3 vs 3) |
| RSPG training time (time per step) | 1445 s (2.89 s) | 304 s (0.572 s) | 307 s (0.575 s) |
| RSPG time to exploitability ≤ 0.10 / 0.07 / 0.05 (median) | 864 / 1002 / 1083 s | 215 / 247 / 276 s | **190 / 219** / 278 s |
| RSPG runs reaching ≤ 0.042 | 4 / 5 | 4 / 5 | **2 / 5** |

- **Faster early:** about 12% less time to reach 0.10 and 0.07. SPG's geometric-mean exploitability
  is 0.53 vs 0.93 at step 50, and 0.20 vs 0.37 at step 100.
- **Worse at the end:** the result plateaus higher and is more variable. All three SPG seeds end
  above every (4, 2, 1, 1) seed.
- **No speed gain per step:** the shorter GRU replay doesn't show in wall-clock time.

**2. Learned equilibrium (RSPG)** (gaps to original seed 0; the original's own seed-to-seed noise is
in brackets)

| | Original | (4, 2, 1, 1) | (6, 2) |
|---|---|---|---|
| Policy gap (noise 2.7%) | 2.7% | 2.9% | **5.2%** |
| Capital gap (noise 7.2%) | 7.2% | 7.2% | **15.2%** |
| Capital at period 100 | 28.0 | 28.7 | **32.6** |
| Consumption share, periods 75–127 | 0.134–0.138 | 0.128–0.130 | **0.115–0.123** |

(6, 2) **over-saves**, increasingly late in the episode. This is the same direction as the
first-version horizon bias (capital 31.1 at period 100), but stronger. SPG shows the same pattern:
capital 29.8 vs 27.4.

**3. Critic error at the end of training** (consumption-equivalent, population-weighted, vs 32-path
Monte Carlo; `critic_by_period_{rspg,spg}_gp62.json`)

| | (4, 2, 1, 1), period 32 | (4, 2, 1, 1), period 64 | (6, 2), period 32 | (6, 2), period 64 |
|---|---|---|---|---|
| RSPG, 5 seeds | 0.6–0.7% | 0.6–1.7% | **2.4–42%** | **8.9–78%** |
| SPG, 3 seeds | 0.6% | 0.5–1.6% | **2.1–42%** | **73–85%, and one seed diverged (+5000%)** |

- Every (6, 2) error is **negative**: the critic is far too pessimistic.
- The test reproduces the stored check exactly (gpfix_s0: 0.99% pooled; −0.7 / +0.8 / +0.5% signed).

**4. Gradient vs the full 128-period gradient** (relative error, with the Monte Carlo noise floor in
brackets; `gradient_layouts_rspg_*.json`)

| Policy, critic | (4, 2, 1, 1) | (6, 2) |
|---|---|---|
| Initial policy, fresh (accurate) critic | 0.5% (0.4%) | **4.8%** (0.4%), norm ratio 0.955 |
| Trained (4, 2, 1, 1) policy, its own anchored critic | 2.2% (11%) | 3.1% (12%) |
| Trained (6, 2) policy, its own drifted critic | 89% | **121%, cosine −0.98** |

#### Reading the results

1. **The windowed estimator is not the problem.**
   - Dropping periods 64–127 biases the gradient by about 5% at the start of training.
   - With a good critic, the (6, 2) gradient is within noise near the optimum.
2. **The critic losing its anchor is the problem.**
   - In (6, 2) every refit target is bootstrapped from the GP, and nothing reaches the true end of
     the episode.
   - The critic's finite-horizon knowledge only comes from the first fit, and it drifts over
     training: strongly pessimistic, and divergent in one seed.
   - This was predicted, but at a few percent (the infinite-horizon bias), not tens of percent.
3. **The drifted critic turns the gradient around.**
   - By the end of training, the (6, 2) gradient points against the true one (cosine −0.98).
   - The policy settles at the optimum of the wrong objective, and over-saves.
4. **Why exploitability is only moderately worse:**
   - most progress happens early, while the critic is still close to its first fit;
   - exploitability is flat near the optimum.

#### Takeaways

- **Keep (4, 2, 1, 1).** The last window (periods 96–127) costs 1/8 of the compute for 0.6% of the
  objective. It is the only source of non-bootstrapped targets, and so it keeps the critic honest.
- **For the report:** this is a stronger reason for tiling the episode than unbiasedness alone. A
  one-line ablation could go in section 2, point 4.
- **Next, if useful:**
  - **(5, 1, 1, 1)** tests "more economies early" while keeping the anchor. It remains open whether
    the early speed-up survives.
  - If (6, 2)-style allocation is wanted, the critic needs another anchor. Options: keep the first
    fit's full-episode targets in every refit, or add occasional full-episode targets (this costs
    compute).

### Discussion: why (4, 2, 1, 1)?

- **Hard constraints:**
  - the counts sum to `num_envs` = 8, the same compute as the original;
  - at least one economy per window, or its periods are never trained;
  - non-increasing counts, since a phase-k economy continues a phase-(k−1) one;
  - 4 × 32 = 128, so the windows tile the episode.
- **Heuristic:** halve at each phase, putting more economies where the discount weight is. The
  report and SUMMARY state the layout but don't justify the exact counts. Proportional allocation
  with at least 1 per window would give (5, 1, 1, 1), and that wasn't tested.
- **The last window (96–127) carries only 0.6% of the objective but does two jobs:**
  - it is the only window whose targets aren't GP guesses (it anchors the critic);
  - it makes the windowed gradient equal the full gradient in expectation.
- **Report wording:** in section 2, "the gradient equals the original full-simulation gradient on
  average" should add "up to the GP's error". The phase-k start distributions also come from the
  previous step's policy (SUMMARY 2.4 says so; the report doesn't).

### Codebase observations (from the code walkthrough; not tested)

- **The action grid is linear and coarse at the low end.**
  - The 20 actions are the bin midpoints 0.025, 0.075, …, 0.975.
  - The Beta policy's density is evaluated on these points and softmaxed, so the policy is
    effectively discrete.
  - Back-of-envelope: rich households should consume about 4–5% of cash-on-hand, between the first
    two grid points. Mixing the two costs utility (Jensen). Trained policies average 11–19%, so the
    effect may be small.
  - *Candidate experiment:* a geometric action grid near 0, or more bins.
- **`normalize_obs` is a no-op** for the endogenous environment: both branches return the raw prices.
- **Wealth above the grid's top (99) piles up at 99**, because `distribute` clips. *To check:* the
  mass at the top grid point in trained runs.
- **The income marginal is constant and uniform** (a doubly stochastic chain from a uniform μ₀), so
  average income is about 0.8416. Prices move only through K and z. This is already used by the
  critic.
- **Unused scaffolding:** `is_partially_observable`, `discrete_action_space`, `is_truncated` (always
  0) and the `Generic`/`TypeVar` parameters of `BaseEnvironment`.
- **Tooling:** the repository-root `settings.json` is not read by VS Code, which reads
  `.vscode/settings.json`.

### Environment

The Python environment used for the earlier runs had been partly deleted by macOS's temp cleanup
(again). It was recreated as a conda env with Python 3.10 and the versions in SUMMARY.md (jax 0.5.3
CPU, flax 0.10.5, optax 0.2.3, chex 0.1.86, distrax 0.1.5, gymnax 0.0.8, tyro 0.7.3, numpy 1.26.4,
wandb 0.17.3, scipy 1.15.2).
- **Reproduction check: passed.**
  - `gpfix_s0` rerun as `gpfix_s0_repro`: all 11 evaluation points are identical, as are its
    held-out exploitability (0.0463) and policy gap.
  - The training time differs slightly (311.6 s vs 305.8 s).
  - The existing runs are therefore valid baselines.
- The environment lives in the session scratchpad and will be wiped again. Recreate it with:
  `conda create -p <dir> python=3.10` followed by
  `pip install jax==0.5.3 jaxlib==0.5.3 flax==0.10.5 optax==0.2.3 chex==0.1.86 distrax==0.1.5 gymnax==0.0.8 tyro==0.7.3 numpy==1.26.4 wandb==0.17.3 scipy==1.15.2`.

### Files added today

- `experiments/gp_critic/run_gp62.sh`: the whole (6, 2) experiment.
- `experiments/gp_critic/check_critic_by_period.py`: critic error per window end, absolute and
  signed.
- `experiments/gp_critic/check_gradient_layouts.py`: (4, 2, 1, 1) and (6, 2) gradients against the
  full gradient.
- `experiments/gp_critic/summarize_seeds.py`: `gp62` added to the run-name pattern (one word).

---

## Before 2026-10-12: state of the work

Details and numbers are in [SUMMARY.md](experiments/gp_critic/SUMMARY.md) and the 2026-10-06 report.

**Worked**
- **Time-aware GP critic with 32-period staggered windows (4, 2, 1, 1).** It matches the original
  method's exploitability (RSPG 0.039 ± 0.005 vs 0.039 ± 0.007, p = 0.92; SPG 0.049 vs 0.048) with
  4.8× (RSPG) and 5.0× (SPG) less training time.
- **Critic accuracy:** about 1% consumption-equivalent error at the end of training, against about
  0.5% Monte Carlo noise.
- **Gradient:** with the run's own critic, it is within 3.7% of the full gradient, against a 4.3%
  noise floor.
- **Exact reproduction of the original train step** with the GP off (gradients agree to about 1e-7).
- **Rebuilding RSPG's GRU state** from the stored observation history, with gradients.

**Didn't work, or not needed**
- **The first version (an infinite-horizon critic, direct port of the Huggett GP)** was biased
  3.4% on average and 7.8% at period 75. Households saved more late in the episode, and 1 of 5 seeds
  collapsed.
- **Truncation with a zero tail** (no critic): the gradient was 5.4× too large near the solution,
  and exploitability stalled at about 3.5.
- **Fast refits early in training** did not reliably reduce the early critic error (5–30% in the
  first 100–200 steps).
- **Lagged prices as critic inputs** did not improve accuracy.
- **Reset-only windows** gave a gradient 25–36% too small, yet the same exploitability: Adam
  normalizes the step sizes.

**Open issues**
- Policy collapse from the unbounded Beta head: 1 of 5 first-version runs, 0 of 13 time-aware runs,
  0 of 8 original runs.
- Early critic lag.
- CPU-only timings.
- One window length and one learning rate tested.
