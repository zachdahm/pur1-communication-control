
import os
import time
import warnings
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# Import simulation script
try:
    from Combined_demo_upload import SimConfig, main_pke
    _SIM_AVAILABLE = True
except Exception as _err:
    _SIM_AVAILABLE = False
    print(f"[WARNING] Could not import Combined_demo_SA: {_err}")
    print("          Ensure Combined_demo_SA.py is on PYTHONPATH for real results.\n")

RESULTS_DIR = "./sensitivity_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

STEP_SIZE = 2

# Checkpoint helpers for resuming run that were stopped

def _checkpoint_path(study_name: str) -> str:
    """Return the path for a study's per-trial checkpoint CSV."""
    return os.path.join(RESULTS_DIR, f"{study_name}_checkpoint.csv")


def _load_completed_seeds(study_name: str) -> set:
    """
    Return the set of seeds already present in the checkpoint CSV.
    Returns an empty set if no checkpoint exists yet.
    """
    path = _checkpoint_path(study_name)
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path)
        if "seed" in df.columns and "method" in df.columns:
            return set(df["seed"].dropna().astype(int).tolist())
    except Exception:
        pass
    return set()


def _append_checkpoint(study_name: str, record: dict) -> None:
    """
    Append a single completed trial record to the checkpoint CSV.
    Creates the file with a header on the first write.
    """
    path = _checkpoint_path(study_name)
    row  = pd.DataFrame([record])
    if not os.path.exists(path):
        row.to_csv(path, index=False)
    else:
        row.to_csv(path, index=False, mode="a", header=False)


def _load_checkpoint(study_name: str) -> pd.DataFrame:
    #Load the full checkpoint CSV, returning an empty DataFrame if absent.
    path = _checkpoint_path(study_name)
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


# Core simulation wrapper

def run_trial(cfg: "SimConfig") -> dict:
    """
    Run one simulation trial using the provided SimConfig and return metrics.
    return_metrics and plot_results are enforced here.
    """
    cfg.return_metrics = True
    cfg.plot_results   = False

    if _SIM_AVAILABLE:
        result = main_pke(cfg)
        if result is None:
            raise RuntimeError(
                "main_pke returned None with return_metrics=True. "
                "Check that the return branch in main_pke is reached.")

    return result


# Study A - Monte Carlo

def study_monte_carlo(
    n_trials: int = 30,
    methods: list = None,
    scenarios: list = None,
    base_seed: int = 42,
    tuned_pid: dict = None,
    output_filename: str = "mc_raw_results.csv",
) -> pd.DataFrame:
    """
    Run n_trials independent trials for every (method, scenario) combination.

    tuned_pid : dict with keys Kp, Ki, Kd.  If provided, these gains are
                applied to all PID trials.  If None, SimConfig defaults are used.
    """
    if methods is None:
        methods = ["MPC", 'PID']
    if scenarios is None:
        scenarios = ["DoS", "Nominal", "Stochastic Packet Loss"]

    study_name      = "mc"
    completed_keys = _load_completed_seeds(study_name)

    # Build the full ordered config list before doing any work so that
    # seeds are assigned deterministically regardless of resume state.
    all_configs = []
    global_idx  = 0
    scenario_trial_seeds = {
    (scenario, trial): base_seed + (scenario_idx * n_trials) + trial
    for scenario_idx, scenario in enumerate(scenarios)
    for trial in range(n_trials)}
    
    for method, scenario in product(methods, scenarios):
        st = 0.90  if scenario == "Stochastic Packet Loss" else 0.01
        bp = 0.005 if scenario == "DoS"                    else 0.0
        for trial in range(n_trials):
            seed = scenario_trial_seeds[(scenario, trial)]
            cfg  = SimConfig(
                method               = method,
                scenario             = scenario,
                stochastic_threshold = st,
                burst_prob           = bp,
                seed                 = seed,
            )
            if method == "PID" and tuned_pid is not None:
                cfg.pid_kp = tuned_pid["Kp"]
                cfg.pid_ki = tuned_pid["Ki"]
                cfg.pid_kd = tuned_pid["Kd"]
            all_configs.append((cfg, trial, seed))
            global_idx += 1

    total     = len(all_configs)
    remaining = [(cfg, trial, seed) for cfg, trial, seed in all_configs
                 if (cfg.method, seed) not in completed_keys]

    print(f"\n{'='*65}")
    print(f"  Study A - Monte Carlo")
    print(f"  {n_trials} trials x {len(methods)} methods x {len(scenarios)} "
          f"scenarios = {total} runs")
    print(f"  {len(completed_keys)} already completed, "
          f"{len(remaining)} remaining")
    print(f"{'='*65}")

    done = len(completed_keys)
    for cfg, trial, seed in remaining:
        t0  = time.perf_counter()
        rec = run_trial(cfg)
        rec["trial"] = trial
        _append_checkpoint(study_name, rec)
        done += 1
        print(
            f"  [{done:>4}/{total}]  {cfg.method:<10} {cfg.scenario:<28}"
            f"  trial={trial:>2}  RMSE={rec['rmse']:.5f}"
            f"  MPC={rec['avg_mpc_runtime']*1e3:.1f}ms"
            f"  MHE={rec['avg_mhe_runtime']*1e3:.1f}ms"
            f"  wall={time.perf_counter()-t0:.1f}s"
        )

    # Assemble final output from the complete checkpoint
    df = _load_checkpoint(study_name)
    df.to_csv(os.path.join(RESULTS_DIR, output_filename), index=False)
    print(f"\n  Results -> {RESULTS_DIR}/{output_filename}")
    return df


def summarise_monte_carlo(df: pd.DataFrame) -> pd.DataFrame:

    #Compute per-cell descriptive statistics

    rows = []
    for scenario in df["scenario"].unique():
        sub = df[df["scenario"] == scenario]
        for method in sub["method"].unique():
            m   = sub[sub["method"] == method]
            n   = len(m)
            mu  = m["rmse"].mean()
            sig = m["rmse"].std()
            sem = sig / np.sqrt(n)
            rows.append({
                "scenario":              scenario,
                "method":                method,
                "n_trials":              n,
                "rmse_mean":             mu,
                "rmse_std":              sig,
                "rmse_95ci_low":         mu - 1.96 * sem,
                "rmse_95ci_high":        mu + 1.96 * sem,
                "effort_mean":           m["control_effort"].mean(),
                "effort_std":            m["control_effort"].std(),
                "avg_mpc_runtime_ms":    m["avg_mpc_runtime"].mean()   * 1e3,
                "std_mpc_runtime_ms":    m["std_mpc_runtime"].mean()   * 1e3,
                "avg_mhe_runtime_ms":    m["avg_mhe_runtime"].mean()   * 1e3,
                "std_mhe_runtime_ms":    m["std_mhe_runtime"].mean()   * 1e3,
                "avg_total_runtime_ms":  m["avg_total_runtime"].mean() * 1e3,
                "std_total_runtime_ms":  m["std_total_runtime"].mean() * 1e3,
                "p_value_MPC_lt_PID":    np.nan,
            })

        mpc_rmse = sub[sub["method"] == "MPC"]["rmse"].values
        pid_rmse = sub[sub["method"] == "PID"]["rmse"].values
        if len(mpc_rmse) > 1 and len(pid_rmse) > 1:
            _, p_val = stats.mannwhitneyu(mpc_rmse, pid_rmse, alternative="less")
        else:
            p_val = np.nan
        n_methods = len(sub["method"].unique())
        for row in rows[-n_methods:]:
            row["p_value_MPC_lt_PID"] = p_val

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(RESULTS_DIR, "mc_statistics.csv"), index=False)
    print(f"  Summary statistics -> {RESULTS_DIR}/mc_statistics.csv")
    return summary

# Study B - Parameter mismatch x blackout duration

def study_parameter_sensitivity(
    methods: list = None,
    mismatch_levels: list = None,
    blackout_durations_min: list = None,
    n_repeats: int = 5,
    base_seed: int = 1000,
    tuned_pid: dict = None,
) -> pd.DataFrame:
    """
    Grid sweep over kinetic parameter mismatch and blackout duration.
    Checkpointed: resumes from last completed trial on restart.
    """
    if methods is None:
        methods = ["MPC"]
    if mismatch_levels is None:
        beta_mismatch_levels = [1.05]
        lamb_mismatch_levels = [1.10]
    if blackout_durations_min is None:
        blackout_durations_min = [20]

    blackout_steps  = [int(m * 60 / STEP_SIZE) for m in blackout_durations_min]
    study_name      = "param_sensitivity"
    completed_seeds = _load_completed_seeds(study_name)

    # Build full ordered config list
    all_configs  = []
    seed_counter = base_seed
    for method, beta_scale, lamb_scale, bl_steps in product(methods, beta_mismatch_levels, lamb_mismatch_levels, blackout_steps):
        bl_min = bl_steps * STEP_SIZE / 60
        for _ in range(n_repeats):
            cfg = SimConfig(
                method       = method,
                scenario     = "DoS",
                burst_len    = bl_steps,
                burst_prob   = 0.005,
                beta_scale   = beta_scale,
                lambda_scale = lamb_scale,
                seed         = seed_counter,
            )
            if method == "PID" and tuned_pid is not None:
                cfg.pid_kp = tuned_pid["Kp"]
                cfg.pid_ki = tuned_pid["Ki"]
                cfg.pid_kd = tuned_pid["Kd"]
            all_configs.append((cfg, method, beta_scale, lamb_scale, bl_min, seed_counter))
            seed_counter += 1

    total     = len(all_configs)
    remaining = [(cfg, m, b, l, bl, s) for cfg, m, b, l, bl, s in all_configs
                 if s not in completed_seeds]

    print(f"\n{'='*65}")
    print(f"  Study B - Parameter Mismatch x Blackout ({total} runs)")
    print(f"  {len(completed_seeds)} already completed, "
          f"{len(remaining)} remaining")
    print(f"{'='*65}")

    done = len(completed_seeds)
    for cfg, method, beta_scale, lamb_scale, bl_min, seed in remaining:
        rec = run_trial(cfg)
        rec["beta_mismatch_pct"] = beta_scale * 100
        rec["lamb_mismatch_pct"] = lamb_scale * 100
        rec["blackout_min"] = bl_min
        _append_checkpoint(study_name, rec)
        done += 1
        print(
            f"  [{done:>5}/{total}]  {method:<6}"
            f"  beta_mismatch={beta_scale*100:+.0f}%"
            f"  lamb_mismatch={lamb_scale*100:+.0f}%"
            f"  blackout={bl_min:.0f}min"
            f"  RMSE={rec['rmse']:.5f}"
        )

    # Aggregate per-cell means from checkpoint
    raw_df  = _load_checkpoint(study_name)
    records = []
    for (method, beta_mismatch_pct, lamb_mismatch_pct, blackout_min), grp in raw_df.groupby(
            ["method", "mismatch_pct", "blackout_min"]):
        records.append({
            "method":       method,
            "beta_mismatch_pct": beta_mismatch_pct,
            "lamb_mismatch_pct": lamb_mismatch_pct,
            "blackout_min": blackout_min,
            "beta_scale":   grp["beta_scale"].iloc[0],
            "lambda_scale": grp["lambda_scale"].iloc[0],
            "rmse_mean":    grp["rmse"].mean(),
            "rmse_std":     grp["rmse"].std(),
            "effort_mean":  grp["control_effort"].mean(),
            "effort_std":   grp["control_effort"].std(),
            "n_repeats":    len(grp),
        })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(RESULTS_DIR, "param_mismatch.csv"), index=False)
    print(f"  Parameter sensitivity -> {RESULTS_DIR}/param_mismatch.csv")
    return df

# Study C - MHE window length sensitivity

def study_mhe_window(
    window_lengths: list = None,
    n_repeats: int = 10,
    base_seed: int = 2000,
) -> pd.DataFrame:
    """
    Vary MHE estimator horizon and measure RMSE and solver runtimes.
    Checkpointed: resumes from last completed trial on restart.
    """
    if window_lengths is None:
        window_lengths = [5, 10, 15, 20, 30]

    study_name      = "mhe_window"
    completed_seeds = _load_completed_seeds(study_name)

    all_configs  = []
    seed_counter = base_seed
    for wl in window_lengths:
        for _ in range(n_repeats):
            cfg = SimConfig(
                method     = "MPC",
                scenario   = "DoS",
                burst_prob = 0.005,
                mhe_window = wl,
                seed       = seed_counter,
            )
            all_configs.append((cfg, wl, seed_counter))
            seed_counter += 1

    total     = len(all_configs)
    remaining = [(cfg, wl, s) for cfg, wl, s in all_configs
                 if s not in completed_seeds]

    print(f"\n{'='*65}")
    print(f"  Study C - MHE Window Length Sensitivity ({total} runs)")
    print(f"  {len(completed_seeds)} already completed, "
          f"{len(remaining)} remaining")
    print(f"{'='*65}")

    done = len(completed_seeds)
    for cfg, wl, seed in remaining:
        rec = run_trial(cfg)
        rec["mhe_window_label"] = wl
        _append_checkpoint(study_name, rec)
        done += 1
        print(
            f"  [{done:>4}/{total}]  window={wl:>3} steps"
            f"  RMSE={rec['rmse']:.5f}"
            f"  MHE={rec['avg_mhe_runtime']*1e3:.1f}ms"
        )

    raw_df  = _load_checkpoint(study_name)
    records = []
    for wl, grp in raw_df.groupby("mhe_window_label"):
        records.append({
            "mhe_window":           int(wl),
            "rmse_mean":            grp["rmse"].mean(),
            "rmse_std":             grp["rmse"].std(),
            "effort_mean":          grp["control_effort"].mean(),
            "effort_std":           grp["control_effort"].std(),
            "avg_mpc_runtime_ms":   grp["avg_mpc_runtime"].mean()   * 1e3,
            "std_mpc_runtime_ms":   grp["avg_mpc_runtime"].std()    * 1e3,
            "avg_mhe_runtime_ms":   grp["avg_mhe_runtime"].mean()   * 1e3,
            "std_mhe_runtime_ms":   grp["avg_mhe_runtime"].std()    * 1e3,
            "avg_total_runtime_ms": grp["avg_total_runtime"].mean() * 1e3,
            "std_total_runtime_ms": grp["avg_total_runtime"].std()  * 1e3,
            "n_repeats":            len(grp),
        })

    df = pd.DataFrame(records).sort_values("mhe_window").reset_index(drop=True)
    df.to_csv(os.path.join(RESULTS_DIR, "mhe_window_sensitivity.csv"), index=False)
    print(f"  MHE window results -> {RESULTS_DIR}/mhe_window_sensitivity.csv")
    return df

# Study D - MPC objective weight sensitivity
def study_mpc_weights(
    mv_weight_values: list = None,
    rate_weight_values: list = None,
    n_repeats: int = 3,
    base_seed: int = 3000,
) -> pd.DataFrame:
    """
    Grid sweep over mpc_mv_weight and mpc_rate_weight.
    Checkpointed: resumes from last completed trial on restart.
    """
    if mv_weight_values is None:
        mv_weight_values = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    if rate_weight_values is None:
        rate_weight_values = [5e-2, 1e-1, 5e-1, 1, 5]

    study_name      = "mpc_weights"
    completed_seeds = _load_completed_seeds(study_name)

    all_configs  = []
    seed_counter = base_seed
    for mv_w, rate_w in product(mv_weight_values, rate_weight_values):
        for _ in range(n_repeats):
            cfg = SimConfig(
                method          = "MPC",
                scenario        = "Nominal",
                burst_prob      = 0.000,
                mpc_mv_weight   = mv_w,
                mpc_rate_weight = rate_w,
                seed            = seed_counter,
            )
            all_configs.append((cfg, mv_w, rate_w, seed_counter))
            seed_counter += 1

    total     = len(all_configs)
    remaining = [(cfg, mv, rw, s) for cfg, mv, rw, s in all_configs
                 if s not in completed_seeds]

    print(f"\n{'='*65}")
    print(f"  Study D - MPC Weight Sensitivity ({total} runs)")
    print(f"  {len(completed_seeds)} already completed, "
          f"{len(remaining)} remaining")
    print(f"{'='*65}")

    done = len(completed_seeds)
    for cfg, mv_w, rate_w, seed in remaining:
        rec = run_trial(cfg)
        rec["mv_w_label"]   = mv_w
        rec["rate_w_label"] = rate_w
        _append_checkpoint(study_name, rec)
        done += 1
        print(
            f"  [{done:>4}/{total}]"
            f"  mv={mv_w:.0e}  rate={rate_w:.0e}"
            f"  RMSE={rec['rmse']:.5f}"
        )

    raw_df  = _load_checkpoint(study_name)
    records = []
    for (mv_w, rate_w), grp in raw_df.groupby(["mv_w_label", "rate_w_label"]):
        records.append({
            "mpc_mv_weight":   mv_w,
            "mpc_rate_weight": rate_w,
            "rmse_mean":       grp["rmse"].mean(),
            "rmse_std":        grp["rmse"].std(),
            "effort_mean":     grp["control_effort"].mean(),
            "effort_std":      grp["control_effort"].std(),
            "n_repeats":       len(grp),
        })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(RESULTS_DIR, "mpc_weight_sensitivity.csv"), index=False)
    print(f"  MPC weight results -> {RESULTS_DIR}/mpc_weight_sensitivity.csv")
    return df

# Console summary table

def print_mc_summary(mc_stats: pd.DataFrame) -> None:
    cols = [
        "scenario", "method", "n_trials",
        "rmse_mean", "rmse_std", "rmse_95ci_low", "rmse_95ci_high",
        "effort_mean", "effort_std",
        "avg_mpc_runtime_ms", "std_mpc_runtime_ms",
        "avg_mhe_runtime_ms", "std_mhe_runtime_ms",
        "avg_total_runtime_ms", "std_total_runtime_ms",
        "p_value_MPC_lt_PID",
    ]
    cols = [c for c in cols if c in mc_stats.columns]
    print("\n" + "=" * 85)
    print("  MONTE CARLO SUMMARY  (Table for paper)")
    print("=" * 85)
    print(mc_stats[cols].to_string(index=False, float_format="{:.5f}".format))
    print("=" * 85)


# Entry point

if __name__ == "__main__":

    N_MC_TRIALS  = 30
    N_SENS_REPS  = 10
    BASE_SEED    = 2024
    N_PID_SEEDS  = 1    # seeds averaged per DE evaluation
    tests = ['mismatch']

    print("\n" + "#" * 65)
    print("#  Sensitivity Analysis - ANUCENE-D-26-00246              #")
    print("#" * 65)
    if not _SIM_AVAILABLE:
        print("  *** Ensure Combined_demo_upload.py is importable        ***\n")

    tuned_pid = None
    # Study A: Monte Carlo (checkpointed)
    if 'MC' in tests:
        mc_df    = study_monte_carlo(
            n_trials  = N_MC_TRIALS,
            methods   = ["MPC"],
            base_seed = BASE_SEED + 1000,
            tuned_pid = tuned_pid,
        )
        mc_stats = summarise_monte_carlo(mc_df)
        print_mc_summary(mc_stats)

    # Study B: Parameter sensitivity (checkpointed)
    if 'mismatch' in tests:
        sens_df = study_parameter_sensitivity(
            n_repeats = N_SENS_REPS,
            base_seed = BASE_SEED + 1000,
            tuned_pid = tuned_pid,
        )

    # Study C: MHE window length (checkpointed)
    if 'window_len' in tests:
        mhe_df = study_mhe_window(
            n_repeats = N_SENS_REPS,
            base_seed = BASE_SEED + 2000,
        )


    # Study D: MPC objective weights (checkpointed)
    if 'MPC weights' in tests:
        weight_df = study_mpc_weights(
            n_repeats = N_SENS_REPS,
            base_seed = BASE_SEED + 3000,
        )

    print(f"\nAll outputs written to: {os.path.abspath(RESULTS_DIR)}/")
    print("Done.\n")
