#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-paper")
FIG_WIDTH = 468.0 / 72.27
FIG_HEIGHT = FIG_WIDTH * 0.75
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.titlesize": 14,
    "figure.autolayout": True,
    "savefig.bbox": "tight",
})

SEGMENTS = ["12", "34", "56", "78"]
REGIONS = ["background", "signal", "end"]

RDE_SEG_COLS = {
    "12": "Un12 RDE",
    "34": "Un34 RDE",
    "56": "Un1112 RDE",
    "78": "Un1314 RDE",
}

FILL_SEG_COLS = {
    "12": "fillUCN12",
    "34": "fillUCN34",
    "56": "fillUCN1112",
    "78": "fillUCN1314",
}


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def savefig(path):
    ensure_dir(Path(path).parent)
    plt.savefig(path, dpi=160)
    plt.close()
    print("[plot]", path)


def save_df(df, path):
    ensure_dir(Path(path).parent)
    df.to_csv(path, index=False)
    print("[write]", path)


def clean_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_epoch_range(epoch_info, year, epoch):
    with open(epoch_info) as f:
        e = json.load(f)[str(year)][str(epoch)]
    return int(e["start_run_number"]), int(e["end_run_number"])


def load_outputs(input_dir):
    input_dir = Path(input_dir)

    if (input_dir / "all_pulses.csv").exists():
        pulses = clean_cols(pd.read_csv(input_dir / "all_pulses.csv"))
        windows = clean_cols(pd.read_csv(input_dir / "all_windows.csv"))
        summary = clean_cols(pd.read_csv(input_dir / "run_segment_summary.csv"))
        return pulses, windows, summary

    task_dirs = sorted((input_dir / "array").glob("task_*"))
    if not task_dirs:
        task_dirs = sorted(input_dir.glob("task_*"))
    if not task_dirs:
        task_dirs = sorted((input_dir.parent / "array").glob("task_*"))

    if not task_dirs:
        raise FileNotFoundError(f"No merged CSVs or task_* folders found near {input_dir}")

    def cat(name):
        dfs = []
        for d in task_dirs:
            f = d / name
            if f.exists() and f.stat().st_size > 0:
                dfs.append(clean_cols(pd.read_csv(f)))
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    return cat("all_pulses.csv"), cat("all_windows.csv"), cat("run_segment_summary.csv")


def coerce(pulses, windows, summary):
    for df in [pulses, windows, summary]:
        if df.empty:
            continue
        if "run" in df:
            df["run"] = pd.to_numeric(df["run"], errors="coerce").astype("Int64")
        if "segment" in df:
            df["segment"] = df["segment"].astype(str).str.strip()
        if "hold_time_s" in df:
            df["hold_time_s"] = pd.to_numeric(df["hold_time_s"], errors="coerce")
        if "region" in df:
            df["region"] = df["region"].astype(str).str.strip().str.lower()

    for c in ["time_us", "amplitude_pe", "window_width_us", "background_rate_hz"]:
        if c in pulses:
            pulses[c] = pd.to_numeric(pulses[c], errors="coerce")

    for c in ["start_time_us", "end_time_us", "bin_width_us", "pulse_count",
              "seed_count", "observed_count", "expected_count", "final_nll"]:
        if c in windows:
            windows[c] = pd.to_numeric(windows[c], errors="coerce")

    for c in ["start_us", "background_start_us", "background_end_us",
              "signal_start_us", "signal_end_us", "end_start_us", "end_end_us"]:
        if c in summary:
            summary[c] = pd.to_numeric(summary[c], errors="coerce")

    return pulses, windows, summary


def filter_epoch(df, r0, r1):
    if df.empty or "run" not in df:
        return df
    return df[(df["run"] >= r0) & (df["run"] <= r1)].copy()


def load_rde(path, r0, r1):
    if path is None:
        return pd.DataFrame()
    path = Path(path)
    if not path.exists():
        print("[warn] no RDE csv:", path)
        return pd.DataFrame()
    df = clean_cols(pd.read_csv(path))
    if "Run Number" in df:
        df["run"] = pd.to_numeric(df["Run Number"], errors="coerce").astype("Int64")
    elif "run" in df:
        df["run"] = pd.to_numeric(df["run"], errors="coerce").astype("Int64")
    else:
        print("[warn] RDE missing Run Number/run")
        return pd.DataFrame()
    return df[(df["run"] >= r0) & (df["run"] <= r1)].copy()


def fig1():
    return plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))


def figgrid():
    return plt.subplots(2, 2, figsize=(FIG_WIDTH * 1.55, FIG_HEIGHT * 1.55))


def norm_value(row, mode):
    if mode == "none":
        return 1.0
    seg = str(row["segment"])
    if mode == "rhdd":
        return float(row.get("RHDD", np.nan))
    if mode == "segment_rde":
        return float(row.get(RDE_SEG_COLS.get(seg, ""), np.nan))
    if mode == "fill":
        return float(row.get(FILL_SEG_COLS.get(seg, ""), np.nan))
    return np.nan


def summarize_counts(pulses, summary, rde, thresholds, subtract, norm):
    keys = ["run", "segment", "hold_time_s"]

    widths = summary[keys + [
        "background_start_us", "background_end_us",
        "signal_start_us", "signal_end_us",
        "end_start_us", "end_end_us",
    ]].drop_duplicates(keys).copy()

    widths["background_width_s"] = (widths["background_end_us"] - widths["background_start_us"]) * 1e-6
    widths["signal_width_s"] = (widths["signal_end_us"] - widths["signal_start_us"]) * 1e-6
    widths["end_width_s"] = (widths["end_end_us"] - widths["end_start_us"]) * 1e-6

    if not rde.empty:
        keep = ["run"]
        for c in ["RHDD", *RDE_SEG_COLS.values(), *FILL_SEG_COLS.values()]:
            if c in rde.columns:
                keep.append(c)
        widths = widths.merge(rde[keep].drop_duplicates("run"), on="run", how="left")

    rows = []
    for thr in thresholds:
        d = pulses[pulses["amplitude_pe"] >= thr].copy()
        if d.empty:
            continue

        c = d.groupby(keys + ["region"]).size().reset_index(name="n")
        wide = c.pivot_table(index=keys, columns="region", values="n", fill_value=0).reset_index()

        for reg in REGIONS:
            if reg not in wide:
                wide[reg] = 0.0

        wide = widths.merge(wide, on=keys, how="left")
        for reg in REGIONS:
            wide[reg] = wide[reg].fillna(0.0)

        sig_w = wide["signal_width_s"].replace(0, np.nan)
        bg_scaled = wide["background"] * sig_w / wide["background_width_s"].replace(0, np.nan)
        end_scaled = wide["end"] * sig_w / wide["end_width_s"].replace(0, np.nan)

        if subtract == "background":
            raw = wide["signal"] - bg_scaled
            var = np.maximum(wide["signal"] + bg_scaled, 1.0)
        elif subtract == "end":
            raw = wide["signal"] - end_scaled
            var = np.maximum(wide["signal"] + end_scaled, 1.0)
        elif subtract == "sideband":
            raw = wide["signal"] - 0.5 * (bg_scaled + end_scaled)
            var = np.maximum(wide["signal"] + 0.25 * bg_scaled + 0.25 * end_scaled, 1.0)
        elif subtract == "none":
            raw = wide["signal"].astype(float)
            var = np.maximum(wide["signal"], 1.0)
        else:
            raise ValueError(subtract)

        wide["threshold_pe"] = thr
        wide["raw_count"] = raw
        wide["raw_count_var"] = var
        wide["background_scaled_to_signal"] = bg_scaled
        wide["end_scaled_to_signal"] = end_scaled

        wide["norm_value"] = wide.apply(lambda r: norm_value(r, norm), axis=1)
        wide.loc[~np.isfinite(wide["norm_value"]) | (wide["norm_value"] <= 0), "norm_value"] = np.nan
        wide["norm_count"] = wide["raw_count"] / wide["norm_value"]
        wide["norm_count_var"] = wide["raw_count_var"] / (wide["norm_value"] ** 2)
        rows.append(wide)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def ssq_scaled(ht_y, tau):
    if tau <= 0 or not np.isfinite(tau):
        return np.inf
    t = ht_y[:, 0]
    y = ht_y[:, 1]
    scaled = y * np.exp(t / tau)
    if not np.all(np.isfinite(scaled)):
        return np.inf
    m = np.mean(scaled)
    return float(np.sum((scaled - m) ** 2))


def golden_minimize(func, lo=0.1, hi=1e5, tol=1e-6, max_iter=300):
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    fc, fd = func(c), func(d)
    for _ in range(max_iter):
        if abs(hi - lo) <= tol * max(1.0, abs(0.5 * (hi + lo))):
            break
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = func(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = func(d)
    return 0.5 * (lo + hi)


def fit_tau_old_scatter(points):
    d = points[["run", "hold_time_s", "norm_count"]].dropna().copy()
    d = d[np.isfinite(d["hold_time_s"]) & np.isfinite(d["norm_count"]) & (d["norm_count"] > 0)]
    if len(d) < 3:
        return np.nan, np.nan, len(d)

    ht_y = d[["hold_time_s", "norm_count"]].to_numpy(float)
    tau = golden_minimize(lambda x: ssq_scaled(ht_y, x))

    taus = []
    for run in d["run"].dropna().unique():
        sample = d[d["run"] != run]
        if len(sample) < 3:
            continue
        arr = sample[["hold_time_s", "norm_count"]].to_numpy(float)
        tj = golden_minimize(lambda x: ssq_scaled(arr, x))
        if np.isfinite(tj):
            taus.append(tj)

    if len(taus) < 2:
        return tau, np.nan, len(d)

    taus = np.asarray(taus)
    dtau = math.sqrt((len(taus) - 1.0) / len(taus) * np.sum((taus - np.mean(taus)) ** 2))
    return tau, dtau, len(d)


def fit_tau_profile_chi2(points):
    d = points[["hold_time_s", "norm_count", "norm_count_var"]].dropna().copy()
    d = d[np.isfinite(d["hold_time_s"]) & np.isfinite(d["norm_count"]) & (d["norm_count"] > 0)]
    if len(d) < 3:
        return np.nan, np.nan, len(d)

    t = d["hold_time_s"].to_numpy(float)
    y = d["norm_count"].to_numpy(float)
    var = np.maximum(d["norm_count_var"].to_numpy(float), 1e-30)
    w = 1.0 / var

    def chi2(tau):
        if tau <= 0:
            return np.inf
        m = np.exp(-t / tau)
        den = np.sum(w * m * m)
        if den <= 0:
            return np.inf
        A = np.sum(w * y * m) / den
        return float(np.sum(w * (y - A * m) ** 2))

    tau = golden_minimize(chi2)
    cmin = chi2(tau)

    span = max(50.0, 0.25 * tau)
    grid = np.linspace(max(0.1, tau - span), tau + span, 4000)
    vals = np.array([chi2(x) for x in grid])
    ok = vals <= cmin + 1.0
    if not ok.any():
        return tau, np.nan, len(d)

    left = grid[np.argmax(ok)]
    right = grid[len(ok) - np.argmax(ok[::-1]) - 1]
    return tau, 0.5 * (right - left), len(d)


def lifetime_vs_threshold(counts, out_dir, mode):
    if counts.empty:
        return pd.DataFrame()

    rows = []
    fit_fn = fit_tau_old_scatter if mode == "old_scatter" else fit_tau_profile_chi2

    for thr in sorted(counts["threshold_pe"].dropna().unique()):
        dthr = counts[counts["threshold_pe"] == thr]

        for seg in SEGMENTS:
            ds = dthr[dthr["segment"] == seg]
            tau, dtau, n = fit_fn(ds)
            rows.append({"threshold_pe": thr, "segment": seg, "tau_s": tau, "dtau_s": dtau, "n_points": n})

        overall = (
            dthr[dthr["norm_count"] > 0]
            .groupby(["run", "hold_time_s"], dropna=False)
            .agg(norm_count=("norm_count", "sum"), norm_count_var=("norm_count_var", "sum"))
            .reset_index()
        )
        tau, dtau, n = fit_fn(overall)
        rows.append({"threshold_pe": thr, "segment": "overall", "tau_s": tau, "dtau_s": dtau, "n_points": n})

    out = pd.DataFrame(rows)
    save_df(out, Path(out_dir) / "tables" / "lifetime_vs_threshold.csv")

    fig, ax = fig1()
    for seg in [*SEGMENTS, "overall"]:
        d = out[out["segment"] == seg]
        if len(d):
            ax.errorbar(d["threshold_pe"], d["tau_s"], yerr=d["dtau_s"], marker="o", capsize=2, label=seg)
    ax.set_xlabel("PE threshold")
    ax.set_ylabel(r"Lifetime $\tau$ [s]")
    ax.set_title(f"Lifetime vs PE threshold ({mode})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    savefig(Path(out_dir) / "lifetime" / "tau_vs_threshold.png")
    return out


def attach_region_start(pulses, summary):
    if pulses.empty or summary.empty:
        return pulses.copy()

    keys = ["run", "segment", "hold_time_s"]
    cols = keys + [
        "background_start_us", "background_end_us",
        "signal_start_us", "signal_end_us",
        "end_start_us", "end_end_us",
    ]
    s = summary[cols].drop_duplicates(keys)
    p = pulses.merge(s, on=keys, how="left")

    p["region_start_us"] = np.nan
    for reg, col in {
        "background": "background_start_us",
        "signal": "signal_start_us",
        "end": "end_start_us",
    }.items():
        p.loc[p["region"] == reg, "region_start_us"] = p.loc[p["region"] == reg, col]

    p["time_rel_s"] = (p["time_us"] - p["region_start_us"]) * 1e-6
    return p


def plot_inter_neutron_separation(pulses, out_dir, threshold, region, max_s):
    d = pulses[pulses["amplitude_pe"] >= threshold].copy()
    if region != "all":
        d = d[d["region"] == region]
    if d.empty:
        print("[warn] no pulses for inter-neutron separation")
        return

    d = d.sort_values(["region", "run", "segment", "time_us"])
    group_cols = ["run", "segment"] if region != "all" else ["region", "run", "segment"]
    d["next_time_us"] = d.groupby(group_cols)["time_us"].shift(-1)
    d["dt_s"] = (d["next_time_us"] - d["time_us"]) * 1e-6
    sep = d[np.isfinite(d["dt_s"]) & (d["dt_s"] > 0)].copy()
    if max_s > 0:
        sep = sep[sep["dt_s"] <= max_s]
    if sep.empty:
        return

    save_df(sep[["run", "segment", "region", "time_us", "next_time_us", "dt_s", "amplitude_pe"]],
            Path(out_dir) / "tables" / f"inter_neutron_separation_{region}_pe{threshold:g}.csv")

    fig, ax = fig1()
    for seg in SEGMENTS:
        vals = sep.loc[sep["segment"] == seg, "dt_s"].to_numpy()
        vals = vals[vals > 0]
        if len(vals):
            bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()), 80) if vals.max() > vals.min() else 50
            ax.hist(vals, bins=bins, histtype="step", label=f"seg {seg}")
    ax.set_xscale("log")
    ax.set_xlabel(r"Inter-neutron separation $\Delta t$ [s]")
    ax.set_ylabel("Pairs")
    ax.set_title(f"Inter-neutron separation, {region}, PE $\\geq$ {threshold:g}")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    savefig(Path(out_dir) / "inter_neutron" / f"inter_neutron_sep_{region}_pe{threshold:g}.png")


def plot_pe_spectra(pulses, out_dir, pe_max):
    if pulses.empty:
        return
    bins = np.arange(0, pe_max + 1, 1)
    for region in [r for r in REGIONS if r in set(pulses["region"])]:
        fig, axes = figgrid()
        d = pulses[pulses["region"] == region]
        for ax, seg in zip(axes.ravel(), SEGMENTS):
            vals = d.loc[d["segment"] == seg, "amplitude_pe"].dropna()
            if len(vals):
                ax.hist(vals, bins=bins, histtype="step")
            ax.set_title(f"{region}, seg {seg}")
            ax.set_xlabel("PE")
            ax.set_ylabel("Counts")
            ax.grid(True, alpha=0.3)
        savefig(Path(out_dir) / "pe_spectra" / f"pe_by_segment_{region}.png")


def plot_nll(windows, out_dir, nll_max):
    if windows.empty:
        return
    for region in [r for r in REGIONS if r in set(windows["region"])]:
        d = windows[(windows["region"] == region) & windows["final_nll"].notna()]
        fig, ax = fig1()
        for n in sorted(d["pulse_count"].dropna().unique()):
            vals = d.loc[d["pulse_count"] == n, "final_nll"].clip(upper=nll_max)
            if len(vals):
                ax.hist(vals, bins=80, histtype="step", label=f"{int(n)} pulses")
        ax.set_yscale("log")
        ax.set_xlabel(r"$-\log L$")
        ax.set_ylabel("Windows")
        ax.set_title(f"NLL by pulse count, {region}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        savefig(Path(out_dir) / "nll" / f"nll_by_pulse_count_{region}.png")


def plot_obs_expected(windows, out_dir):
    if windows.empty or not {"observed_count", "expected_count"}.issubset(windows.columns):
        return
    for region in [r for r in REGIONS if r in set(windows["region"])]:
        d = windows[windows["region"] == region].dropna(subset=["observed_count", "expected_count"])
        if d.empty:
            continue
        hi = max(10.0, float(np.nanpercentile(np.r_[d["observed_count"], d["expected_count"]], 99) * 1.1))
        fig, axes = figgrid()
        for ax, seg in zip(axes.ravel(), SEGMENTS):
            ds = d[d["segment"] == seg]
            ax.scatter(ds["observed_count"], ds["expected_count"], s=4, alpha=0.35)
            ax.plot([0, hi], [0, hi], "--", lw=1)
            ax.set_title(f"{region}, seg {seg}")
            ax.set_xlabel(r"$N_{\rm obs}$")
            ax.set_ylabel(r"$N_{\rm exp}$")
            ax.grid(True, alpha=0.3)
        savefig(Path(out_dir) / "obs_expected" / f"obs_vs_expected_{region}.png")


def plot_time_distribution(pulses, summary, out_dir):
    p = attach_region_start(pulses, summary)
    if p.empty:
        return
    bins = np.linspace(0, 60, 121)
    for region in [r for r in REGIONS if r in set(p["region"])]:
        d = p[(p["region"] == region) & (p["time_rel_s"] >= 0) & (p["time_rel_s"] < 60)]
        fig, ax = fig1()
        for seg in SEGMENTS:
            vals = d.loc[d["segment"] == seg, "time_rel_s"].dropna()
            if len(vals):
                ax.hist(vals, bins=bins, histtype="step", label=f"seg {seg}")
        ax.set_xlabel("Time since region start [s]")
        ax.set_ylabel("Pulses")
        ax.set_title(f"Pulse time distribution, {region}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        savefig(Path(out_dir) / "time" / f"time_distribution_{region}.png")


def plot_window_separation(windows, out_dir):
    if windows.empty:
        return
    d = windows.sort_values(["region", "run", "segment", "start_time_us"]).copy()
    d["next_start_us"] = d.groupby(["region", "run", "segment"])["start_time_us"].shift(-1)
    d["sep_s"] = (d["next_start_us"] - d["end_time_us"]) * 1e-6
    d = d[np.isfinite(d["sep_s"]) & (d["sep_s"] > 0)].copy()
    save_df(d[["run", "segment", "region", "window_index", "sep_s"]],
            Path(out_dir) / "tables" / "window_separations.csv")
    for region in [r for r in REGIONS if r in set(d["region"])]:
        fig, ax = fig1()
        dr = d[d["region"] == region]
        for seg in SEGMENTS:
            vals = dr.loc[dr["segment"] == seg, "sep_s"].to_numpy()
            if len(vals):
                bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()), 80) if vals.max() > vals.min() else 50
                ax.hist(vals, bins=bins, histtype="step", label=f"seg {seg}")
        ax.set_xscale("log")
        ax.set_xlabel(r"Next window start $-$ current window end [s]")
        ax.set_ylabel("Windows")
        ax.set_title(f"Window separation, {region}")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend()
        savefig(Path(out_dir) / "window_separation" / f"window_separation_{region}.png")


def plot_counts(counts, out_dir):
    if counts.empty:
        return
    save_df(counts, Path(out_dir) / "tables" / "threshold_counts_by_run_segment.csv")

    agg = counts.groupby(["threshold_pe", "segment"]).agg(
        raw_sum=("raw_count", "sum"),
        norm_sum=("norm_count", "sum"),
        n_points=("run", "count"),
    ).reset_index()
    save_df(agg, Path(out_dir) / "tables" / "counts_vs_threshold_by_segment.csv")

    fig, ax = fig1()
    for seg in SEGMENTS:
        d = agg[agg["segment"] == seg]
        if len(d):
            ax.plot(d["threshold_pe"], d["raw_sum"], marker="o", label=f"seg {seg}")
    ax.set_xlabel("PE threshold")
    ax.set_ylabel("Corrected count sum")
    ax.set_title("Corrected counts vs PE threshold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    savefig(Path(out_dir) / "thresholds" / "corrected_counts_vs_threshold.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=Path("output/merged"))
    ap.add_argument("--out-dir", type=Path, default=Path("output/graphs/epoch"))
    ap.add_argument("--epoch-info", type=Path, default=Path("config/epoch_info.json"))
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--epoch", type=str, default="2")
    ap.add_argument("--rde-csv", type=Path, default=None)
    ap.add_argument("--subtract", choices=["background", "end", "sideband", "none"], default="sideband")
    ap.add_argument("--norm", choices=["rhdd", "segment_rde", "fill", "none"], default="rhdd")
    ap.add_argument("--lifetime-fit", choices=["old_scatter", "profile_chi2"], default="old_scatter")
    ap.add_argument("--threshold-min", type=float, default=5.0)
    ap.add_argument("--threshold-max", type=float, default=20.0)
    ap.add_argument("--threshold-step", type=float, default=1.0)
    ap.add_argument("--pe-max", type=float, default=200.0)
    ap.add_argument("--nll-max", type=float, default=1500.0)
    ap.add_argument("--sep-region", choices=["signal", "background", "end", "all"], default="signal")
    ap.add_argument("--sep-threshold", type=float, default=None)
    ap.add_argument("--sep-max-s", type=float, default=60.0)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    r0, r1 = load_epoch_range(args.epoch_info, args.year, args.epoch)

    pulses, windows, summary = load_outputs(args.input_dir)
    pulses, windows, summary = coerce(pulses, windows, summary)

    pulses = filter_epoch(pulses, r0, r1)
    windows = filter_epoch(windows, r0, r1)
    summary = filter_epoch(summary, r0, r1)

    rde = load_rde(args.rde_csv, r0, r1)

    print(f"[info] year={args.year}, epoch={args.epoch}, runs=[{r0},{r1}]")
    print(f"[info] pulses={len(pulses)}, windows={len(windows)}, summary={len(summary)}")

    save_df(pulses, args.out_dir / "tables" / "epoch_pulses_used.csv")
    save_df(windows, args.out_dir / "tables" / "epoch_windows_used.csv")
    save_df(summary, args.out_dir / "tables" / "epoch_summary_used.csv")

    plot_pe_spectra(pulses, args.out_dir, args.pe_max)
    plot_nll(windows, args.out_dir, args.nll_max)
    plot_obs_expected(windows, args.out_dir)
    plot_window_separation(windows, args.out_dir)
    plot_time_distribution(pulses, summary, args.out_dir)

    sep_thr = args.sep_threshold if args.sep_threshold is not None else args.threshold_min
    plot_inter_neutron_separation(pulses, args.out_dir, sep_thr, args.sep_region, args.sep_max_s)

    thresholds = np.arange(args.threshold_min, args.threshold_max + 0.5 * args.threshold_step, args.threshold_step)
    counts = summarize_counts(pulses, summary, rde, thresholds, args.subtract, args.norm)
    plot_counts(counts, args.out_dir)
    lifetime_vs_threshold(counts, args.out_dir, args.lifetime_fit)


if __name__ == "__main__":
    main()