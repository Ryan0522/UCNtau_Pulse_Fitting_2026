#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


PE_BINS = np.arange(0, 201, 1)
DEFAULT_HOLD_TIMES = (20, 50, 100, 200, 1550)


def read_csvs(root, name):
    root = Path(root)
    task_dirs = sorted(root.glob("task_*"))
    dirs = task_dirs if task_dirs else [root]

    dfs = []
    for d in dirs:
        path = d / name
        if path.exists():
            dfs.append(pd.read_csv(path))

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def read_outputs(root):
    return {
        "pulses": read_csvs(root, "all_pulses.csv"),
        "coinc": read_csvs(root, "all_coincidences.csv"),
        "windows": read_csvs(root, "all_windows.csv"),
        "summary": read_csvs(root, "run_segment_summary.csv"),
    }

def clean(df):
    if df.empty:
        return df

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "run" in df.columns:
        df["run"] = pd.to_numeric(df["run"])
    if "segment" in df.columns:
        df["segment"] = df["segment"].astype(str)
    if "hold_time_s" in df.columns:
        df["hold_time_s"] = pd.to_numeric(df["hold_time_s"])
    if "region" in df.columns:
        df["region"] = df["region"].astype(str).str.lower().str.strip()

    return df

def add_region_time(events, summary):
    events = clean(events)
    summary = clean(summary)

    keys = ["run", "segment", "hold_time_s"]
    keep = keys + [
        "background_start_us",
        "background_end_us",
        "signal_start_us",
        "signal_end_us",
        "end_start_us",
        "end_end_us",
    ]

    meta = summary[keep].drop_duplicates(keys)
    out = events.merge(meta, on=keys, how="left")

    out["region_start_us"] = np.nan
    out["region_end_us"] = np.nan

    regions = [
        ("background", "background_start_us", "background_end_us"),
        ("signal", "signal_start_us", "signal_end_us"),
        ("end", "end_start_us", "end_end_us"),
    ]

    for region, start_col, end_col in regions:
        m = out["region"].eq(region)
        out.loc[m, "region_start_us"] = out.loc[m, start_col]
        out.loc[m, "region_end_us"] = out.loc[m, end_col]

    out["region_duration_s"] = (out["region_end_us"] - out["region_start_us"]) * 1.0e-6
    out["time_rel_s"] = (out["time_us"] - out["region_start_us"]) * 1.0e-6
    return out

def make_duration_table(summary):
    summary = clean(summary)
    keys = ["run", "segment", "hold_time_s"]
    cols = keys + [
        "background_start_us", "background_end_us",
        "signal_start_us", "signal_end_us",
    ]
    missing = [c for c in cols if c not in summary.columns]
    if missing:
        raise RuntimeError(f"summary is missing columns: {missing}")

    d = summary[cols].drop_duplicates(keys).copy()
    d["signal_duration_s"] = (d["signal_end_us"] - d["signal_start_us"]) * 1.0e-6
    d["background_duration_s"] = (d["background_end_us"] - d["background_start_us"]) * 1.0e-6
    return d[keys + ["signal_duration_s", "background_duration_s"]]

def pe_col(df):
    for c in ["amplitude_pe", "n_pe", "PE", "pe"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"No PE column found. Columns: {df.columns.tolist()}")

def savefig(outdir, name):
    path = Path(outdir) / name
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    print("wrote", path)

def exposure_by_hold(events, region):
    d = events[events["region"].eq(region)]
    if d.empty:
        return pd.Series(dtype=float)
    return (
        d[["run", "segment", "hold_time_s", "region_duration_s"]]
        .drop_duplicates()
        .groupby("hold_time_s")["region_duration_s"]
        .sum()
    )

def plot_pe_raw(events, outdir, label):
    col = pe_col(events)
    sig = events[events["region"].eq("signal")]
    rows = []

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for hold, g in sorted(sig.groupby("hold_time_s"), key=lambda x: x[0]):
        hist, edges = np.histogram(g[col], bins=PE_BINS)
        ax.step(edges[:-1], hist, where="post", label="{:g} s".format(hold))
        for x, y in zip(edges[:-1], hist):
            rows.append({"stream": label, "hold_time_s": hold, "pe_left": x, "count": y})

    ax.set_xlabel("PE")
    ax.set_ylabel("signal counts")
    ax.set_yscale("log")
    ax.set_title(f"{label} raw signal PE")
    ax.grid(True, alpha=0.3)
    ax.legend(title="hold")
    savefig(outdir, f"{label}_pe_raw.png")

    table = pd.DataFrame(rows)
    table.to_csv(Path(outdir) / f"{label}_pe_raw.csv", index=False)
    return table

def plot_pe_bgsub(events, outdir, label):
    col = pe_col(events)
    sig = events[events["region"].eq("signal")]
    bg = events[events["region"].eq("background")]
    sig_exp = exposure_by_hold(events, "signal")
    bg_exp = exposure_by_hold(events, "background")
    rows = []

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for hold in sorted(sig["hold_time_s"].dropna().unique()):
        s = sig[sig["hold_time_s"].eq(hold)]
        b = bg[bg["hold_time_s"].eq(hold)]

        sig_hist, edges = np.histogram(s[col], bins=PE_BINS)
        bg_hist, _ = np.histogram(b[col], bins=PE_BINS)
        scale = sig_exp.loc[hold] / bg_exp.loc[hold]
        sub = sig_hist.astype(float) - scale * bg_hist.astype(float)

        ax.step(edges[:-1], sub, where="post", label="{:g} s".format(hold))
        for x, raw_s, raw_b, y in zip(edges[:-1], sig_hist, bg_hist, sub):
            rows.append({
                "stream": label,
                "hold_time_s": hold,
                "pe_left": x,
                "signal_count": raw_s,
                "background_count": raw_b,
                "background_scale": scale,
                "bgsub_count": y,
            })

    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("PE")
    ax.set_ylabel("signal - scaled background")
    ax.set_title(f"{label} background-subtracted PE")
    ax.grid(True, alpha=0.3)
    ax.legend(title="hold")
    savefig(outdir, f"{label}_pe_bgsub.png")

    table = pd.DataFrame(rows)
    table.to_csv(Path(outdir) / f"{label}_pe_bgsub.csv", index=False)
    return table

def plot_threshold_bgsub(events, outdir, label, thresholds):
    col = pe_col(events)
    sig = events[events["region"].eq("signal")]
    bg = events[events["region"].eq("background")]
    sig_exp = exposure_by_hold(events, "signal")
    bg_exp = exposure_by_hold(events, "background")
    rows = []

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for hold in sorted(sig["hold_time_s"].dropna().unique()):
        s = sig[sig["hold_time_s"].eq(hold)]
        b = bg[bg["hold_time_s"].eq(hold)]
        scale = sig_exp.loc[hold] / bg_exp.loc[hold]
        n_total = len(s) - scale * len(b)

        y = []
        for thr in thresholds:
            n_pass = (s[col] >= thr).sum() - scale * (b[col] >= thr).sum()
            frac = n_pass / n_total if n_total > 0 else np.nan
            y.append(frac)
            rows.append({
                "stream": label,
                "hold_time_s": hold,
                "threshold_pe": thr,
                "signal_pass": sig_pass,
                "background_pass": bg_pass,
                "background_scale": scale,
                "bgsub_pass": n_pass,
                "bgsub_total": n_total,
                "bgsub_fraction": frac,
            })

        ax.plot(thresholds, y, marker=".", label="{:g} s".format(hold))

    ax.set_xlabel("PE threshold")
    ax.set_ylabel("bg-sub fraction passing")
    ax.set_title("{} bg-sub threshold survival".format(label))
    ax.grid(True, alpha=0.3)
    ax.legend(title="hold")
    savefig(outdir, "{}_threshold_bgsub.png".format(label))

    table = pd.DataFrame(rows)
    table.to_csv(Path(outdir) / f"{label}_threshold_bgsub.csv", index=False)
    return table

def plot_nearest_dt(events, outdir, label):
    sig = events[events["region"].eq("signal")].copy()
    sig = sig.sort_values(["run", "segment", "hold_time_s", "time_us"])
    group_cols = ["run", "segment", "hold_time_s"]
    g = sig.groupby(group_cols)["time_us"]
    sig["prev_dt_us"] = sig["time_us"] - g.shift(1)
    sig["next_dt_us"] = g.shift(-1) - sig["time_us"]
    sig["nearest_dt_us"] = np.nanmin(sig[["prev_dt_us", "next_dt_us"]].to_numpy(float), axis=1)

    rows = []
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.logspace(-1, 8, 120)

    for hold, h in sorted(sig.groupby("hold_time_s"), key=lambda x: x[0]):
        vals = h["nearest_dt_us"].to_numpy(float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=bins, histtype="step", density=True, label="{:g} s".format(hold))
        rows.append({
            "stream": label,
            "hold_time_s": hold,
            "n": len(vals),
            "median_dt_us": np.median(vals),
            "q10_dt_us": np.percentile(vals, 10),
            "q90_dt_us": np.percentile(vals, 90),
            "frac_lt_2us": np.mean(vals < 2.0),
            "frac_lt_10us": np.mean(vals < 10.0),
            "frac_lt_100us": np.mean(vals < 100.0),
        })

    ax.set_xscale("log")
    ax.set_xlabel("nearest signal-event separation [us]")
    ax.set_ylabel("density")
    ax.set_title(f"{label} nearest-neighbor separation")
    ax.grid(True, alpha=0.3)
    ax.legend(title="hold")
    savefig(outdir, f"{label}_nearest_dt.png")

    table = pd.DataFrame(rows)
    table.to_csv(Path(outdir) / f"{label}_nearest_dt.csv", index=False)
    return table

def plot_rate_vs_time(events, outdir, label, bin_s):
    sig = events[events["region"].eq("signal")].copy()
    sig = sig[np.isfinite(sig["time_rel_s"])]
    sig = sig[sig["time_rel_s"] >= 0.0]
    sig["time_bin_s"] = np.floor(sig["time_rel_s"] / bin_s) * bin_s

    counts = sig.groupby(["hold_time_s", "time_bin_s"]).size().reset_index(name="n_events")
    exposure = (
        sig[["run", "segment", "hold_time_s", "time_bin_s"]]
        .drop_duplicates()
        .groupby(["hold_time_s", "time_bin_s"])
        .size()
        .reset_index(name="n_run_segments")
    )
    counts = counts.merge(exposure, on=["hold_time_s", "time_bin_s"], how="left")
    counts["rate_per_s_per_run_segment"] = counts["n_events"] / (bin_s * counts["n_run_segments"])
    counts["stream"] = label

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for hold, h in sorted(counts.groupby("hold_time_s"), key=lambda x: x[0]):
        ax.plot(h["time_bin_s"], h["rate_per_s_per_run_segment"], marker=".", label="{:g} s".format(hold))

    ax.set_xlabel("time from signal start [s]")
    ax.set_ylabel("events / s / run-segment")
    ax.set_title(f"{label} signal rate vs time")
    ax.grid(True, alpha=0.3)
    ax.legend(title="hold")
    savefig(outdir, f"{label}_rate_vs_time.png")

    counts.to_csv(Path(outdir) / f"{label}_rate_vs_time.csv", index=False)
    return counts

def plot_windows(windows, outdir):
    windows = clean(windows)
    if "region" in windows.columns:
        windows = windows[windows["region"].eq("signal")]

    cols = [
        "pulse_count", "seed_count", "width_us", "observed_count", "expected_count",
        "fitted_pe_sum", "fit_expected_sum", "fixed_expected_sum", "background_expected_sum",
        "pe_per_observed_count", "background_fraction", "fit_fraction",
        "local_background_rate_hz", "local_background_gap_us",
    ]
    cols = [c for c in cols if c in windows.columns]
    rows = []

    for hold, h in sorted(windows.groupby("hold_time_s"), key=lambda x: x[0]):
        row = {"table": "window_summary", "hold_time_s": hold, "n_windows": len(h)}
        for col in cols:
            x = pd.to_numeric(h[col], errors="coerce")
            row[col + "_mean"] = x.mean()
            row[col + "_median"] = x.median()
            row[col + "_q90"] = x.quantile(0.90)
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(Path(outdir) / "window_summary.csv", index=False)

    for col in cols:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for hold, h in sorted(windows.groupby("hold_time_s"), key=lambda x: x[0]):
            vals = pd.to_numeric(h[col], errors="coerce").dropna()
            if vals.empty:
                continue
            ax.hist(vals, bins=80, histtype="step", density=True, label="{:g} s".format(hold))
        ax.set_xlabel(col)
        ax.set_ylabel("density")
        ax.set_title(f"window {col}")
        ax.grid(True, alpha=0.3)
        ax.legend(title="hold")
        savefig(outdir, f"window_{col}.png")

    return summary

def load_rhdd(path):
    if path is None:
        return pd.DataFrame()
    d = pd.read_csv(path)
    d.columns = [str(c).strip() for c in d.columns]
    if "run" not in d.columns:
        if "Run Number" in d.columns:
            d["run"] = d["Run Number"]
        else:
            raise RuntimeError("RDE CSV needs either `run` or `Run Number`.")
    if "RHDD" not in d.columns:
        raise RuntimeError("RDE CSV needs `RHDD` for the notebook-style lifetime fit.")
    d["run"] = pd.to_numeric(d["run"], errors="coerce")
    d["RHDD"] = pd.to_numeric(d["RHDD"], errors="coerce")
    return d[["run", "RHDD"]].dropna().drop_duplicates("run")

def make_threshold_counts(events, pe_name, threshold, duration_table, subtract_bg):
    keys = ["run", "segment", "hold_time_s"]
    d = events.copy()
    d = d[d["region"].isin(["signal", "background"])]
    d = d[np.isfinite(d[pe_name]) & (d[pe_name] >= threshold)]

    s_counts = d[d["region"].eq("signal")].groupby(keys).size().rename("s_count")
    b_counts = d[d["region"].eq("background")].groupby(keys).size().rename("b_count")

    out = duration_table.copy()
    out = out.merge(s_counts.reset_index(), on=keys, how="left")
    out = out.merge(b_counts.reset_index(), on=keys, how="left")
    out["s_count"] = out["s_count"].fillna(0.0)
    out["b_count"] = out["b_count"].fillna(0.0)

    out["bg_scale"] = out["signal_duration_s"] / out["background_duration_s"]
    bad_bg = (~np.isfinite(out["bg_scale"])) | (out["background_duration_s"] <= 0)
    out.loc[bad_bg, "bg_scale"] = np.nan

    if subtract_bg:
        out["net_count"] = out["s_count"] - out["bg_scale"] * out["b_count"]
        out["var_count"] = out["s_count"] + (out["bg_scale"] ** 2) * out["b_count"]
    else:
        out["net_count"] = out["s_count"]
        out["var_count"] = out["s_count"]

    return out

def add_deadtime(counts, k0, window_s):
    out = counts.copy()
    n = out["net_count"].astype(float)
    v = out["var_count"].astype(float)

    if k0 == 0:
        out["dt_denom"] = 1.0
        out["count_corr"] = n
        out["var_corr"] = v
        return out

    rate = n / window_s
    denom = 1.0 - rate * k0
    out["dt_denom"] = denom
    out["count_corr"] = n / denom
    out["var_corr"] = v / denom**4

    bad = (~np.isfinite(denom)) | (denom <= 0)
    out.loc[bad, ["count_corr", "var_corr"]] = np.nan
    return out

def summarize_hold_yield(counts, rhdd):
    out = counts.merge(rhdd, on="run", how="left") if not rhdd.empty else counts.copy()
    if "RHDD" not in out.columns:
        out["RHDD"] = 1.0

    g = (
        out.groupby("hold_time_s", as_index=False)
        .agg(
            count_corr_sum=("count_corr", "sum"),
            var_corr_sum=("var_corr", "sum"),
            RHDD_sum=("RHDD", "sum"),
            n_runs=("run", "nunique"),
            n_run_segments=("run", "size"),
            signal_raw_sum=("s_count", "sum"),
            background_raw_sum=("b_count", "sum"),
        )
    )
    g["yield_norm"] = g["count_corr_sum"] / g["RHDD_sum"]
    g["yield_err"] = np.sqrt(g["var_corr_sum"]) / g["RHDD_sum"]
    return g.sort_values("hold_time_s").reset_index(drop=True)

def exp_decay(t, A, tau):
    return A * np.exp(-t / tau)

def fit_tau_all_holds(hold_summary, hold_times, threshold, stream, subtract_bg, min_points=3):
    if curve_fit is None:
        raise RuntimeError(f"scipy.optimize.curve_fit is needed. Import error was: {SCIPY_IMPORT_ERROR!r}")

    d = hold_summary.copy()
    d = d[d["hold_time_s"].isin(hold_times)].sort_values("hold_time_s")
    d = d[
        np.isfinite(d["yield_norm"]) & np.isfinite(d["yield_err"]) &
        (d["yield_norm"] > 0) & (d["yield_err"] > 0)
    ].copy()

    row = {
        "stream": stream,
        "threshold_pe": threshold,
        "subtract_bg": subtract_bg,
        "n_points": len(d),
        "hold_times_used": " ".join(str(int(x)) for x in d["hold_time_s"].tolist()),
        "fit_ok": False,
        "A": np.nan,
        "dA": np.nan,
        "tau_s": np.nan,
        "dtau_s": np.nan,
        "dtau_chi2_scaled_s": np.nan,
        "chi2": np.nan,
        "dof": np.nan,
        "redchi2": np.nan,
        "status": "",
    }

    if len(d) < min_points:
        row["status"] = f"too few valid points: {len(d)}"
        return row, d

    t = d["hold_time_s"].to_numpy(float)
    y = d["yield_norm"].to_numpy(float)
    sy = d["yield_err"].to_numpy(float)

    try:
        logy = np.log(y)
        slog = sy / y
        good = np.isfinite(logy) & np.isfinite(slog) & (slog > 0)

        if np.sum(good) >= 2:
            slope, intercept = np.polyfit(t[good], logy[good], deg=1, w=1.0 / slog[good])
            tau0 = -1.0 / slope if slope < 0 else 880.0
            A0 = np.exp(intercept)
        else:
            tau0 = 880.0
            A0 = y[0] * np.exp(t[0] / tau0)

        if not np.isfinite(tau0) or tau0 <= 0:
            tau0 = 880.0
        if not np.isfinite(A0) or A0 <= 0:
            A0 = np.nanmax(y)

        popt, pcov = curve_fit(
            exp_decay,
            t,
            y,
            sigma=sy,
            p0=(A0, tau0),
            bounds=([0.0, 1.0], [np.inf, 10000.0]),
            absolute_sigma=True,
            maxfev=20000,
        )
        A, tau = popt
        dA, dtau = np.sqrt(np.diag(pcov))
        yfit = exp_decay(t, A, tau)
        pull = (y - yfit) / sy
        chi2 = float(np.sum(pull**2))
        dof = len(y) - 2
        redchi2 = chi2 / dof if dof > 0 else np.nan
        dtau_scaled = dtau * np.sqrt(redchi2) if np.isfinite(redchi2) and redchi2 > 1 else dtau

        row.update({
            "fit_ok": True,
            "A": A,
            "dA": dA,
            "tau_s": tau,
            "dtau_s": dtau,
            "dtau_chi2_scaled_s": dtau_scaled,
            "chi2": chi2,
            "dof": dof,
            "redchi2": redchi2,
            "status": "ok",
        })

        points = d.copy()
        points["stream"] = stream
        points["threshold_pe"] = threshold
        points["subtract_bg"] = subtract_bg
        points["fit_yield_norm"] = yfit
        points["pull"] = pull
        return row, points

    except Exception as exc:
        row["status"] = repr(exc)
        return row, d

def compute_lifetime_vs_pe(events, duration_table, rhdd, thresholds, hold_times, deadtime_k0, deadtime_window_s, stream):
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()

    col = pe_col(events)
    fit_rows = []
    point_rows = []

    for subtract_bg in [False, True]:
        for thr in thresholds:
            counts = make_threshold_counts(events, col, thr, duration_table, subtract_bg=subtract_bg)
            counts = add_deadtime(counts, k0=deadtime_k0, window_s=deadtime_window_s)
            ysum = summarize_hold_yield(counts, rhdd)
            fit_row, points = fit_tau_all_holds(
                ysum,
                hold_times=hold_times,
                threshold=thr,
                stream=stream,
                subtract_bg=subtract_bg,
            )
            fit_rows.append(fit_row)
            if not points.empty:
                point_rows.append(points)

    fit_df = pd.DataFrame(fit_rows)
    points_df = pd.concat(point_rows, ignore_index=True) if point_rows else pd.DataFrame()
    return fit_df, points_df

def plot_lifetime_vs_pe(fit_df, outdir, label, use_chi2_scaled_error=False):
    if fit_df.empty:
        return
    d = fit_df[fit_df["fit_ok"].eq(True)].copy()
    if d.empty:
        return

    err_col = "dtau_chi2_scaled_s" if use_chi2_scaled_error else "dtau_s"
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    for subtract_bg, g in d.groupby("subtract_bg"):
        g = g.sort_values("threshold_pe")
        mode = "bg-sub" if subtract_bg else "raw"
        ax.errorbar(
            g["threshold_pe"],
            g["tau_s"],
            yerr=g[err_col],
            marker="o",
            capsize=3,
            label=mode,
        )

    ax.set_xlabel("PE threshold")
    ax.set_ylabel(r"all-5-hold fitted $\tau$ [s]")
    ax.set_title(f"{label} lifetime vs PE threshold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    savefig(outdir, f"{label}_lifetime_vs_pe.png")

def plot_stream(events, outdir, label, thresholds, bin_s):
    if events.empty:
        return []
    return [
        plot_pe_raw(events, outdir, label),
        plot_pe_bgsub(events, outdir, label),
        plot_threshold_bgsub(events, outdir, label, thresholds),
        plot_nearest_dt(events, outdir, label),
        plot_rate_vs_time(events, outdir, label, bin_s),
    ]


def threshold_grid(min_value, max_value, step):
    return np.arange(min_value, max_value + 0.5 * step, step)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="Batch output directory or sweep value directory.")
    p.add_argument("--out", required=True, help="Plot/table output directory.")
    p.add_argument("--thresholds", nargs="+", type=float, default=[0, 5, 10, 20, 30, 40, 50])
    p.add_argument("--bin-s", type=float, default=5.0)

    p.add_argument("--rde-csv", default=None, help="CSV with run/Run Number and RHDD columns. Needed for notebook-style lifetime normalization.")
    p.add_argument("--lifetime-threshold-min", type=float, default=0.0)
    p.add_argument("--lifetime-threshold-max", type=float, default=21.0)
    p.add_argument("--lifetime-threshold-step", type=float, default=1.0)
    p.add_argument("--hold-times", nargs="+", type=float, default=list(DEFAULT_HOLD_TIMES))
    p.add_argument("--deadtime-k0", type=float, default=1.0e-6, help="Notebook-style correction. Use 0 to disable.")
    p.add_argument("--deadtime-window-s", type=float, default=60.0)
    p.add_argument("--skip-lifetime", action="store_true")
    p.add_argument("--chi2-scaled-error", action="store_true")
    args = p.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    data = read_outputs(args.dir)
    summary = clean(data["summary"])
    duration_table = make_duration_table(summary)

    pulses = add_region_time(data["pulses"], summary)
    coinc = add_region_time(data["coinc"], summary)
    windows = data["windows"]

    all_tables = []
    all_tables += plot_stream(pulses, outdir, "pulse", args.thresholds, args.bin_s)
    all_tables += plot_stream(coinc, outdir, "coinc", args.thresholds, args.bin_s)

    if not windows.empty:
        all_tables.append(plot_windows(windows, outdir))

    if not args.skip_lifetime:
        rhdd = load_rhdd(args.rde_csv) if args.rde_csv else pd.DataFrame()
        if rhdd.empty:
            print("[warn] --rde-csv not supplied. Lifetime will be normalized by 1 per run, not RHDD.")

        life_thresholds = threshold_grid(
            args.lifetime_threshold_min,
            args.lifetime_threshold_max,
            args.lifetime_threshold_step,
        )
        hold_times = tuple(args.hold_times)

        for label, events in [("pulse", pulses), ("coinc", coinc)]:
            if events.empty:
                continue
            fit_df, points_df = compute_lifetime_vs_pe(
                events=events,
                duration_table=duration_table,
                rhdd=rhdd,
                thresholds=life_thresholds,
                hold_times=hold_times,
                deadtime_k0=args.deadtime_k0,
                deadtime_window_s=args.deadtime_window_s,
                stream=label,
            )
            fit_df.to_csv(outdir / f"{label}_lifetime_vs_pe.csv", index=False)
            points_df.to_csv(outdir / f"{label}_lifetime_points_vs_pe.csv", index=False)
            plot_lifetime_vs_pe(
                fit_df,
                outdir,
                label,
                use_chi2_scaled_error=args.chi2_scaled_error,
            )
            all_tables.append(fit_df)

    nonempty = [t for t in all_tables if t is not None and not t.empty]
    if nonempty:
        pd.concat(nonempty, ignore_index=True, sort=False).to_csv(
            outdir / "observable_summary_all.csv",
            index=False,
        )

    print("done:", outdir)

if __name__ == "__main__":
    main()