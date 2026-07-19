#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


PE_BINS = np.arange(0, 201, 1)
DEFAULT_HOLD_TIMES = (20, 50, 100, 200, 1550)
DEFAULT_RUNINFO_SEGMENTS_BY_YEAR = {
    2021: ("12",),
    2022: ("12", "34", "56", "78"),
}
RUNINFO_SEGMENT_LABEL = {
    "1112": "56",
    "1314": "78",
}
RUNINFO_SEGMENT_ALIASES = {
    "56": ("56", "1112"),
    "78": ("78", "1314"),
    "1112": ("1112", "56"),
    "1314": ("1314", "78"),
}

def resolve_segment_for_counts(seg, available_segments):
    seg = str(seg).strip()
    candidates = RUNINFO_SEGMENT_ALIASES.get(seg, (seg,))
    for candidate in candidates:
        if candidate in available_segments:
            return candidate
    return seg

def output_segment_label(raw_seg):
    raw_seg = str(raw_seg).strip()
    return RUNINFO_SEGMENT_LABEL.get(raw_seg, raw_seg)

def read_csvs(root, name):
    root = Path(root)
    task_dirs = sorted(root.glob("task_*"))
    if not task_dirs:
        task_dirs = sorted(root.glob("*/task_*"))
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
        x = pd.to_numeric(df["segment"], errors="coerce")
        df["segment"] = df["segment"].astype(str).str.strip()
        m = x.notna()
        df.loc[m, "segment"] = x.loc[m].astype(int).astype(str)
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
            sig_pass = (s[col] >= thr).sum()
            background_pass = (b[col] >= thr).sum()
            
            n_pass = sig_pass - scale * background_pass
            frac = n_pass / n_total if n_total > 0 else np.nan
            y.append(frac)
            rows.append({
                "stream": label,
                "hold_time_s": hold,
                "threshold_pe": thr,
                "signal_pass": sig_pass,
                "background_pass": background_pass,
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
    keep = ["run", "RHDD"]
    if "Holding Time" in d.columns:
        d["hold_time_s"] = pd.to_numeric(d["Holding Time"], errors="coerce")
        keep.append("hold_time_s")
    return d[["run", "RHDD"]].dropna().drop_duplicates("run")

def attach_rhdd(counts, rhdd):
    if counts.empty:
        return pd.DataFrame()

    out = counts.copy()

    if rhdd.empty:
        out["RHDD"] = 1.0
        return out

    r = rhdd[["run", "RHDD"]].drop_duplicates("run").copy()
    out = out.merge(r, on="run", how="left")
    out = out[np.isfinite(out["RHDD"]) & (out["RHDD"] > 0)].copy()
    return out

def finish_yield_summary(out, group_cols):
    if out.empty:
        return pd.DataFrame()

    keys = list(group_cols) + ["hold_time_s"]

    g = (
        out.groupby(keys, as_index=False)
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

    g = g[g["RHDD_sum"] > 0].copy()
    g["yield_norm"] = g["count_corr_sum"] / g["RHDD_sum"]
    g["yield_err"] = np.sqrt(g["var_corr_sum"]) / g["RHDD_sum"]

    return g.sort_values(keys).reset_index(drop=True)

def summarize_overall_yield(counts, rhdd):
    if counts.empty:
        return pd.DataFrame()

    d = (
        counts.groupby(["run", "hold_time_s"], as_index=False)
        .agg(
            s_count=("s_count", "sum"),
            b_count=("b_count", "sum"),
            net_count=("net_count", "sum"),
            var_count=("var_count", "sum"),
            count_corr=("count_corr", "sum"),
            var_corr=("var_corr", "sum"),
        )
    )

    d = attach_rhdd(d, rhdd)
    return finish_yield_summary(d, [])

def summarize_segment_yield(counts, rhdd):
    if counts.empty:
        return pd.DataFrame()

    d = attach_rhdd(counts, rhdd)
    return finish_yield_summary(d, ["segment"])

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

def compute_instantaneous_deadtime_counts(events, pe_name, threshold, duration_table, deadtime_k0, subtract_bg, bin_width_s=5.0):
    keys = ["run", "segment", "hold_time_s"]
    
    d = events.copy()
    d = d[d["region"].isin(["signal", "background"])]
    d = d[np.isfinite(d[pe_name]) & (d[pe_name] >= threshold)].copy()
    
    d["time_bin"] = np.floor(d["time_rel_s"] / bin_width_s) * bin_width_s
    
    binned = d.groupby(keys + ["region", "time_bin"]).size().unstack(level="region", fill_value=0).reset_index()
    
    if "signal" not in binned.columns: binned["signal"] = 0
    if "background" not in binned.columns: binned["background"] = 0

    binned = binned.merge(duration_table, on=keys, how="left")
    binned["bg_scale"] = binned["signal_duration_s"] / binned["background_duration_s"]
    
    if subtract_bg:
        binned["bin_net"] = binned["signal"] - binned["bg_scale"] * binned["background"]
        binned["bin_var"] = binned["signal"] + (binned["bg_scale"] ** 2) * binned["background"]
    else:
        binned["bin_net"] = binned["signal"]
        binned["bin_var"] = binned["signal"]
        
    binned["bin_rate"] = binned["bin_net"] / bin_width_s
    binned["denom"] = 1.0 - (binned["bin_rate"] * deadtime_k0)
    
    binned["corr_net"] = binned["bin_net"] / binned["denom"]
    binned["corr_var"] = binned["bin_var"] / (binned["denom"] ** 4)
    
    invalid = (~np.isfinite(binned["denom"])) | (binned["denom"] <= 0)
    binned.loc[invalid, ["corr_net", "corr_var"]] = np.nan
    
    aggregated = binned.groupby(keys).agg(
        s_count=("signal", "sum"),
        b_count=("background", "sum"),
        net_count=("bin_net", "sum"),
        var_count=("bin_var", "sum"),
        count_corr=("corr_net", "sum"),
        var_corr=("corr_var", "sum")
    ).reset_index()
    
    aggregated = aggregated.merge(duration_table, on=keys, how="left")
    return aggregated

def compute_lifetime_vs_pe(events, duration_table, rhdd, thresholds, hold_times, deadtime_k0, deadtime_window_s, stream):
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()

    col = pe_col(events)
    fit_rows = []
    point_rows = []

    for subtract_bg in [False, True]:
        for thr in thresholds:
            counts = compute_instantaneous_deadtime_counts(
                events=events,
                pe_name=col,
                threshold=thr,
                duration_table=duration_table,
                deadtime_k0=deadtime_k0,
                subtract_bg=subtract_bg,
                bin_width_s=deadtime_window_s,
            )

            overall = summarize_overall_yield(counts, rhdd)
            fit_row, points = fit_tau_all_holds(
                overall,
                hold_times=hold_times,
                threshold=thr,
                stream=stream,
                subtract_bg=subtract_bg,
            )

            fit_row["lifetime_view"] = "overall"
            fit_row["segment"] = "all"
            fit_rows.append(fit_row)

            if not points.empty:
                points = points.copy()
                points["lifetime_view"] = "overall"
                points["segment"] = "all"
                point_rows.append(points)

            segments = sorted(counts["segment"].dropna().unique())

            if len(segments) > 1:
                by_segment = summarize_segment_yield(counts, rhdd)

                for seg in segments:
                    one = by_segment[by_segment["segment"].eq(seg)].copy()

                    fit_row, points = fit_tau_all_holds(
                        one,
                        hold_times=hold_times,
                        threshold=thr,
                        stream=stream,
                        subtract_bg=subtract_bg,
                    )

                    fit_row["lifetime_view"] = "segment"
                    fit_row["segment"] = seg
                    fit_rows.append(fit_row)

                    if not points.empty:
                        points = points.copy()
                        points["lifetime_view"] = "segment"
                        points["segment"] = seg
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

    for subtract_bg, h in d.groupby("subtract_bg"):
        mode = "bgsub" if subtract_bg else "raw"

        fig, ax = plt.subplots(figsize=(8.0, 5.0))

        overall = h[h["lifetime_view"].eq("overall")].sort_values("threshold_pe")
        if not overall.empty:
            ax.errorbar(
                overall["threshold_pe"],
                overall["tau_s"],
                yerr=overall[err_col],
                marker="o",
                capsize=3,
                linewidth=1.8,
                label="overall",
            )

        segs = sorted(h.loc[h["lifetime_view"].eq("segment"), "segment"].dropna().unique())

        for seg in segs:
            g = h[
                h["lifetime_view"].eq("segment") &
                h["segment"].eq(seg)
            ].sort_values("threshold_pe")

            if g.empty:
                continue

            ax.errorbar(
                g["threshold_pe"],
                g["tau_s"],
                yerr=g[err_col],
                marker=".",
                capsize=2,
                linewidth=1.2,
                label=f"seg {seg}",
            )

        ax.set_xlabel("PE threshold")
        ax.set_ylabel(r"fitted $\tau$ [s]")
        ax.set_title(f"{label} lifetime vs PE threshold ({mode})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        savefig(outdir, f"{label}_lifetime_vs_pe_{mode}.png")

def compute_next_dt(events, group_cols):
    if events.empty:
        return pd.DataFrame()

    d = events.sort_values(group_cols + ["time_us"]).copy()
    d["next_time_us"] = d.groupby(group_cols)["time_us"].shift(-1)
    d["dt_s"] = (d["next_time_us"] - d["time_us"]) * 1.0e-6
    d = d[np.isfinite(d["dt_s"]) & (d["dt_s"] > 0)].copy()
    return d

def plot_inter_neutron_plots(pulses, coinc, outdir, pulse_thr=10.0, coinc_thr=10.0, nbins=80):
    rows = []

    for label, events, thr in [
        ("pulse", pulses, pulse_thr),
        ("coinc", coinc, coinc_thr),
    ]:
        if events.empty:
            continue

        col = pe_col(events)
        d = events[
            events["region"].eq("signal") &
            np.isfinite(events[col]) &
            (events[col] >= thr)
        ].copy()

        sep = compute_next_dt(d, ["run", "segment", "hold_time_s", "region"])
        if sep.empty:
            continue

        sep["stream"] = label
        sep.to_csv(Path(outdir) / f"{label}_inter_neutron_separation.csv", index=False)
        rows.append(sep)

    if not rows:
        return pd.DataFrame()

    sep = pd.concat(rows, ignore_index=True)

    holds = sorted(sep["hold_time_s"].dropna().unique())
    fw, fh = 6.47, 4.85
    fig, axes = plt.subplots(len(holds), 2, figsize=(2.0 * fw, len(holds) * fh), squeeze=False)

    for i, hold in enumerate(holds):
        for j, label in enumerate(["pulse", "coinc"]):
            ax = axes[i, j]
            h = sep[(sep["hold_time_s"].eq(hold)) & (sep["stream"].eq(label))]

            for seg, g in sorted(h.groupby("segment"), key=lambda x: x[0]):
                vals = g["dt_s"].to_numpy(float)
                vals = vals[np.isfinite(vals) & (vals > 0)]
                if len(vals) < 2:
                    continue
                if vals.max() <= vals.min():
                    continue

                bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()), nbins)
                ax.hist(vals, bins=bins, histtype="step", linewidth=1.2, label=f"seg {seg}")

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(r"inter-neutron separation $\Delta t$ [s]")
            ax.set_ylabel("pairs")
            ax.set_title(f"{label}, hold {hold:g} s")
            ax.grid(True, alpha=0.3, which="both")

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=8)

    savefig(outdir, "method_comparison_inter_neutron_separation_by_hold.png")

    fig, axes = plt.subplots(1, 2, figsize=(2.0 * fw, fh), squeeze=False)

    hist_records = []

    for j, label in enumerate(["pulse", "coinc"]):
        ax = axes[0, j]
        h = sep[sep["stream"].eq(label)]

        vals = h["dt_s"].to_numpy(float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals) >= 2 and vals.max() > vals.min():
            bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()), nbins)
            counts, edge_bins = np.histogram(vals, bins=bins)  # pre-calculate counts
            ax.hist(bins[:-1], bins=bins, weights=counts, histtype="step", linewidth=1.5)

            for left, right, count in zip(edge_bins[:-1], edge_bins[1:], counts):
                hist_records.append({
                    "stream": label,
                    "bin_left_s": left,
                    "bin_right_s": right,
                    "count": count
                })

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"inter-neutron separation $\Delta t$ [s]")
        ax.set_ylabel("pairs")
        ax.set_title(f"{label}, overall")
        ax.grid(True, alpha=0.3, which="both")

    savefig(outdir, "method_comparison_inter_neutron_separation_overall.png")
    
    if hist_records:
        hist_df = pd.DataFrame(hist_records)
        hist_df.to_csv(Path(outdir) / "method_comparison_inter_neutron_separation_distributions.csv", index=False)
    
    return sep

def plot_rate_close_dashboard(events, outdir, label, pe_threshold=10.0, close_thr_s=100.0e-6, bin_s=0.5):
    if events.empty:
        return pd.DataFrame()

    col = pe_col(events)
    d = events[
        events["region"].eq("signal") &
        np.isfinite(events[col]) &
        (events[col] >= pe_threshold) &
        np.isfinite(events["time_rel_s"])
    ].copy()

    if d.empty:
        return pd.DataFrame()

    t_min, t_max = 0.0, 60.0
    bins = np.arange(t_min, t_max + bin_s, bin_s)
    centers = 0.5 * (bins[:-1] + bins[1:])

    rows = []

    for (run, seg, hold), g in d.groupby(["run", "segment", "hold_time_s"]):
        vals = g["time_rel_s"].to_numpy(float)
        vals = vals[np.isfinite(vals) & (vals >= t_min) & (vals < t_max)]
        counts, _ = np.histogram(vals, bins=bins)

        rows.append(pd.DataFrame({
            "run": run,
            "segment": seg,
            "hold_time_s": hold,
            "time_center_s": centers,
            "bin_width_s": bin_s,
            "count": counts,
            "rate_hz": counts / bin_s,
        }))

    if not rows:
        return pd.DataFrame()

    rate = pd.concat(rows, ignore_index=True)

    dt = compute_next_dt(d, ["run", "segment", "hold_time_s"])
    if dt.empty:
        return rate

    dt["is_close"] = dt["dt_s"] < close_thr_s
    dt["time_bin"] = pd.cut(
        dt["time_rel_s"],
        bins=bins,
        right=False,
        include_lowest=True,
        labels=False,
    )
    dt = dt.dropna(subset=["time_bin"]).copy()
    dt["time_center_s"] = centers[dt["time_bin"].to_numpy(int)]

    close = (
        dt[dt["is_close"]]
        .groupby(["run", "segment", "hold_time_s", "time_center_s"])
        .size()
        .reset_index(name="n_close")
    )

    rate = rate.merge(
        close,
        on=["run", "segment", "hold_time_s", "time_center_s"],
        how="left",
    )
    rate["n_close"] = rate["n_close"].fillna(0.0)
    rate["unit"] = (
        rate["run"].astype(str) + "_" +
        rate["segment"].astype(str) + "_" +
        rate["hold_time_s"].astype(str)
    )

    by_hold = (
        rate.groupby(["hold_time_s", "time_center_s", "bin_width_s"], as_index=False)
        .agg(
            total_count=("count", "sum"),
            total_close=("n_close", "sum"),
            n_units=("unit", "nunique"),
        )
    )

    by_hold["rate_hz"] = by_hold["total_count"] / (by_hold["n_units"] * bin_s)
    by_hold["rate_err_hz"] = np.sqrt(by_hold["total_count"]) / (by_hold["n_units"] * bin_s)
    by_hold["p_close"] = by_hold["total_close"] / by_hold["total_count"].where(by_hold["total_count"] > 0, np.nan)
    by_hold["pct_close"] = 100.0 * by_hold["p_close"]
    by_hold["pct_close_err"] = 100.0 * np.sqrt(
        by_hold["p_close"] * (1.0 - by_hold["p_close"]) /
        by_hold["total_count"].where(by_hold["total_count"] > 0, np.nan)
    )
    by_hold["view"] = "by_hold"
    by_hold.to_csv(Path(outdir) / f"{label}_instantaneous_frequency_rate_summary_by_hold.csv", index=False)

    overall = (
        rate.groupby(["time_center_s", "bin_width_s"], as_index=False)
        .agg(
            total_count=("count", "sum"),
            total_close=("n_close", "sum"),
            n_units=("unit", "nunique"),
        )
    )

    overall["rate_hz"] = overall["total_count"] / (overall["n_units"] * bin_s)
    overall["rate_err_hz"] = np.sqrt(overall["total_count"]) / (overall["n_units"] * bin_s)
    overall["p_close"] = overall["total_close"] / overall["total_count"].where(overall["total_count"] > 0, np.nan)
    overall["pct_close"] = 100.0 * overall["p_close"]
    overall["pct_close_err"] = 100.0 * np.sqrt(
        overall["p_close"] * (1.0 - overall["p_close"]) /
        overall["total_count"].where(overall["total_count"] > 0, np.nan)
    )
    overall["hold_time_s"] = "overall"
    overall["view"] = "overall"
    overall.to_csv(Path(outdir) / f"{label}_instantaneous_frequency_rate_summary_overall.csv", index=False)

    holds = sorted(by_hold["hold_time_s"].dropna().unique())
    fw, fh = 6.47, 4.85
    fig, axes = plt.subplots(len(holds), 2, figsize=(2.2 * fw, len(holds) * fh * 0.95), sharex="col", squeeze=False)

    rate_max = by_hold["rate_hz"].max()
    rate_max = rate_max if np.isfinite(rate_max) and rate_max > 0 else 10.0
    x = np.linspace(0.0, 1.05 * rate_max, 400)
    y_poisson = 100.0 * (1.0 - np.exp(-x * close_thr_s))

    for i, hold in enumerate(holds):
        ds = by_hold[by_hold["hold_time_s"].eq(hold)].sort_values("time_center_s").copy()

        ax = axes[i, 0]
        ax.step(ds["time_center_s"], ds["rate_hz"], where="mid", linewidth=1.6)
        ax.fill_between(
            ds["time_center_s"],
            ds["rate_hz"] - ds["rate_err_hz"],
            ds["rate_hz"] + ds["rate_err_hz"],
            step="mid",
            alpha=0.15,
        )
        ax.set_ylabel(f"hold {hold:g} s\nrate [Hz]")
        ax.grid(True, alpha=0.3)

        ax = axes[i, 1]
        ds2 = ds[
            (ds["total_count"] > 0) &
            np.isfinite(ds["rate_hz"]) &
            np.isfinite(ds["pct_close"])
        ].copy()

        if not ds2.empty:
            ax.errorbar(
                ds2["rate_hz"],
                ds2["pct_close"],
                xerr=ds2["rate_err_hz"],
                yerr=ds2["pct_close_err"],
                fmt="o",
                ms=4,
                alpha=0.75,
                capsize=2,
            )

        ax.plot(x, y_poisson, linewidth=1.5, label="Poisson")

        high = ds2[ds2["rate_hz"] > 10.0]
        if len(high) >= 2:
            m, b = np.polyfit(high["rate_hz"], high["pct_close"], deg=1)
            ax.plot(x, m * x + b, linestyle="--", linewidth=1.5, label=f"slope {m:.3g} %/Hz")

        ax.set_ylabel(f"close [%]\n" + r"$\Delta t_{\rm next} < " + f"{close_thr_s * 1e6:.0f}" + r"\,\mu s$")
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.3)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)

    axes[-1, 0].set_xlabel(r"$t_{\rm rel}$ [s]")
    axes[-1, 1].set_xlabel("instantaneous frequency [Hz]")
    axes[0, 0].set_title(f"{label} instantaneous rate by hold")
    axes[0, 1].set_title("close-neutron fraction vs rate by hold")
    savefig(outdir, f"{label}_instantaneous_frequency_dashboard_by_hold.png")

    fig, axes = plt.subplots(1, 2, figsize=(2.2 * fw, fh), squeeze=False)

    ds = overall.sort_values("time_center_s").copy()

    ax = axes[0, 0]
    ax.step(ds["time_center_s"], ds["rate_hz"], where="mid", linewidth=1.6)
    ax.fill_between(
        ds["time_center_s"],
        ds["rate_hz"] - ds["rate_err_hz"],
        ds["rate_hz"] + ds["rate_err_hz"],
        step="mid",
        alpha=0.15,
    )
    ax.set_xlabel(r"$t_{\rm rel}$ [s]")
    ax.set_ylabel("rate [Hz]")
    ax.set_title(f"{label} instantaneous rate overall")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ds2 = ds[
        (ds["total_count"] > 0) &
        np.isfinite(ds["rate_hz"]) &
        np.isfinite(ds["pct_close"])
    ].copy()

    rate_max = ds["rate_hz"].max()
    rate_max = rate_max if np.isfinite(rate_max) and rate_max > 0 else 10.0
    x = np.linspace(0.0, 1.05 * rate_max, 400)
    y_poisson = 100.0 * (1.0 - np.exp(-x * close_thr_s))

    if not ds2.empty:
        ax.errorbar(
            ds2["rate_hz"],
            ds2["pct_close"],
            xerr=ds2["rate_err_hz"],
            yerr=ds2["pct_close_err"],
            fmt="o",
            ms=4,
            alpha=0.75,
            capsize=2,
        )

    ax.plot(x, y_poisson, linewidth=1.5, label="Poisson")

    high = ds2[ds2["rate_hz"] > 10.0]
    if len(high) >= 2:
        m, b = np.polyfit(high["rate_hz"], high["pct_close"], deg=1)
        ax.plot(x, m * x + b, linestyle="--", linewidth=1.5, label=f"slope {m:.3g} %/Hz")

    ax.set_xlabel("instantaneous frequency [Hz]")
    ax.set_ylabel(f"close [%]\n" + r"$\Delta t_{\rm next} < " + f"{close_thr_s * 1e6:.0f}" + r"\,\mu s$")
    ax.set_ylim(bottom=0.0)
    ax.set_title("close-neutron fraction vs rate overall")
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)

    savefig(outdir, f"{label}_instantaneous_frequency_dashboard_overall.png")

    return pd.concat([by_hold, overall], ignore_index=True, sort=False)

def default_runinfo_segments(year, summary):
    if year in DEFAULT_RUNINFO_SEGMENTS_BY_YEAR:
        return list(DEFAULT_RUNINFO_SEGMENTS_BY_YEAR[year])

    if summary.empty or "segment" not in summary.columns:
        return []

    segs = summary["segment"].dropna().astype(str).str.strip().unique().tolist()
    return sorted(segs, key=lambda x: (len(x), x))

def make_runinfo_like_table(
    events,
    summary,
    year,
    threshold,
    deadtime_k0,
    deadtime_window_s,
    segments=None,
):
    summary = clean(summary)

    if summary.empty:
        return pd.DataFrame()

    if segments is None:
        segments = default_runinfo_segments(year, summary)
    else:
        segments = [str(s).strip() for s in segments]

    available_segments = set(summary["segment"].dropna().astype(str).str.strip())

    seg_pairs = []
    seen_labels = set()

    for seg in segments:
        raw_seg = resolve_segment_for_counts(seg, available_segments)
        out_seg = output_segment_label(raw_seg)

        if out_seg in seen_labels:
            continue

        seg_pairs.append((raw_seg, out_seg))
        seen_labels.add(out_seg)

    base = (
        summary[["run", "hold_time_s"]]
        .drop_duplicates()
        .rename(columns={"run": "Run Number", "hold_time_s": "Holding Time"})
        .sort_values(["Run Number", "Holding Time"])
        .reset_index(drop=True)
    )

    seg_cols = {}
    for raw_seg, out_seg in seg_pairs:
        seg_cols[out_seg] = [
            f"Un{out_seg}",
            f"Bg{out_seg}",
            f"Un{out_seg} DT",
            f"Bg{out_seg} DT",
            f"Un{out_seg} RDE",
            f"Bg{out_seg} RDE",
        ]
        for colname in seg_cols[out_seg]:
            base[colname] = 0.0

    if events.empty or not seg_pairs:
        cols = ["Run Number", "Holding Time"]
        for raw_seg, out_seg in seg_pairs:
            cols += seg_cols[out_seg]
        return base[cols]

    duration_table = make_duration_table(summary)
    events = add_region_time(events, summary)

    if events.empty:
        cols = ["Run Number", "Holding Time"]
        for raw_seg, out_seg in seg_pairs:
            cols += seg_cols[out_seg]
        return base[cols]

    col = pe_col(events)

    raw_counts = compute_instantaneous_deadtime_counts(
        events=events,
        pe_name=col,
        threshold=threshold,
        duration_table=duration_table,
        deadtime_k0=deadtime_k0,
        subtract_bg=False,
        bin_width_s=deadtime_window_s,
    )

    bgsub_counts = compute_instantaneous_deadtime_counts(
        events=events,
        pe_name=col,
        threshold=threshold,
        duration_table=duration_table,
        deadtime_k0=deadtime_k0,
        subtract_bg=True,
        bin_width_s=deadtime_window_s,
    )

    if raw_counts.empty:
        cols = ["Run Number", "Holding Time"]
        for raw_seg, out_seg in seg_pairs:
            cols += seg_cols[out_seg]
        return base[cols]

    raw_counts = clean(raw_counts)
    bgsub_counts = clean(bgsub_counts)

    for d in [raw_counts, bgsub_counts]:
        if not d.empty and "segment" in d.columns:
            d["segment"] = d["segment"].astype(str).str.strip()

    keys = ["run", "segment", "hold_time_s"]

    raw_small = raw_counts[keys + ["s_count", "b_count", "count_corr"]].rename(
        columns={
            "s_count": "un",
            "b_count": "bg",
            "count_corr": "un_dt",
        }
    )

    bgsub_small = bgsub_counts[keys + ["count_corr"]].rename(
        columns={"count_corr": "un_rde"}
    )

    counts = raw_small.merge(bgsub_small, on=keys, how="left")
    counts["un_rde"] = counts["un_rde"].fillna(counts["un_dt"])

    # Effective background correction in signal-window units.
    # This makes the output reconstructable:
    #     UnXX RDE = UnXX DT - BgXX RDE
    counts["bg_rde"] = counts["un_dt"] - counts["un_rde"]
    counts["bg_dt"] = counts["bg_rde"]

    out = base.copy()

    for raw_seg, out_seg in seg_pairs:
        one = counts[counts["segment"].eq(str(raw_seg))].copy()
        if one.empty:
            continue

        one = (
            one.groupby(["run", "hold_time_s"], as_index=False)
            .agg(
                un=("un", "sum"),
                bg=("bg", "sum"),
                un_dt=("un_dt", "sum"),
                bg_dt=("bg_dt", "sum"),
                un_rde=("un_rde", "sum"),
                bg_rde=("bg_rde", "sum"),
            )
            .rename(
                columns={
                    "run": "Run Number",
                    "hold_time_s": "Holding Time",
                    "un": f"Un{out_seg}",
                    "bg": f"Bg{out_seg}",
                    "un_dt": f"Un{out_seg} DT",
                    "bg_dt": f"Bg{out_seg} DT",
                    "un_rde": f"Un{out_seg} RDE",
                    "bg_rde": f"Bg{out_seg} RDE",
                }
            )
        )

        out = out.drop(columns=seg_cols[out_seg]).merge(
            one,
            on=["Run Number", "Holding Time"],
            how="left",
        )

        out[f"Un{out_seg}"] = out[f"Un{out_seg}"].fillna(0).round().astype(int)
        out[f"Bg{out_seg}"] = out[f"Bg{out_seg}"].fillna(0).round().astype(int)

        out[f"Un{out_seg} DT"] = out[f"Un{out_seg} DT"].fillna(0.0)
        out[f"Bg{out_seg} DT"] = out[f"Bg{out_seg} DT"].fillna(0.0)

        out[f"Un{out_seg} RDE"] = out[f"Un{out_seg} RDE"].fillna(0.0)
        out[f"Bg{out_seg} RDE"] = out[f"Bg{out_seg} RDE"].fillna(0.0)

    cols = ["Run Number", "Holding Time"]
    for raw_seg, out_seg in seg_pairs:
        cols += seg_cols[out_seg]

    return out[cols].sort_values(["Run Number", "Holding Time"]).reset_index(drop=True)

def write_runinfo_like_csv(raw_data, raw_summary, args):
    streams = [args.runinfo_stream]
    if args.runinfo_stream == "both":
        streams = ["pulse", "coinc"]

    written = []

    for stream in streams:
        events = raw_data["pulses"] if stream == "pulse" else raw_data["coinc"]

        if events.empty:
            print(f"[warn] No {stream} events found. Skipping runinfo-style CSV.")
            continue

        table = make_runinfo_like_table(
            events=events,
            summary=raw_summary,
            year=args.year,
            threshold=args.runinfo_threshold,
            deadtime_k0=args.deadtime_k0,
            deadtime_window_s=args.deadtime_window_s,
            segments=args.runinfo_segments,
        )

        if table.empty:
            print(f"[warn] Could not build {stream} runinfo-style CSV.")
            continue

        outdir = Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)

        path = outdir / f"{stream}_runinfo_{args.year}_all.csv"
        table.to_csv(path, index=False)

        print("wrote", path)
        written.append(path)

    return written

def plot_stream(events, outdir, label, thresholds, bin_s):
    if events.empty:
        return []
    return [
        plot_pe_raw(events, outdir, label),
        plot_pe_bgsub(events, outdir, label),
        plot_threshold_bgsub(events, outdir, label, thresholds),
        plot_nearest_dt(events, outdir, label),
        plot_rate_vs_time(events, outdir, label, bin_s),
        plot_rate_close_dashboard(events, outdir, label),
    ]

def threshold_grid(min_value, max_value, step):
    return np.arange(min_value, max_value + 0.5 * step, step)

def get_epoch_range(epoch_info_path, year, epoch):
    with open(epoch_info_path) as f:
        info = json.load(f)
    try:
        rec = info[str(year)][str(epoch)]
        return int(rec["start_run_number"]), int(rec["end_run_number"])
    except KeyError:
        print(f"[warn] No run numbers found for year {year}, epoch {epoch}")
        return None, None

def filter_epoch(df, start_run, end_run):
    if df.empty or "run" not in df.columns:
        return df
    return df[(df["run"] >= start_run) & (df["run"] <= end_run)].copy()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="Batch output directory or sweep value directory.")
    p.add_argument("--out", required=True, help="Plot/table output directory.")
    p.add_argument("--epoch-info", default="config/epoch_info.json", help="Path to epoch_info.json")
    p.add_argument("--year", type=int, default=2021, help="Data year for the epoch config")
    p.add_argument("--epochs", nargs="+", type=int, default=[1, 2, 4, 5, 6], help="Epochs to process")
    
    p.add_argument("--thresholds", nargs="+", type=float, default=[0, 5, 10, 20, 30, 40, 50])
    p.add_argument("--bin-s", type=float, default=5.0)
    p.add_argument("--rde-csv", default=None, help="CSV with run/Run Number and RHDD columns. Needed for notebook-style lifetime normalization.")
    p.add_argument("--lifetime-threshold-min", type=float, default=0.0)
    p.add_argument("--lifetime-threshold-max", type=float, default=21.0)
    p.add_argument("--lifetime-threshold-step", type=float, default=1.0)
    p.add_argument("--hold-times", nargs="+", type=float, default=list(DEFAULT_HOLD_TIMES))
    p.add_argument("--deadtime-k0", type=float, default=3.687e-6, help="Notebook-style correction. Use 0 to disable.")
    p.add_argument("--deadtime-window-s", type=float, default=60.0)
    p.add_argument("--skip-lifetime", action="store_true")

    p.add_argument(
        "--runinfo-stream",
        choices=["pulse", "coinc", "both"],
        default="pulse",
        help="Save one runinfo-style CSV in --out before epoch filtering.",
    )
    p.add_argument(
        "--runinfo-threshold",
        type=float,
        default=0.0,
        help="PE threshold used for the runinfo-style Un counts.",
    )
    p.add_argument(
        "--runinfo-segments",
        nargs="+",
        default=None,
        help="Override runinfo segment suffixes. Defaults: 2021 -> 12; 2022 -> 12 34 56 78. For 2022, requested 56/78 can match internal 1112/1314 while writing CSV headers as 56/78.",    
    )

    p.add_argument("--chi2-scaled-error", action="store_true")
    args = p.parse_args()
    print(args)

    raw_data = read_outputs(args.dir)
    raw_summary = clean(raw_data["summary"])

    if raw_summary.empty:
        print(f"[error] No run summary data found in {args.dir}. Exiting.")
        return

    rhdd = load_rhdd(args.rde_csv) if args.rde_csv else pd.DataFrame()
    if rhdd.empty and not args.skip_lifetime:
        print("[warn] --rde-csv not supplied. Lifetime will be normalized by 1 per run, not RHDD.")

    write_runinfo_like_csv(raw_data, raw_summary, args)

    for epoch in args.epochs:
        print(f"\n--- Processing Epoch {epoch} ---")
        
        start_run, end_run = get_epoch_range(args.epoch_info, args.year, epoch)
        if start_run is None or end_run is None:
            print(f"Skipping Epoch {epoch}: Configuration missing.")
            continue
            
        # Filter the summary file to verify if data exists for this epoch
        epoch_summary = filter_epoch(raw_summary, start_run, end_run)
        if epoch_summary.empty:
            print(f"Skipping Epoch {epoch}: No runs found in the range [{start_run}, {end_run}].")
            continue

        # Establish centralized output directory specific to this epoch
        epoch_outdir = Path(args.out) / f"epoch_{epoch}"
        epoch_outdir.mkdir(parents=True, exist_ok=True)

        # Filter remaining streams
        duration_table = make_duration_table(epoch_summary)
        pulses = add_region_time(filter_epoch(raw_data["pulses"], start_run, end_run), epoch_summary)
        coinc = add_region_time(filter_epoch(raw_data["coinc"], start_run, end_run), epoch_summary)
        
        windows = pd.DataFrame()
        if not raw_data["windows"].empty:
            windows = filter_epoch(clean(raw_data["windows"]), start_run, end_run)
            if "region" in windows.columns:
                windows = windows[windows["region"].eq("signal")]

        # Skip processing entirely if there are no pulse or coincidence events
        if pulses.empty and coinc.empty:
            print(f"Skipping Epoch {epoch}: Filtered data tables are empty.")
            continue

        # 3. Generate non-task overall plots for this epoch
        all_tables = []
        all_tables += plot_stream(pulses, epoch_outdir, "pulse", args.thresholds, args.bin_s)
        all_tables += plot_stream(coinc, epoch_outdir, "coinc", args.thresholds, args.bin_s)
        all_tables.append(plot_inter_neutron_plots(pulses, coinc, epoch_outdir))

        if not windows.empty:
            all_tables.append(plot_windows(windows, epoch_outdir))

        if not args.skip_lifetime:
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
                fit_df.to_csv(epoch_outdir / f"{label}_lifetime_vs_pe.csv", index=False)
                points_df.to_csv(epoch_outdir / f"{label}_lifetime_points_vs_pe.csv", index=False)
                plot_lifetime_vs_pe(
                    fit_df,
                    epoch_outdir,
                    label,
                    use_chi2_scaled_error=args.chi2_scaled_error,
                )
                all_tables.append(fit_df)

        nonempty = [t for t in all_tables if t is not None and not t.empty]
        if nonempty:
            pd.concat(nonempty, ignore_index=True, sort=False).to_csv(
                epoch_outdir / "observable_summary_all.csv",
                index=False,
            )

        print(f"Finished Epoch {epoch}. Output sent to: {epoch_outdir}")

if __name__ == "__main__":
    main()