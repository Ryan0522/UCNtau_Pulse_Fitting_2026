import pandas as pd
import numpy as np
from pathlib import Path

def load_pe_group_pmf(
    pmf_csv, segment=12, region="signal", pe_group="40-50", support_end_us=500.0,    
):
    df = pd.read_csv(pmf_csv)

    d = df[
        (df["segment"].astype(str) == str(segment)) &
        (df["region"].astype(str) == str(region)) & 
        (df["pe_group"].astype(str) == str(pe_group))
    ].copy()

    pcols = sorted([c for c in d.columns if c.startswith("p") and c[1:].isdigit()])
    pmf = d.iloc[0][pcols].to_numpy(float)
    pmf = np.clip(pmf, 0.0, None)

    pmf /= pmf.sum()
    cdf = np.cumsum(pmf)
    cdf[-1] = 1.0

    bin_width_us = support_end_us / len(pmf)
    return pmf, cdf, bin_width_us

def load_empirical_pe_sampler(amplitude_csv, pe_min=2, pe_max=None):
    df0 = pd.read_csv(amplitude_csv).copy()

    for c in ["count", "bin_left", "bin_right"]:
        df0[c] = pd.to_numeric(df0[c], errors="coerce")

    df = df0.dropna(subset=["count", "bin_left", "bin_right"]).copy()
    df = df[df["count"] > 0].copy()

    if pe_min is not None:
        df = df[df["bin_right"] > pe_min].copy()
    if pe_max is not None:
        df = df[df["bin_left"] < pe_max].copy()

    if df.empty:
        lo = df0["bin_left"].min()
        hi = df0["bin_right"].max()
        raise ValueError(
            f"No PE histogram bins survive pe_min={pe_min}, pe_max={pe_max}. "
            f"Available histogram range appears to be [{lo}, {hi}]. "
            "For fixed PE, use fixed_amplitude_pe=..., not a narrow pe_min/pe_max window."
        )

    weights = df["count"].to_numpy(float)
    weights_sum = weights.sum()
    if weights_sum <= 0 or not np.isfinite(weights_sum):
        raise ValueError("PE histogram weights are invalid after filtering.")

    weights /= weights_sum

    bin_left = df["bin_left"].to_numpy(float)
    bin_right = df["bin_right"].to_numpy(float)

    def sample_pe(rng, n):
        if n <= 0:
            return np.asarray([], dtype=int)

        idx = rng.choice(len(df), size=n, replace=True, p=weights)
        vals = rng.uniform(bin_left[idx], bin_right[idx])
        vals = np.rint(vals).astype(int)
        vals = np.maximum(vals, 2)
        return vals

    return sample_pe

def sample_arrivals_from_rate_curve_us(
    rng,
    rate_csv,
    segment=12,
    signal_start_us=10.0e6,
    duration_s=None,
    rate_scale=1.0,
    repeat_rate_curve=False,
):
    df = pd.read_csv(rate_csv).copy()
    df = df[df["segment"].astype(str) == str(segment)].copy()

    df["time_center_s"] = pd.to_numeric(df["time_center_s"], errors="coerce")
    df["bin_width_s"] = pd.to_numeric(df["bin_width_s"], errors="coerce")
    df["rate_hz"] = pd.to_numeric(df["rate_hz"], errors="coerce").fillna(0.0)

    df = df.dropna(subset=["time_center_s", "bin_width_s"]).copy()
    df = df.sort_values("time_center_s")

    left_edges = df["time_center_s"].to_numpy(float) - 0.5 * df["bin_width_s"].to_numpy(float)
    right_edges = df["time_center_s"].to_numpy(float) + 0.5 * df["bin_width_s"].to_numpy(float)

    rate_start_s = float(np.nanmin(left_edges))
    rate_end_s = float(np.nanmax(right_edges))
    rate_period_s = rate_end_s - rate_start_s

    if duration_s is None:
        duration_s = rate_period_s

    all_times = []

    if repeat_rate_curve:
        n_blocks = int(np.ceil(duration_s / rate_period_s))
    else:
        n_blocks = 1

    for block in range(n_blocks):
        block_offset_s = block * rate_period_s

        for _, row in df.iterrows():
            bw_s = float(row["bin_width_s"])
            center_s = float(row["time_center_s"])
            rate_hz = max(float(row["rate_hz"]) * rate_scale, 0.0)

            left_s = block_offset_s + (center_s - 0.5 * bw_s - rate_start_s)
            right_s = block_offset_s + (center_s + 0.5 * bw_s - rate_start_s)

            left_s = max(left_s, 0.0)
            right_s = min(right_s, duration_s)

            if right_s <= left_s:
                continue

            effective_bw_s = right_s - left_s
            lam = rate_hz * effective_bw_s
            n = rng.poisson(lam)

            if n > 0:
                t_rel_s = rng.uniform(left_s, right_s, size=n)
                all_times.extend(signal_start_us + 1.0e6 * t_rel_s)

    out = np.asarray(all_times, dtype=float)
    out = out[np.isfinite(out)]
    out.sort()

    return out, rate_period_s


def load_response_template(csv_path, column="Segment_12", support_end_us=100.0):
    df = pd.read_csv(csv_path)
    dist = np.asarray(df[column].values)

    dist = np.clip(dist, 0.0, None)
    dist = dist / np.sum(dist)
    bin_width_us = support_end_us / len(dist)

    cdf = np.cumsum(dist)
    cdf[-1] = 1.0

    return dist, cdf, bin_width_us

def sample_template_times_us(rng, cdf, n, bin_width_us, deformation_k=1.0):
    u = rng.random(n)
    u_warped = np.power(u, deformation_k)
    idx = np.searchsorted(cdf, u_warped, side="right")
    jitter = rng.uniform(0.0, bin_width_us, size=n)

    return idx * bin_width_us + jitter

def sample_poisson_arrivals_us(rng, rate_hz, start_us, end_us):
    if rate_hz <= 0.0 or not np.isfinite(rate_hz):
        return np.array([], dtype=float)

    mean_dt_us = 1e6 / rate_hz
    arrivals = []

    t = start_us
    while True:
        t += rng.exponential(mean_dt_us)
        if t >= end_us:
            break
        arrivals.append(t)

    return np.asarray(arrivals)

def sample_piecewise_rate_arrivals_us(
    rng, rate_schedule_hz,
    signal_start_us=10.0e6, block_duration_s=60.0,
):
    all_times = []
    rows = []

    for block_id, rate_hz in enumerate(rate_schedule_hz):
        rate_hz = float(rate_hz)
        block_start_s = block_id * block_duration_s
        block_end_s = (block_id + 1) * block_duration_s

        lam = max(rate_hz, 0.0) * block_duration_s
        n = rng.poisson(lam)

        if n > 0:
            t_rel_s = rng.uniform(block_start_s, block_end_s, size=n)
            all_times.extend(signal_start_us + 1.0e6 * t_rel_s)

        rows.append({
            "block_id": block_id,
            "block_start_s": block_start_s,
            "block_end_s": block_end_s,
            "rate_hz": rate_hz,
            "expected_events": lam,
            "generated_events": n,
        })

    times = np.asarray(all_times, dtype=float)
    times = times[np.isfinite(times)]
    times.sort()
    schedule_df = pd.DataFrame(rows)
    return times, schedule_df

def generate_pulse_hits_guaranteed_coincidence(
    rng,
    t0_us,
    amp,
    cdf,
    template_bin_width_us,
    coincidence_window_us=0.05,
    deformation_k=1.0,
):
    all_rel_times = sample_template_times_us(rng, cdf, amp, template_bin_width_us, deformation_k)
    anchor_rel = np.min(all_rel_times)
    max_pair_offset = 0.4 * coincidence_window_us
    delta = rng.uniform(-max_pair_offset, max_pair_offset)

    t_pair_1 = t0_us + anchor_rel
    t_pair_2 = t0_us + anchor_rel + delta
    times = [t_pair_1, t_pair_2]
    channels = [1, 2]

    remaining_rel = np.sort(all_rel_times)[2:]
    if len(remaining_rel) > 0:
        times.extend(t0_us + remaining_rel)
        channels.extend(rng.choice([1, 2], size=len(remaining_rel)))

    return np.asarray(times), np.asanyarray(channels)


# ============================================================
# Dataset generators
# ============================================================

def generate_empirical_mc_dataset(
    rate_csv,
    amplitude_csv,
    pmf_csv,
    suffix,
    segment=12,
    pe_group="40-50",
    region="signal",
    pre_trigger_s=10.0,
    support_end_us=500.0,
    duration_s=60.0,
    bkg_rate_hz_per_channel=1000.0,
    rate_scale=1.0,
    deformation_k=1.0,
    coincidence_window_us=0.05,
    timing_shift_us=0.0,
    repeat_rate_curve=False,
    rng=None,
    output_dir = "test/",
    template_reference_offset_us=20.0,
    amplitude_scale=1.0,
    amplitude_shift_pe=0.0,
    amplitude_max_pe=400,
):
    """
    General empirical MC generator.

    For 60 s:
        repeat_rate_curve=False

    For 1000 s:
        use generate_empirical_mc_dataset_1000s(...)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    pmf, cdf, template_bin_width_us = load_pe_group_pmf(
        pmf_csv=pmf_csv,
        segment=segment,
        region=region,
        pe_group=pe_group,
        support_end_us=support_end_us,
    )

    signal_start_us = pre_trigger_s * 1.0e6
    signal_end_us = signal_start_us + duration_s * 1.0e6
    total_record_end_us = signal_end_us + support_end_us

    true_times_us, rate_period_s = sample_arrivals_from_rate_curve_us(
        rng=rng,
        rate_csv=rate_csv,
        segment=segment,
        signal_start_us=signal_start_us,
        duration_s=duration_s,
        rate_scale=rate_scale,
        repeat_rate_curve=repeat_rate_curve,
    )

    n_pulses = len(true_times_us)

    sample_pe = load_empirical_pe_sampler(
        amplitude_csv=amplitude_csv,
        pe_min=2,
        pe_max=200,
    )
    raw_amplitudes = sample_pe(rng, n_pulses)
    true_amplitudes = np.rint(
        raw_amplitudes * amplitude_scale + amplitude_shift_pe
    ).astype(int)
    true_amplitudes = np.clip(true_amplitudes, 2, amplitude_max_pe)

    template_start_times_us = true_times_us - template_reference_offset_us

    truth_df = pd.DataFrame({
        "time_us": true_times_us,
        "template_start_us": template_start_times_us,
        "template_reference_offset_us": template_reference_offset_us,
        "time_rel_s": (true_times_us - signal_start_us) * 1.0e-6,
        "amplitude_pe": true_amplitudes,
        "raw_amplitude_pe": raw_amplitudes,
        "amplitude_scale": amplitude_scale,
        "amplitude_shift_pe": amplitude_shift_pe,
        "segment": segment,
        "pe_group_for_pmf": pe_group,
        "rate_scale": rate_scale,
        "timing_shift_us": timing_shift_us,
        "duration_s": duration_s,
        "repeat_rate_curve": repeat_rate_curve,
        "rate_period_s": rate_period_s,
    })

    output_dir = Path(output_dir)
    truth_path = output_dir / f"mc_truth_{suffix}.csv"
    hits_path = output_dir / f"mc_hits_{suffix}.csv"

    truth_df.to_csv(truth_path, index=False)

    all_times = []
    all_channels = []
    all_types = []

    for pulse_time_us, template_start_us, amp in zip(
        true_times_us,
        template_start_times_us,
        true_amplitudes,
    ):
        abs_t, ch = generate_pulse_hits_guaranteed_coincidence(
            rng=rng,
            t0_us=template_start_us,
            amp=int(amp),
            cdf=cdf,
            template_bin_width_us=template_bin_width_us,
            coincidence_window_us=coincidence_window_us,
            deformation_k=deformation_k,
        )

        abs_t = abs_t + timing_shift_us

        mask = (abs_t >= 0.0) & (abs_t < total_record_end_us)
        if np.any(mask):
            all_times.extend(abs_t[mask])
            all_channels.extend(ch[mask])
            all_types.extend(["signal"] * int(mask.sum()))

    # Background/random hits.
    for ch_id in [1, 2]:
        bkg_times = sample_poisson_arrivals_us(
            rng=rng,
            rate_hz=bkg_rate_hz_per_channel,
            start_us=0.0,
            end_us=total_record_end_us,
        )
        all_times.extend(bkg_times)
        all_channels.extend([ch_id] * len(bkg_times))
        all_types.extend(["noise"] * len(bkg_times))

    hits_df = pd.DataFrame({
        "time_us": np.asarray(all_times, dtype=float),
        "channel": np.asarray(all_channels, dtype=int),
        "type": np.asarray(all_types),
    }).sort_values("time_us", kind="mergesort")

    hits_df[["time_us", "channel"]].to_csv(hits_path, index=False)

    print(f"[write] {truth_path}")
    print(f"[write] {hits_path}")
    print(f"[info] suffix={suffix}")
    print(f"[info] n_truth={len(truth_df)}, n_hits={len(hits_df)}")
    print(f"[info] duration_s={duration_s}, repeat_rate_curve={repeat_rate_curve}, rate_period_s={rate_period_s:.6g}")
    print(f"[info] rate_scale={rate_scale}, PE sampled from {amplitude_csv}")
    print(f"[info] PMF: segment={segment}, region={region}, pe_group={pe_group}")
    print(f"[info] bkg_rate_hz_per_channel={bkg_rate_hz_per_channel}")
    return hits_df, truth_df


def generate_frequency_sweep_dataset(
    amplitude_csv, pmf_csv, suffix, segment=12, pe_group="40-50", region="signal",
    seed=1, output_dir="test/", pre_trigger_s=10.0, support_end_us=500.0,
    block_duration_s=60.0, rate_schedule_hz=(100, 200, 400, 800, 1600, 800, 400, 200, 100),
    bkg_rate_hz_per_channel=0.0, pe_min=5, pe_max=200,
    amplitude_scale=1.0, amplitude_shift_pe=0.0, amplitude_max_pe=400,
    fixed_amplitude_pe=None,
    deformation_k=1.0,
    coincidence_window_us=0.05, timing_shift_us=0.0, template_reference_offset_us=20.0,
):
    rng = np.random.default_rng(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rate_schedule_hz = np.asarray(rate_schedule_hz, dtype=float)
    n_blocks = len(rate_schedule_hz)
    duration_s = n_blocks * block_duration_s

    signal_start_us = pre_trigger_s * 1.0e6
    signal_end_us = signal_start_us + duration_s * 1.0e6
    total_record_end_us = signal_end_us + support_end_us

    pmf, cdf, template_bin_width_us = load_pe_group_pmf(
        pmf_csv=pmf_csv,
        segment=segment,
        region=region,
        pe_group=pe_group,
        support_end_us=support_end_us,
    )
    
    true_times_us, schedule_df = sample_piecewise_rate_arrivals_us(
        rng=rng,
        rate_schedule_hz=rate_schedule_hz,
        signal_start_us=signal_start_us,
        block_duration_s=block_duration_s,
    )

    n_pulses = len(true_times_us)

    if fixed_amplitude_pe is not None:
        raw_amplitudes = np.full(n_pulses, int(fixed_amplitude_pe), dtype=int)
        true_amplitudes = raw_amplitudes.copy()
    else:
        sample_pe = load_empirical_pe_sampler(
            amplitude_csv=amplitude_csv,
            pe_min=pe_min,
            pe_max=pe_max,
        )

        raw_amplitudes = sample_pe(rng, n_pulses)

        true_amplitudes = np.rint(
            raw_amplitudes * amplitude_scale + amplitude_shift_pe
        ).astype(int)

        true_amplitudes = np.clip(true_amplitudes, 2, amplitude_max_pe)

    block_id = np.floor(
        ((true_times_us - signal_start_us) * 1.0e-6) / block_duration_s
    ).astype(int)
    block_id = np.clip(block_id, 0, n_blocks - 1)

    template_start_times_us = true_times_us - template_reference_offset_us

    truth_df = pd.DataFrame({
        "time_us": true_times_us,
        "template_start_us": template_start_times_us,
        "template_reference_offset_us": template_reference_offset_us,
        "time_rel_s": (true_times_us - signal_start_us) * 1.0e-6,

        "block_id": block_id,
        "block_start_s": block_id * block_duration_s,
        "block_end_s": (block_id + 1) * block_duration_s,
        "rate_hz": rate_schedule_hz[block_id],

        "amplitude_pe": true_amplitudes,
        "raw_amplitude_pe": raw_amplitudes,
        "amplitude_scale": amplitude_scale,
        "amplitude_shift_pe": amplitude_shift_pe,
        "amplitude_max_pe": amplitude_max_pe,
        "fixed_amplitude_pe": fixed_amplitude_pe if fixed_amplitude_pe is not None else np.nan,
        "pe_min": pe_min,
        "pe_max": pe_max,

        "segment": segment,
        "pe_group_for_pmf": pe_group,
        "duration_s": duration_s,
        "bkg_rate_hz_per_channel": bkg_rate_hz_per_channel,
    })

    all_times = []
    all_channels = []
    all_types = []

    for template_start_us, amp in zip(template_start_times_us, true_amplitudes):
        abs_t, ch = generate_pulse_hits_guaranteed_coincidence(
            rng=rng,
            t0_us=template_start_us,
            amp=int(amp),
            cdf=cdf,
            template_bin_width_us=template_bin_width_us,
            coincidence_window_us=coincidence_window_us,
            deformation_k=deformation_k,
        )

        abs_t = abs_t + timing_shift_us

        mask = (abs_t >= 0.0) & (abs_t < total_record_end_us)
        if np.any(mask):
            all_times.extend(abs_t[mask])
            all_channels.extend(ch[mask])
            all_types.extend(["signal"] * int(mask.sum()))

    # Constant background only. Set to 0 for the cleanest test.
    if bkg_rate_hz_per_channel > 0:
        for ch_id in [1, 2]:
            bkg_times = sample_poisson_arrivals_us(
                rng=rng,
                rate_hz=bkg_rate_hz_per_channel,
                start_us=0.0,
                end_us=total_record_end_us,
            )
            all_times.extend(bkg_times)
            all_channels.extend([ch_id] * len(bkg_times))
            all_types.extend(["noise"] * len(bkg_times))

    hits_df = pd.DataFrame({
        "time_us": np.asarray(all_times, dtype=float),
        "channel": np.asarray(all_channels, dtype=int),
        "type": np.asarray(all_types),
    }).sort_values("time_us", kind="mergesort")

    truth_path = output_dir / f"mc_truth_{suffix}.csv"
    hits_path = output_dir / f"mc_hits_{suffix}.csv"
    hits_labeled_path = output_dir / f"mc_hits_labeled_{suffix}.csv"
    schedule_path = output_dir / f"mc_schedule_{suffix}.csv"

    truth_df.to_csv(truth_path, index=False)
    hits_df[["time_us", "channel"]].to_csv(hits_path, index=False)
    hits_df.to_csv(hits_labeled_path, index=False)
    schedule_df.to_csv(schedule_path, index=False)

    print(f"[write] {truth_path}")
    print(f"[write] {hits_path}")
    print(f"[write] {hits_labeled_path}")
    print(f"[write] {schedule_path}")
    print(f"[info] suffix={suffix}")
    print(f"[info] duration_s={duration_s}")
    print(f"[info] n_truth={len(truth_df)}, n_hits={len(hits_df)}")
    print(f"[info] rate_schedule_hz={rate_schedule_hz}")
    print(f"[info] PE sampled from {amplitude_csv}")
    print(f"[info] bkg_rate_hz_per_channel={bkg_rate_hz_per_channel}")

    return hits_df, truth_df, schedule_df


# ============================================================
# Optional quick plotting helper: truth-only dt
# ============================================================

def compute_inter_neutron_dt_s(df, time_col="time_us"):
    t = np.sort(pd.to_numeric(df[time_col], errors="coerce").dropna().to_numpy(float))
    dt_s = np.diff(t) * 1.0e-6
    dt_s = dt_s[np.isfinite(dt_s) & (dt_s > 0)]
    return dt_s


def save_truth_dt_summary(truth_df, suffix, output_dir="test/"):
    """
    Small text/CSV summary so you can quickly check whether the 1 s region
    has enough statistics.
    """
    dt_s = compute_inter_neutron_dt_s(truth_df)

    ranges = [
        (1e-6, 1e-5),
        (1e-5, 1e-4),
        (1e-4, 1e-3),
        (1e-3, 1e-2),
        (1e-2, 1e-1),
        (1e-1, 3e-1),
        (3e-1, 1.0),
        (1.0, 3.0),
        (3.0, 10.0),
    ]

    rows = []
    for lo, hi in ranges:
        n = int(np.sum((dt_s >= lo) & (dt_s < hi)))
        rows.append({
            "dt_lo_s": lo,
            "dt_hi_s": hi,
            "count": n,
            "frac_err_poisson": np.inf if n == 0 else 1.0 / np.sqrt(n),
        })

    out = pd.DataFrame(rows)
    output_dir = Path(output_dir)
    path = output_dir / f"mc_truth_dt_summary_{suffix}.csv"
    out.to_csv(path, index=False)
    print(f"[write] {path}")
    print(out.to_string(index=False))
    return out


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    suffix = "closure_lowrate_nobkg"

    hits_df, truth_df = generate_empirical_mc_dataset(
        rate_csv="./test/input/rate.csv",
        amplitude_csv="./test/input/amplitude_pe.csv",
        pmf_csv="./test/input/Fine_PE_Group_PMF.csv",
        suffix=suffix,
        segment=12,
        pe_group="40-50",
        region="signal",
        pre_trigger_s=10.0,
        support_end_us=500.0,
        duration_s=60.0,
        bkg_rate_hz_per_channel=0.0,
        rate_scale=1.0,
        deformation_k=1.0,
        coincidence_window_us=0.05,
        timing_shift_us=0.0,
        repeat_rate_curve=False,
        rng=np.random.default_rng(1),
        output_dir=Path("test/"),
        template_reference_offset_us=20.0,
        amplitude_scale=1.15,
        amplitude_shift_pe=5.0,
        amplitude_max_pe=400.0,
    )

    save_truth_dt_summary(truth_df, suffix, output_dir=Path("test/"))

    
    suffix = "freq_sweep_fixedPE40_nobkg"
    hits_df, truth_df, schedule_df = generate_frequency_sweep_dataset(
        amplitude_csv="./test/input/amplitude_pe.csv",
        pmf_csv="./test/input/Fine_PE_Group_PMF.csv",
        suffix=suffix,
        segment=12,
        pe_group="40-50",
        region="signal",
        seed=1,
        output_dir=Path("test/"),

        block_duration_s=5.0,
        rate_schedule_hz=[100, 200, 400, 800, 1600, 800, 400, 200, 100],

        # no background first
        bkg_rate_hz_per_channel=0.0,

        # fixed PE mode
        fixed_amplitude_pe=40,

        # ignored in fixed mode, but keep harmless values
        pe_min=5,
        pe_max=400,
        amplitude_scale=1.0,
        amplitude_shift_pe=0.0,
        amplitude_max_pe=400,

        deformation_k=1.0,
        coincidence_window_us=0.05,
        timing_shift_us=0.0,
        template_reference_offset_us=20.0,
    )
    save_truth_dt_summary(truth_df, suffix, output_dir=Path("test/"))

    suffix = "freq_sweep_empPE_nobkg"
    hits_df, truth_df, schedule_df = generate_frequency_sweep_dataset(
        amplitude_csv="./test/input/amplitude_pe.csv",
        pmf_csv="./test/input/Fine_PE_Group_PMF.csv",
        suffix=suffix,
        segment=12,
        pe_group="40-50",
        region="signal",
        seed=1,
        output_dir=Path("test/"),

        block_duration_s=5.0,
        rate_schedule_hz=[100, 200, 400, 800, 1600, 800, 400, 200, 100],

        # no background first
        bkg_rate_hz_per_channel=0.0,

        # same PE histogram distribution for all blocks
        fixed_amplitude_pe=None,
        pe_min=5,
        pe_max=400,
        amplitude_scale=1.0,
        amplitude_shift_pe=0.0,
        amplitude_max_pe=400,

        deformation_k=1.0,
        coincidence_window_us=0.05,
        timing_shift_us=0.0,
        template_reference_offset_us=20.0,
    )
    save_truth_dt_summary(truth_df, suffix, output_dir=Path("test/"))