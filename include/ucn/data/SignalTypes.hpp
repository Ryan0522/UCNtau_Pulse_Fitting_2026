#pragma once
#include <vector>
#include <string>

namespace ucn
{
    
struct Hit {
    double time_us = 0.0; //  absolute hit time
    int channel = -1; // raw DAQ channel
};

struct WindowSummary {
    int window_index = 0;
    double start_time_us = 0.0;
    double end_time_us = 0.0;
    double width_us = 0.0;
    double bin_width_us = 0.0;

    int pulse_count = 0;
    int seed_count = 0;

    int observed_count = 0;
    double expected_count = 0.0;
    double final_nll = 0.0;

    double fitted_pe_sum = 0.0;
    double fit_expected_sum = 0.0;
    double fixed_expected_sum = 0.0;
    double background_expected_sum = 0.0;

    double template_mass_in_window = 0.0;
    double pe_per_observed_count = 0.0;
    double background_fraction = 0.0;
    double fit_fraction = 0.0;

    double local_background_rate_hz = 0.0;
    double local_background_gap_us = 0.0;
    int local_background_gap_hits = 0;
    bool local_background_updated = false;
};

struct TaggedPulse {
    double time_us = 0.0;
    double amplitude_pe = 0.0;
    
    int window_index = -1;
    double width_us = 0.0;

    bool is_pileup = false;
    bool uses_fine_bins = false;

    int seed_count_in_window = 0;
    int fit_pulse_count_in_window = 0;
    int pulse_rank_in_window = -1;
    double nearest_fit_dt_us = -1.0;
    std::string fit_topology = "unknown";
};

struct RegionSettings {
    double coarse_bin_width_us = 1.0;
    double fine_bin_width_us = 0.25;
    double min_gap_us = 10.0;

    double fit_start_padding_us = 20.0;
    double fit_end_padding_us = 100.0;

    std::string window_mode = "telescoping";
    double fixed_seed_pretrigger_us = 10.0;
    double fixed_seed_window_us = 500.0;

    double coincidence_window_us = 0.1;
    int coincidence_min_hits = 2;
    double seed_veto_window_us = 2.0;
    double background_duration_us = 60.0e6;
    int background_iterations = 10;
    double background_tolerance_fraction = 1.0e-2;
    bool enable_background_fit = true;

    bool enable_local_background = false;
    double local_bg_lookback_us = 5000.0; // 5 ms
    double local_bg_min_us = 1000.0; // 1 ms
    double local_bg_guard_us = 10.0;
    double local_bg_alpha = 0.5; // update speed
    double local_bg_max_scale = 10.0;

    bool recover_preseed_pile = true; // To prevent coincidence happening too late.
    double max_preseed_lookback_us = 50.0;
};

struct RegionResult {
    std::vector<TaggedPulse> signal_pulses;
    std::vector<TaggedPulse> background_pulses;
    std::vector<TaggedPulse> end_pulses;

    std::vector<WindowSummary> background_window_summaries;
    std::vector<WindowSummary> signal_window_summaries;
    std::vector<WindowSummary> end_window_summaries;
    
    double background_rate_hz = 0.0;
};

} // namespace ucn