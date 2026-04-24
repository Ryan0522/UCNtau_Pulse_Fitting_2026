#pragma once
#include <vector>

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
    double bin_width_us = 0.0;
    int pulse_count = 0;
    int observed_count = 0;
    double expected_count = 0.0;
    double final_nll = 0.0;
    int seed_count = 0;
};

struct TaggedPulse {
    double time_us = 0.0;
    double amplitude_pe = 0.0;
    
    int window_index = -1;
    double window_width_us = 0.0;

    bool is_pileup = false;
    bool uses_fine_bins = false;
};

struct RegionSettings {
    double coarse_bin_width_us = 1.0;
    double fine_bin_width_us = 0.25;
    double min_gap_us = 10.0;
    double fit_end_padding_us = 100.0;
    double coincidence_window_us = 0.1;
    int coincidence_min_hits = 2;
    double seed_veto_window_us = 2.0;
    double background_duration_us = 60.0e6;
    int background_iterations = 10;
    double background_tolerance_fraction = 1.0e-2;
    bool enable_background_fit = true;
    bool debug = false;
};

struct RegionResult {
    std::vector<TaggedPulse> signal_pulses;
    std::vector<TaggedPulse> background_pulses;
    std::vector<WindowSummary> window_summaries;
    double background_rate_hz = 0.0;
};

} // namespace ucn