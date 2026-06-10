#pragma once
#include "ucn/data/SignalTypes.hpp"
#include <map>
#include <vector>

namespace ucn
{
    
struct PulseCandidate {
    double time_us = 0.0;
    double amplitude_pe = 0.0;
};

struct Histogram {
    std::vector<double> bin_edges_us;
    std::vector<double> counts;
};

struct ClusterBound {
    double cluster_time_us = 0.0;
    double left_us = 0.0;
    double right_us = 0.0;
};

struct FitSettings {
    // Scan / discovery
    double scan_step_us = 0.5;
    double max_offset_us = 5.0;
    double cluster_gap_us = 1.0;
    double min_spacing_us = 2.0;

    // LRT acceptance
    double discovery_delta_nll_cut = 10.0;

    // Amplitude bounds
    double min_amplitude_pe = 5.0;
    double max_amplitude_pe = 400.0;

    // Post-fit PE gain correction.
    double pe_gain = 1.0;
    double pe_gain_default = 1.0;
    std::map<int, double> pe_gain_by_hold_s;

    // Sequential local discovery
    double local_template_mass_floor = 1.0e-3;

    // Final simultaneous refit at fixed discovered times.
    int max_refit_steps = 50;
    double refit_tolerance = 1.0e-8;

    // Pruning after refit
    double prune_delta_nll_cut = 3.0;
    int max_prune_passes = 3;

    // External expected counts
    double background_per_bin = 0.0;
    std::vector<double> fixed_expected;
};

struct FitResult {
    std::vector<PulseCandidate> pulses;
    std::vector<double> expected_total;
    double final_nll = 0.0;
};

} // namespace ucn
