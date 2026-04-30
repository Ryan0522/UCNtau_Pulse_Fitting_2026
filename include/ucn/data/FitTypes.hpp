#pragma once
#include "ucn/data/SignalTypes.hpp"
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
    double scan_step_us = 0.5;
    double max_offset_us = 5.0;
    double delta_nll_cut = 10.0;
    double min_spacing_us = 2.0;
    double min_amplitude_pe = 5.0;
    double max_amplitude_pe = 300.0;
    double background_per_bin = 0.0;
    double cluster_gap_us = 1.0;
    int max_coordinate_descent_steps = 20;
    bool enable_back_pruning = true;

    // Method 1: close-pulse residual regularization.
    // Penalty = lambda * exp(-dt / close_tau) * exp(-A_new / residual_scale),
    // residual_scale = eta * A_prev * exp(-dt / tail_tau) + floor_pe.
    bool enable_close_pulse_regularization = false;
    double close_reg_lambda_nll = 0.0;
    double close_reg_window_us = 100.0;
    double close_reg_close_tau_us = 10.0;
    double close_reg_tail_tau_us = 30.0;
    double close_reg_eta = 0.10;
    double close_reg_floor_pe = 1.0;

    // Method 2: local evidence requirement.
    // The local NLL improvement around the candidate time must exceed local_delta_nll_cut.
    bool enable_local_evidence = false;
    double local_evidence_pre_us = 0.5;
    double local_evidence_post_us = 2.0;
    double local_delta_nll_cut = 3.0;

    bool debug = false;
    std::vector<double> fixed_expected;
};

struct FitResult {
    std::vector<PulseCandidate> pulses;
    std::vector<double> expected_total;
    double final_nll = 0.0;
};

} // namespace ucn
