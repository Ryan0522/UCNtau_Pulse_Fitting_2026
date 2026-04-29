#pragma once
#include "ucn/data/FitTypes.hpp"
#include <cstdint>
#include <vector>

namespace ucn {

struct CoincidenceSettings {
    double coincidence_window_us = 0.1;
    double telescoping_window_us = 1.0;
    int pe_threshold = 8;
    bool apply_pileup_correction = true;
    bool threshold_on_pileup_corrected = false;

    std::uint64_t rng_seed = 0x9e3779b97f4a7c15ULL;
};

struct CoincidenceEvent {
    double start_time_us = 0.0;
    double end_time_us = 0.0;
    double length_us = 0.0;

    int start_channel = -1;

    int n_pe = 0;
    double n_pileup = 0.0;
    double free_pe_interval_us = 0.0;

    bool passes_raw_threshold = false;
    bool passes_pileup_threshold = false;
};

struct CoincidenceFitResult {
    std::vector<CoincidenceEvent> events;

    double average_length_us = 0.0;
    double average_n_pe = 0.0;
};

class CoincidenceFitter {
public:
    explicit CoincidenceFitter(CoincidenceSettings settings);

    CoincidenceFitResult find(const std::vector<Hit>& hits) const;

private:
    CoincidenceSettings settings_;
};

} // namespace ucn