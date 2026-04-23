#pragma once

#include "ucn/data/FitTypes.hpp"
#include "ucn/data/SignalTypes.hpp"
#include "ucn/inference/GreedyLRTFitter.hpp"
#include "ucn/templates/PulseTemplates.hpp"

#include <vector>

namespace ucn {

class WindowedPulseProcessor {
public:
    WindowedPulseProcessor(const PulseTemplate& pulse_template,
                           const GreedyLRTFitter& fitter);

    void set_debug_max_windows(int n);

    RegionResult analyze(const std::vector<Hit>& hits,
                         double signal_start_us,
                         double signal_end_us,
                         double background_start_us,
                         const RegionSettings& region_settings,
                         const FitSettings& fit_settings) const;

private:
    std::vector<Hit> select_hits(const std::vector<Hit>& hits,
                                 double start_us,
                                 double end_us) const;

    bool build_window_histogram(const std::vector<Hit>& hits,
                                int start_index,
                                double bin_width_us,
                                const RegionSettings& region_settings,
                                int& next_index,
                                double& start_time_us,
                                double& end_time_us,
                                Histogram& histogram) const;

    std::vector<double> find_coincidence_seeds(const std::vector<Hit>& hits,
                                               double window_start_us,
                                               double window_end_us,
                                               double bin_width_us,
                                               const RegionSettings& region_settings) const;

    std::vector<double> build_carry_expected(const std::vector<PulseCandidate>& carry_pulses,
                                             const Histogram& histogram) const;

    void fit_stream(const std::vector<Hit>& hits,
                    const RegionSettings& region_settings,
                    const FitSettings& fit_settings,
                    double background_rate_hz,
                    std::vector<TaggedPulse>& output_pulses,
                    std::vector<WindowSummary>& output_summaries) const;

    const PulseTemplate& pulse_template_;
    const GreedyLRTFitter& fitter_;
    int debug_max_windows_ = -1;
};

} // namespace ucn
