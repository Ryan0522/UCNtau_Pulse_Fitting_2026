#pragma once

#include "ucn/data/FitTypes.hpp"
#include "ucn/data/SignalTypes.hpp"
#include "ucn/inference/GreedyLRTFitter.hpp"
#include "ucn/templates/PulseTemplates.hpp"
#include "ucn/debug/DebugCsvWriter.hpp"

#include <vector>
#include <memory>
#include <string>

namespace ucn {

class WindowedPulseProcessor {
public:
    WindowedPulseProcessor(const PulseTemplate& pulse_template,
                           const GreedyLRTFitter& fitter);

    RegionResult analyze(const std::vector<Hit>& hits,
                         double background_start_us,
                         double background_end_us,
                         double signal_start_us,
                         double signal_end_us,
                         double end_start_us,
                         double end_end_us,
                         const RegionSettings& region_settings,
                         const FitSettings& fit_settings,
                         const std::vector<debug::TruthPulse>* truth_pulses = nullptr,
                         int chunk_index = -1,
                         int global_window_offset = 0,
                         double hold_time_s = -1.0) const;

    void set_debug_writer(std::shared_ptr<debug::DebugCsvWriter> writer);

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
                                double& model_end_time_us,
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
                    std::vector<WindowSummary>& output_summaries,
                    const std::vector<debug::TruthPulse>* truth_pulses,
                    int chunk_index,
                    int global_window_offset,
                    const std::string& region_name,
                    double hold_time_s) const;

    std::shared_ptr<debug::DebugCsvWriter> debug_writer_;

    const PulseTemplate& pulse_template_;
    const GreedyLRTFitter& fitter_;
};

} // namespace ucn
