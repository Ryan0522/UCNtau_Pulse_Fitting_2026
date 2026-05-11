#pragma once
#include "ucn/data/FitTypes.hpp"
#include "ucn/templates/PulseTemplates.hpp"
#include "ucn/debug/DebugTypes.hpp"
#include <span>
#include <vector>
#include <string>

namespace ucn
{
    
class GreedyLRTFitter {
public:
    GreedyLRTFitter(const PulseTemplate& pulse_template);

    FitResult fit(
        const Histogram& histogram,
        std::span<const double> coincidence_times_us,
        const FitSettings& settings,
        ucn::debug::DebugSink* debug_sink = nullptr,
        const std::string& debug_case_id = ""
    ) const;

private:
    std::vector<std::vector<double>> cluster_seed_times(
        std::span<const double> coincidence_times_us,
        double cluster_gap_us
    ) const;

    std::vector<ClusterBound> build_cluster_bounds(
        const std::vector<std::vector<double>>& clusters,
        const Histogram& histogram,
        const FitSettings& settings
    ) const;

    FitResult remove_weak_pulses(
        const Histogram& histogram,
        const FitResult& current,
        const std::vector<std::vector<double>>& components,
        const FitSettings& settings
    ) const;

    FitResult fit_cluster_local_sequential(
        const Histogram& histogram,
        const std::vector<ClusterBound>& bounds,
        const FitSettings& settings,
        ucn::debug::DebugSink* debug_sink,
        const std::string& debug_case_id
    ) const;

    const PulseTemplate& pulse_template_;
};

} // namespace ucn
