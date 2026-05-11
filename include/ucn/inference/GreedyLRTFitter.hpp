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

    FitResult discover_local_sequential(
        const Histogram& histogram,
        const std::vector<ClusterBound>& bounds,
        const FitSettings& settings,
        ucn::debug::DebugSink* debug_sink,
        const std::string& debug_case_id
    ) const;

    FitResult refit_amplitudes(
        const Histogram& histogram,
        const FitResult& input,
        const FitSettings& settings
    ) const;

    FitResult prune_by_deletion_lrt(
        const Histogram& histogram,
        const FitResult& input,
        const FitSettings& settings
    ) const;

    FitResult finalize_result(
        const Histogram& histogram,
        const FitResult& discovered,
        const FitSettings& settings
    ) const;

    const PulseTemplate& pulse_template_;
};

} // namespace ucn
