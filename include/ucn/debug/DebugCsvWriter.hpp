#pragma once
#include "ucn/debug/DebugTypes.hpp"
#include "ucn/templates/PulseTemplates.hpp"
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <tuple>

namespace ucn::debug {

class DebugCsvWriter : public DebugSink {
public:
    DebugCsvWriter(const std::filesystem::path& outdir,
                   int max_per_case,
                   const PulseTemplate& pulse_template);

    bool enabled() const { return enabled_; }

    bool can_capture(DebugCaseType type, double hold_time_s) const;
    bool can_capture_any_observed(double hold_time_s) const;

    std::string next_case_id(DebugCaseType type, double hold_time_s);

    void write_window(
        const WindowDebugContext& ctx,
        const Histogram& histogram,
        const FitResult& fit,
        const std::vector<double>& fixed_expected,
        double background_per_bin
    );

    void on_lrt_trial(const LRTTrialDebug& row) override;
    void on_lrt_accept(const LRTTrialDebug& row) override;

    void write_lrt_trials_for_case(
        const std::string& case_id,
        const std::vector<LRTTrialDebug>& rows
    );

    void write_lrt_accepts_for_case(
        const std::string& case_id,
        const std::vector<LRTTrialDebug>& rows
    );

private:
    bool enabled_ = false;
    int max_per_case_ = 0;

    const PulseTemplate& pulse_template_;

    std::map<std::tuple<DebugCaseType, int>, int> counts_;

    std::ofstream manifest_;
    std::ofstream bins_;
    std::ofstream markers_;
    std::ofstream components_;
    std::ofstream lrt_trials_;
    std::ofstream lrt_accepts_;

    void write_headers();
};

} // namespace ucn::debug