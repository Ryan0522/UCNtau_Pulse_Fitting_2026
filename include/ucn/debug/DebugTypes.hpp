#pragma once
#include "ucn/data/FitTypes.hpp"
#include <string>
#include <vector>

namespace ucn::debug {

enum class DebugCaseType {
    SingleNeutronSingleSeed,
    SingleNeutronMultiSeed,
    MultiNeutron,

    ObservedOneSeedOneFit,
    ObservedOneSeedMultiFit,
    ObservedMultiSeed,

    Unknown
};

inline const char* case_name(DebugCaseType t) {
    switch (t) {
        case DebugCaseType::SingleNeutronSingleSeed:
            return "single_neutron_single_seed";
        case DebugCaseType::SingleNeutronMultiSeed:
            return "single_neutron_multi_seed";
        case DebugCaseType::MultiNeutron:
            return "multi_neutron";
        case DebugCaseType::ObservedOneSeedOneFit:
            return "observed_one_seed_one_fit";
        case DebugCaseType::ObservedOneSeedMultiFit:
            return "observed_one_seed_multi_fit";
        case DebugCaseType::ObservedMultiSeed:
            return "observed_multi_seed";
        default:
            return "unknown";
    }
}

struct TruthPulse {
    double time_us = 0.0;
    double amplitude_pe = 0.0;
};

struct WindowDebugContext {
    std::string case_id;
    DebugCaseType case_type = DebugCaseType::Unknown;

    int chunk_index = -1;
    int local_window_index = -1;
    int global_window_index = -1;

    double start_time_us = 0.0;
    double end_time_us = 0.0;
    double model_end_time_us = 0.0;
    double bin_width_us = 0.0;

    std::vector<double> seeds;
    std::vector<TruthPulse> truth_pulses;
};

struct LRTTrialDebug {
    std::string case_id;

    int fit_iter = -1;
    int cluster_index = -1;
    double trial_time_us = 0.0;
    double trial_amp = 0.0;

    double nll_before = 0.0;
    double nll_after = 0.0;
    double delta_nll = 0.0;

    double penalty_nll = 0.0;
    double required_delta_nll = 0.0;

    double local_delta_nll = 0.0;
    int local_pass = 1;

    double margin = 0.0;
    int accepted = 0;
};

class DebugSink {
public:
    virtual ~DebugSink() = default;

    virtual void on_lrt_trial(const LRTTrialDebug&) {}
    virtual void on_lrt_accept(const LRTTrialDebug&) {}
};

class BufferedDebugSink : public DebugSink {
public:
    void on_lrt_trial(const LRTTrialDebug& row) override {
        trials.push_back(row);
    }

    void on_lrt_accept(const LRTTrialDebug& row) override {
        accepts.push_back(row);
    }

    std::vector<LRTTrialDebug> trials;
    std::vector<LRTTrialDebug> accepts;
};

} // namespace ucn::debug