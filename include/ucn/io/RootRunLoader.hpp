#pragma once
#include "ucn/data/SignalTypes.hpp"
#include "ucn/io/AnalysisConfig.hpp"
#include <string>
#include <vector>

namespace ucn::io {

struct LoadedSegment {
    std::string segment_name;
    std::vector<Hit> hits;
};

struct LoadedRun {
    std::string run_id;
    std::vector<LoadedSegment> segments;
};

RunWindow resolve_run_window(const AnalysisConfig& cfg, int run_number);
LoadedRun load_root_run(const AnalysisConfig& cfg, int run_number);

} // namespace ucn::io