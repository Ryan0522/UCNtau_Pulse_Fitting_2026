#pragma once
#include "ucn/data/FitTypes.hpp"
#include "ucn/data/SignalTypes.hpp"
#include <nlohmann/json.hpp>
#include <set>
#include <string>

namespace ucn::io {

using json = nlohmann::json;

struct PulseTemplateConfig {
    double native_bin_width_us = 0.25;
    double pulse_start_offset_us = 5.0;
    double rise_us = 0.8;
    double fast_decay_us = 8.0;
    double slow_decay_us = 60.0;
};

struct RunWindow {
    double signal_start_us = 0.0;
    double signal_end_us = 60.0e6;
    double background_start_us = 70.0e6;
};

struct AnalysisConfig {
    // Run selection
    int year = 2022;
    int start_run = 0;
    int end_run = 0;
    bool restrict_to_good_runs = false;

    // Input / output
    std::string data_folder = "./data/";
    std::string output_folder = "./output/";
    std::string runinfo_path = "";
    std::string good_runs_path = "";

    // ROOT naming
    std::string root_filename_prefix = "processed_output_";
    std::string root_filename_suffix = ".root";
    std::string tmcs0_tree_name = "tmcs_0";
    std::string tmcs1_tree_name = "tmcs_1";
    std::string tems_tree_name  = "tems";

    // Window selection
    bool use_runinfo_windows = false;
    bool runinfo_times_in_seconds = false;
    RunWindow default_window;

    // Template + fitter settings
    PulseTemplateConfig template_config;
    RegionSettings region_settings;
    FitSettings fit_settings;

    // Optional loaded metadata
    json runinfo_json = json::object();
    std::set<std::string> good_runs;
};

AnalysisConfig load_analysis_config(const std::string& config_path);

} // namespace ucn::io