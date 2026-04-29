#pragma once
#include "ucn/data/FitTypes.hpp"
#include "ucn/data/SignalTypes.hpp"
#include "ucn/inference/CoincidenceFitter.hpp"
#include <nlohmann/json.hpp>
#include <set>
#include <string>

namespace ucn::io {

using json = nlohmann::json;

struct PulseTemplateConfig {
    double native_bin_width_us = 0.5;
    double support_end_us = 100.0;

    // Triple Exponential / GaussianTrip Parameters
    double baseline = 0.0;
    double gauss_amp = 0.0;
    double gauss_mu = 0.0;
    double gauss_sigma = 0.0;
    double tail_start_us = 0.0;
    double a1 = 0.0;
    double tau1 = 0.0;
    double a2 = 0.0;
    double tau2 = 0.0;
    double a3 = 0.0;
    double tau3 = 0.0;
};

struct RunWindow {
    double start_us = 0.0;

    double background_start_us = 0.0;
    double background_end_us = 0.0;

    double signal_start_us = 0.0;
    double signal_end_us = 60.0e6;

    double end_start_us = 120.0e6;
    double end_end_us = 180.0e6;
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

    // Slurm-array / sharding
    std::string array_output_subdir = "";
    int shard_index = 0;
    int num_shards = 1;

    // Coincidence Settings
    bool enable_coincidence_output = true;
    CoincidenceSettings coincidence_settings;
};

AnalysisConfig load_analysis_config(const std::string& config_path);

} // namespace ucn::io