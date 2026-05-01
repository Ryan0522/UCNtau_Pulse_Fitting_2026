#pragma once
#include "ucn/data/FitTypes.hpp"
#include "ucn/data/SignalTypes.hpp"
#include "ucn/inference/CoincidenceFitter.hpp"
#include <nlohmann/json.hpp>
#include <map>
#include <set>
#include <string>
#include <vector>

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

struct TailExtractionSettings {
    bool enable = false;
    double bin_width_us = 0.5;
    double window_us = 500.0;
    double pretrigger_us = 0.0;
    double min_amplitude_pe = 0.0;
    double max_amplitude_pe = 1.0e5;
    bool only_non_pileup = true;
    double min_neighbor_separation_us = 0.0;
    std::vector<std::string> regions = {"signal"};
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

    // Template + fitter + tail settings
    PulseTemplateConfig template_config;
    RegionSettings region_settings;
    FitSettings fit_settings;
    TailExtractionSettings tail_extraction;

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

    // 2022 / 2021 Dagger ROOT file setup
    bool require_tems_tree = true;
    // Configurable ROOT tree/channel -> analysis segment mapping.
    // Example: root_channel_maps["tmc_0"][1] = "12"
    std::map<std::string, std::map<int, std::string>> root_channel_maps;
    std::vector<std::string> segment_order;
};

AnalysisConfig load_analysis_config(const std::string& config_path);

} // namespace ucn::io