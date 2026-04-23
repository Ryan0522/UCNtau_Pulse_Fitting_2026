#include "ucn/io/AnalysisConfig.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace ucn::io {

namespace {

template <typename T>
T get_or(const json& j, const char* key, const T& fallback) {
    return j.contains(key) ? j.at(key).get<T>() : fallback;
}

void load_good_runs_if_present(AnalysisConfig& cfg) {
    if (cfg.good_runs_path.empty()) {
        return;
    }

    std::ifstream in(cfg.good_runs_path);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open good runs file: " + cfg.good_runs_path);
    }

    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) {
            cfg.good_runs.insert(line);
        }
    }
}

void load_runinfo_if_present(AnalysisConfig& cfg) {
    if (cfg.runinfo_path.empty()) {
        return;
    }

    std::ifstream in(cfg.runinfo_path);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open runinfo JSON: " + cfg.runinfo_path);
    }

    in >> cfg.runinfo_json;
}

} // namespace

AnalysisConfig load_analysis_config(const std::string& config_path) {
    std::ifstream in(config_path);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open config file: " + config_path);
    }

    json j;
    in >> j;

    AnalysisConfig cfg;

    cfg.year = get_or<int>(j, "year", cfg.year);
    cfg.start_run = get_or<int>(j, "start_run", cfg.start_run);
    cfg.end_run = get_or<int>(j, "end_run", cfg.end_run);
    cfg.restrict_to_good_runs = get_or<bool>(j, "restrict_to_good_runs", cfg.restrict_to_good_runs);

    cfg.data_folder = get_or<std::string>(j, "data_folder", cfg.data_folder);
    cfg.output_folder = get_or<std::string>(j, "output_folder", cfg.output_folder);
    cfg.runinfo_path = get_or<std::string>(j, "runinfo_path", cfg.runinfo_path);
    cfg.good_runs_path = get_or<std::string>(j, "good_runs_path", cfg.good_runs_path);

    cfg.root_filename_prefix = get_or<std::string>(j, "root_filename_prefix", cfg.root_filename_prefix);
    cfg.root_filename_suffix = get_or<std::string>(j, "root_filename_suffix", cfg.root_filename_suffix);
    cfg.tmcs0_tree_name = get_or<std::string>(j, "tmcs0_tree_name", cfg.tmcs0_tree_name);
    cfg.tmcs1_tree_name = get_or<std::string>(j, "tmcs1_tree_name", cfg.tmcs1_tree_name);
    cfg.tems_tree_name = get_or<std::string>(j, "tems_tree_name", cfg.tems_tree_name);

    cfg.use_runinfo_windows = get_or<bool>(j, "use_runinfo_windows", cfg.use_runinfo_windows);
    cfg.runinfo_times_in_seconds = get_or<bool>(j, "runinfo_times_in_seconds", cfg.runinfo_times_in_seconds);

    if (j.contains("default_window")) {
        const json& w = j.at("default_window");
        cfg.default_window.signal_start_us =
            get_or<double>(w, "signal_start_us", cfg.default_window.signal_start_us);
        cfg.default_window.signal_end_us =
            get_or<double>(w, "signal_end_us", cfg.default_window.signal_end_us);
        cfg.default_window.background_start_us =
            get_or<double>(w, "background_start_us", cfg.default_window.background_start_us);
    }

    if (j.contains("template_config")) {
        const json& t = j.at("template_config");
        cfg.template_config.native_bin_width_us =
            get_or<double>(t, "native_bin_width_us", cfg.template_config.native_bin_width_us);
        cfg.template_config.support_end_us =
            get_or<double>(t, "support_end_us", cfg.template_config.support_end_us);
        cfg.template_config.baseline = get_or<double>(t, "baseline", 0.0);
        cfg.template_config.gauss_amp = get_or<double>(t, "gauss_amp", 0.0);
        cfg.template_config.gauss_mu = get_or<double>(t, "gauss_mu", 0.0);
        cfg.template_config.gauss_sigma = get_or<double>(t, "gauss_sigma", 0.0);
        cfg.template_config.tail_start_us = get_or<double>(t, "tail_start_us", 0.0);
        cfg.template_config.a1 = get_or<double>(t, "a1", 0.0);
        cfg.template_config.tau1 = get_or<double>(t, "tau1", 0.0);
        cfg.template_config.a2 = get_or<double>(t, "a2", 0.0);
        cfg.template_config.tau2 = get_or<double>(t, "tau2", 0.0);
        cfg.template_config.a3 = get_or<double>(t, "a3", 0.0);
        cfg.template_config.tau3 = get_or<double>(t, "tau3", 0.0);
    }

    if (j.contains("region_settings")) {
        const json& r = j.at("region_settings");
        cfg.region_settings.coarse_bin_width_us =
            get_or<double>(r, "coarse_bin_width_us", cfg.region_settings.coarse_bin_width_us);
        cfg.region_settings.fine_bin_width_us =
            get_or<double>(r, "fine_bin_width_us", cfg.region_settings.fine_bin_width_us);
        cfg.region_settings.min_gap_us =
            get_or<double>(r, "min_gap_us", cfg.region_settings.min_gap_us);
        cfg.region_settings.coincidence_window_us =
            get_or<double>(r, "coincidence_window_us", cfg.region_settings.coincidence_window_us);
        cfg.region_settings.coincidence_min_hits =
            get_or<int>(r, "coincidence_min_hits", cfg.region_settings.coincidence_min_hits);
        cfg.region_settings.seed_veto_window_us =
            get_or<double>(r, "seed_veto_window_us", cfg.region_settings.seed_veto_window_us);
        cfg.region_settings.background_duration_us =
            get_or<double>(r, "background_duration_us", cfg.region_settings.background_duration_us);
        cfg.region_settings.background_iterations =
            get_or<int>(r, "background_iterations", cfg.region_settings.background_iterations);
        cfg.region_settings.background_tolerance_fraction =
            get_or<double>(r, "background_tolerance_fraction",
                           cfg.region_settings.background_tolerance_fraction);
        cfg.region_settings.enable_background_fit =
            get_or<bool>(r, "enable_background_fit", cfg.region_settings.enable_background_fit);
        cfg.region_settings.debug =
            get_or<bool>(r, "debug", cfg.region_settings.debug);
    }

    if (j.contains("fit_settings")) {
        const json& f = j.at("fit_settings");
        cfg.fit_settings.scan_step_us =
            get_or<double>(f, "scan_step_us", cfg.fit_settings.scan_step_us);
        cfg.fit_settings.max_offset_us =
            get_or<double>(f, "max_offset_us", cfg.fit_settings.max_offset_us);
        cfg.fit_settings.delta_nll_cut =
            get_or<double>(f, "delta_nll_cut", cfg.fit_settings.delta_nll_cut);
        cfg.fit_settings.min_spacing_us =
            get_or<double>(f, "min_spacing_us", cfg.fit_settings.min_spacing_us);
        cfg.fit_settings.min_amplitude_pe =
            get_or<double>(f, "min_amplitude_pe", cfg.fit_settings.min_amplitude_pe);
        cfg.fit_settings.max_amplitude_pe =
            get_or<double>(f, "max_amplitude_pe", cfg.fit_settings.max_amplitude_pe);
        cfg.fit_settings.background_per_bin =
            get_or<double>(f, "background_per_bin", cfg.fit_settings.background_per_bin);
        cfg.fit_settings.cluster_gap_us =
            get_or<double>(f, "cluster_gap_us", cfg.fit_settings.cluster_gap_us);
        cfg.fit_settings.max_coordinate_descent_steps =
            get_or<int>(f, "max_coordinate_descent_steps",
                        cfg.fit_settings.max_coordinate_descent_steps);
        cfg.fit_settings.enable_back_pruning =
            get_or<bool>(f, "enable_back_pruning", cfg.fit_settings.enable_back_pruning);
        cfg.fit_settings.debug =
            get_or<bool>(f, "debug", cfg.fit_settings.debug);
    }

    load_runinfo_if_present(cfg);
    load_good_runs_if_present(cfg);

    fs::create_directories(cfg.output_folder);

    return cfg;
}

} // namespace ucn::io