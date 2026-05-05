#include "ucn/io/AnalysisConfig.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

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

    cfg.array_output_subdir = get_or<std::string>(j, "array_output_subdir", cfg.array_output_subdir);
    cfg.shard_index = get_or<int>(j, "shard_index", cfg.shard_index);
    cfg.num_shards = get_or<int>(j, "num_shards", cfg.num_shards);

    cfg.enable_coincidence_output = get_or<bool>(j, "enable_coincidence_output", cfg.enable_coincidence_output);

    cfg.require_tems_tree = get_or<bool>(j, "require_tems_tree", cfg.require_tems_tree);
    cfg.segment_order = {"12", "34", "56", "78"};
    cfg.root_channel_maps = {
        {cfg.tmcs0_tree_name, {{1, "12"}, {2, "12"}, {3, "34"}, {4, "34"}}},
        {cfg.tmcs1_tree_name, {{11, "56"}, {12, "56"}, {13, "78"}, {14, "78"}}}
    };

    if (j.contains("root_channel_map")) {
        cfg.root_channel_maps.clear();
        cfg.segment_order.clear();

        const json& m = j.at("root_channel_map");
        for (auto tree_it = m.begin(); tree_it != m.end(); ++tree_it) {
            const std::string tree_name = tree_it.key();
            const json& segment_map = tree_it.value();

            for (auto seg_it = segment_map.begin(); seg_it != segment_map.end(); ++seg_it) {
                const std::string segment_name = seg_it.key();

                if (std::find(cfg.segment_order.begin(), cfg.segment_order.end(), segment_name) ==
                    cfg.segment_order.end()) {
                    cfg.segment_order.push_back(segment_name);
                }

                for (const auto& ch : seg_it.value()) {
                    cfg.root_channel_maps[tree_name][ch.get<int>()] = segment_name;
                }
            }
        }
    }

    if (j.contains("segment_order")) {
        cfg.segment_order.clear();
        for (const auto& seg : j.at("segment_order")) {
            cfg.segment_order.push_back(seg.get<std::string>());
        }
    }

    if (j.contains("coincidence_settings")) {
        const json& c = j.at("coincidence_settings");

        cfg.coincidence_settings.coincidence_window_us =
            get_or<double>(c, "coincidence_window_us",
                        cfg.coincidence_settings.coincidence_window_us);

        cfg.coincidence_settings.telescoping_window_us =
            get_or<double>(c, "telescoping_window_us",
                        cfg.coincidence_settings.telescoping_window_us);

        cfg.coincidence_settings.pe_threshold =
            get_or<int>(c, "pe_threshold",
                        cfg.coincidence_settings.pe_threshold);

        cfg.coincidence_settings.apply_pileup_correction =
            get_or<bool>(c, "apply_pileup_correction",
                        cfg.coincidence_settings.apply_pileup_correction);

        cfg.coincidence_settings.threshold_on_pileup_corrected =
            get_or<bool>(c, "threshold_on_pileup_corrected",
                        cfg.coincidence_settings.threshold_on_pileup_corrected);

        cfg.coincidence_settings.rng_seed =
            get_or<std::uint64_t>(c, "rng_seed",
                                cfg.coincidence_settings.rng_seed);
    }

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
        cfg.template_config.type = get_or<std::string>(
            t, "type", cfg.template_config.type
        );
        cfg.template_config.empirical_csv_path = get_or<std::string>(
            t, "empirical_csv_path", cfg.template_config.empirical_csv_path
        );
        cfg.template_config.empirical_pretrigger_us = get_or<double>(
            t, "empirical_pretrigger_us", cfg.template_config.empirical_pretrigger_us
        );
        cfg.template_config.native_bin_width_us =
            get_or<double>(t, "native_bin_width_us", cfg.template_config.native_bin_width_us);
        cfg.template_config.support_end_us =
            get_or<double>(t, "support_end_us", cfg.template_config.support_end_us);
        cfg.template_config.use_smooth_tail_onset = 
            get_or<bool>(t, "use_smooth_tail_onset", cfg.template_config.use_smooth_tail_onset);
        cfg.template_config.baseline = get_or<double>(t, "baseline", 0.0);
        cfg.template_config.gauss_amp = get_or<double>(t, "gauss_amp", 0.0);
        cfg.template_config.gauss_mu = get_or<double>(t, "gauss_mu", 0.0);
        cfg.template_config.gauss_sigma = get_or<double>(t, "gauss_sigma", 0.0);
        cfg.template_config.tail_start_us = get_or<double>(t, "tail_start_us", 0.0);
        cfg.template_config.tail_width_us = get_or<double>(t, "tail_width_us", cfg.template_config.tail_width_us);
        cfg.template_config.a1 = get_or<double>(t, "a1", 0.0);
        cfg.template_config.tau1 = get_or<double>(t, "tau1", 0.0);
        cfg.template_config.a2 = get_or<double>(t, "a2", 0.0);
        cfg.template_config.tau2 = get_or<double>(t, "tau2", 0.0);
        cfg.template_config.a3 = get_or<double>(t, "a3", 0.0);
        cfg.template_config.tau3 = get_or<double>(t, "tau3", 0.0);
        cfg.template_config.a4 = get_or<double>(t, "a4", 0.0);
        cfg.template_config.tau4 = get_or<double>(t, "tau4", 0.0);

        if (cfg.template_config.tail_width_us <= 0.0) {
            throw std::runtime_error("template_config.tail_width_us must be > 0");
        }
    }

    if (j.contains("region_settings")) {
        const json& r = j.at("region_settings");
        cfg.region_settings.coarse_bin_width_us =
            get_or<double>(r, "coarse_bin_width_us", cfg.region_settings.coarse_bin_width_us);
        cfg.region_settings.fine_bin_width_us =
            get_or<double>(r, "fine_bin_width_us", cfg.region_settings.fine_bin_width_us);
        cfg.region_settings.min_gap_us =
            get_or<double>(r, "min_gap_us", cfg.region_settings.min_gap_us);
        cfg.region_settings.fit_start_padding_us = 
            get_or<double>(r, "fit_start_padding_us", cfg.region_settings.fit_start_padding_us);
        cfg.region_settings.fit_end_padding_us =
            get_or<double>(r, "fit_end_padding_us", cfg.region_settings.fit_end_padding_us);
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

    if (j.contains("tail_extraction_settings")) {
        const json& t = j.at("tail_extraction_settings");
        cfg.tail_extraction.enable =
            get_or<bool>(t, "enable", cfg.tail_extraction.enable);
        cfg.tail_extraction.bin_width_us =
            get_or<double>(t, "bin_width_us", cfg.tail_extraction.bin_width_us);
        cfg.tail_extraction.window_us =
            get_or<double>(t, "window_us", cfg.tail_extraction.window_us);
        cfg.tail_extraction.pretrigger_us =
            get_or<double>(t, "pretrigger_us", cfg.tail_extraction.pretrigger_us);
        cfg.tail_extraction.min_amplitude_pe =
            get_or<double>(t, "min_amplitude_pe", cfg.tail_extraction.min_amplitude_pe);
        cfg.tail_extraction.max_amplitude_pe =
            get_or<double>(t, "max_amplitude_pe", cfg.tail_extraction.max_amplitude_pe);
        cfg.tail_extraction.only_non_pileup =
            get_or<bool>(t, "only_non_pileup", cfg.tail_extraction.only_non_pileup);
        cfg.tail_extraction.min_neighbor_separation_us =
            get_or<double>(t, "min_neighbor_separation_us",
                           cfg.tail_extraction.min_neighbor_separation_us);
        if (t.contains("regions")) {
            cfg.tail_extraction.regions.clear();
            for (const auto& r : t.at("regions")) {
                cfg.tail_extraction.regions.push_back(r.get<std::string>());
            }
        }
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
        cfg.fit_settings.allow_multiple_fits_per_cluster = 
            get_or<bool>(f, "allow_multiple_fits_per_cluster", cfg.fit_settings.allow_multiple_fits_per_cluster);
        cfg.fit_settings.enable_close_pulse_regularization =
            get_or<bool>(f, "enable_close_pulse_regularization",
                         cfg.fit_settings.enable_close_pulse_regularization);
        cfg.fit_settings.close_reg_lambda_nll =
            get_or<double>(f, "close_reg_lambda_nll", cfg.fit_settings.close_reg_lambda_nll);
        cfg.fit_settings.close_reg_window_us =
            get_or<double>(f, "close_reg_window_us", cfg.fit_settings.close_reg_window_us);
        cfg.fit_settings.close_reg_close_tau_us =
            get_or<double>(f, "close_reg_close_tau_us", cfg.fit_settings.close_reg_close_tau_us);
        cfg.fit_settings.close_reg_tail_tau_us =
            get_or<double>(f, "close_reg_tail_tau_us", cfg.fit_settings.close_reg_tail_tau_us);
        cfg.fit_settings.close_reg_eta =
            get_or<double>(f, "close_reg_eta", cfg.fit_settings.close_reg_eta);
        cfg.fit_settings.close_reg_floor_pe =
            get_or<double>(f, "close_reg_floor_pe", cfg.fit_settings.close_reg_floor_pe);

        cfg.fit_settings.enable_local_evidence =
            get_or<bool>(f, "enable_local_evidence", cfg.fit_settings.enable_local_evidence);
        cfg.fit_settings.local_evidence_pre_us =
            get_or<double>(f, "local_evidence_pre_us", cfg.fit_settings.local_evidence_pre_us);
        cfg.fit_settings.local_evidence_post_us =
            get_or<double>(f, "local_evidence_post_us", cfg.fit_settings.local_evidence_post_us);
        cfg.fit_settings.local_delta_nll_cut =
            get_or<double>(f, "local_delta_nll_cut", cfg.fit_settings.local_delta_nll_cut);
        cfg.fit_settings.debug =
            get_or<bool>(f, "debug", cfg.fit_settings.debug);
    }

    load_runinfo_if_present(cfg);
    load_good_runs_if_present(cfg);

    fs::create_directories(cfg.output_folder);

    return cfg;
}

} // namespace ucn::io