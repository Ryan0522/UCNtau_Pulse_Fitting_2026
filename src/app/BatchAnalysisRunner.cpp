#include "ucn/app/BatchAnalysisRunner.hpp"
#include "ucn/app/WindowedPulseProcessor.hpp"
#include "ucn/inference/GreedyLRTFitter.hpp"
#include "ucn/io/RootRunLoader.hpp"
#include "ucn/templates/GaussianTripPulseTemplate.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace ucn::app {

namespace {

bool run_is_allowed(const io::AnalysisConfig& cfg, int run_number) {
    if (!cfg.restrict_to_good_runs || cfg.good_runs.empty()) {
        return true;
    }
    return cfg.good_runs.count(std::to_string(run_number)) > 0;
}

bool run_is_production(const io::AnalysisConfig& cfg, int run_number) {
    if (cfg.runinfo_json.empty()) return true;

    const std::string run_key = std::to_string(run_number);
    if (!cfg.runinfo_json.contains(run_key)) return false;

    const auto& r = cfg.runinfo_json.at(run_key);
    if (!r.contains("run_type")) return false;

    return r.at("run_type").get<std::string>() == "production";
}

double get_runinfo_number_or(const io::AnalysisConfig& cfg,
                             int run_number,
                             const std::vector<std::string>& keys,
                             double fallback) 
{
    if (cfg.runinfo_json.empty()) return fallback;

    const std::string run_key = std::to_string(run_number);
    if (!cfg.runinfo_json.contains(run_key)) return fallback;

    const auto& r = cfg.runinfo_json.at(run_key);
    for (const std::string& key : keys) {
        if (r.contains(key) && r.at(key).is_number()) return r.at(key).get<double>();
    }

    return fallback;
}

std::vector<int> build_selected_run_list(const io::AnalysisConfig& cfg) {
    std::vector<int> runs;
    
    for (int run = cfg.start_run; run <= cfg.end_run; ++run) {
        if (!run_is_allowed(cfg, run)) continue;
        if (!run_is_production(cfg, run)) continue;

        runs.push_back(run);
    }

    return runs;
}

std::vector<int> select_shard(const std::vector<int>& all_runs, int shard_index, int num_shards) {
    std::vector<int> selected;
    for (std::size_t i = 0; i < all_runs.size(); ++i) {
        if (static_cast<int>(i % static_cast<std::size_t>(num_shards)) == shard_index) {
            selected.push_back(all_runs[i]);
        }
    }

    return selected;
}

std::string shard_name(int shard_index) {
    std::ostringstream ss;
    ss << "task_" << std::setw(3) << std::setfill('0') << shard_index;
    return ss.str();
}

fs::path make_output_dir(const io::AnalysisConfig& cfg) {
    fs::path out_dir = fs::path(cfg.output_folder);

    if (cfg.num_shards > 1) {
        out_dir /= "array";
        out_dir /= shard_name(cfg.shard_index);
    }

    fs::create_directories(out_dir);
    return out_dir;
}

void write_summary_header(std::ofstream& out) {
    out << "run,segment,hold_time_s,start_us,"
       "background_start_us,background_end_us,"
       "signal_start_us,signal_end_us,"
       "end_start_us,end_end_us,"
       "n_background_pulses,n_signal_pulses,n_end_pulses,"
       "n_background_windows,n_signal_windows,n_end_windows,"
       "background_rate_hz\n";
}

void write_pulse_header(std::ofstream& out) {
    out << "run,segment,region,time_us,amplitude_pe,window_index,window_width_us,"
       "is_pileup,uses_fine_bins,background_rate_hz\n";
}

void write_window_header(std::ofstream& out) {
    out << "run,segment,window_index,start_time_us,end_time_us,bin_width_us,"
           "pulse_count,observed_count,expected_count,final_nll\n";
}

void append_tagged_pulses(std::ofstream& out,
                          const std::string& run_id,
                          const std::string& segment,
                          const std::string& region_name,
                          const std::vector<TaggedPulse>& pulses) {
    for (const TaggedPulse& p : pulses) {
        out << run_id << ','
            << segment << ','
            << region_name << ','
            << std::setprecision(17) << p.time_us << ','
            << p.amplitude_pe << ','
            << p.window_index << ','
            << p.window_width_us << ','
            << (p.is_pileup ? 1 : 0) << ','
            << (p.uses_fine_bins ? 1 : 0) << '\n';
    }
}

void append_window_rows(std::ofstream& out,
                        const std::string& run_id,
                        const std::string& segment,
                        const std::vector<WindowSummary>& windows) {
    for (const WindowSummary& w : windows) {
        out << run_id << ','
            << segment << ','
            << w.window_index << ','
            << std::setprecision(17) << w.start_time_us << ','
            << w.end_time_us << ','
            << w.bin_width_us << ','
            << w.pulse_count << ','
            << w.observed_count << ','
            << w.expected_count << ','
            << w.final_nll << '\n';
    }
}

} // namespace

BatchAnalysisRunner::BatchAnalysisRunner(const io::AnalysisConfig& cfg)
    : cfg_(cfg) {}

void BatchAnalysisRunner::run() const {
    const fs::path out_dir = make_output_dir(cfg_);

    const std::vector<int> all_runs = build_selected_run_list(cfg_);
    const std::vector<int> my_runs = select_shard(
        all_runs,
        cfg_.shard_index,
        cfg_.num_shards
    );

    const fs::path metadata_path = out_dir / "analysis_metadata.json";
    const fs::path summary_path  = out_dir / "run_segment_summary.csv";
    const fs::path pulses_path   = out_dir / "all_pulses.csv";
    const fs::path windows_path  = out_dir / "all_windows.csv";

    std::ofstream meta_out(metadata_path);

    meta_out << "{\n"
             << "  \"year\": " << cfg_.year << ",\n"
             << "  \"start_run\": " << cfg_.start_run << ",\n"
             << "  \"end_run\": " << cfg_.end_run << ",\n"
             << "  \"restrict_to_good_runs\": "
             << (cfg_.restrict_to_good_runs ? "true" : "false") << ",\n"
             << "  \"shard_index\": " << cfg_.shard_index << ",\n"
             << "  \"num_shards\": " << cfg_.num_shards << ",\n"
             << "  \"n_selected_runs_total\": " << all_runs.size() << ",\n"
             << "  \"n_selected_runs_this_shard\": " << my_runs.size() << "\n"
             << "}\n";

    std::ofstream summary_out(summary_path);
    std::ofstream pulses_out(pulses_path);
    std::ofstream windows_out(windows_path);

    if (!summary_out.is_open() || !pulses_out.is_open() || !windows_out.is_open()) {
        throw std::runtime_error("Could not open one or more output CSV files.");
    }

    write_summary_header(summary_out);
    write_pulse_header(pulses_out);
    write_window_header(windows_out);

    GaussianTripPulseTemplate pulse_template(
        cfg_.template_config.native_bin_width_us,
        cfg_.template_config.support_end_us,
        cfg_.template_config.baseline,
        cfg_.template_config.gauss_amp,
        cfg_.template_config.gauss_mu,
        cfg_.template_config.gauss_sigma,
        cfg_.template_config.tail_start_us,
        cfg_.template_config.a1, cfg_.template_config.tau1,
        cfg_.template_config.a2, cfg_.template_config.tau2,
        cfg_.template_config.a3, cfg_.template_config.tau3
    );

    GreedyLRTFitter fitter(pulse_template);
    WindowedPulseProcessor processor(pulse_template, fitter);

    std::cout << "Selected " << all_runs.size() << " production/good runs total.\n"
              << "Shard " << cfg_.shard_index << " / " << cfg_.num_shards
              << " will process " << my_runs.size() << " runs.\n"
              << "Output directory: " << out_dir << '\n';

    int n_runs_processed = 0;
    int n_runs_failed = 0;

    for (int run = cfg_.start_run; run <= cfg_.end_run; ++run) {
        try {
            const io::LoadedRun loaded = io::load_root_run(cfg_, run);
            const io::RunWindow window = io::resolve_run_window(cfg_, run);

            const double hold_time_s = get_runinfo_number_or(
                cfg_,
                run,
                {"hold_time", "Holding Time", "holding_time"},
                -1.0
            );

            std::cout << "Run " << loaded.run_id
                      << " loaded. Segments = " << loaded.segments.size()
                      << ", hold_time_s = " << hold_time_s
                      << '\n';

            for (const io::LoadedSegment& seg : loaded.segments) {
                if (seg.hits.empty()) {
                    std::cerr << "Run " << loaded.run_id
                              << " segment " << seg.segment_name
                              << " has no hits. Skipping segment.\n";
                    continue;
                }

                RegionResult result = processor.analyze(
                    seg.hits,
                    window.background_start_us,
                    window.background_end_us,
                    window.signal_start_us,
                    window.signal_end_us,
                    window.end_start_us,
                    window.end_end_us,
                    cfg_.region_settings,
                    cfg_.fit_settings
                );

                summary_out << loaded.run_id << ','
                            << seg.segment_name << ','
                            << std::setprecision(17) << hold_time_s << ','
                            << window.start_us << ','
                            << window.background_start_us << ','
                            << window.background_end_us << ','
                            << window.signal_start_us << ','
                            << window.signal_end_us << ','
                            << window.end_start_us << ','
                            << window.end_end_us << ','
                            << result.background_pulses.size() << ','
                            << result.signal_pulses.size() << ','
                            << result.end_pulses.size() << ','
                            << result.background_window_summaries.size() << ','
                            << result.signal_window_summaries.size() << ','
                            << result.end_window_summaries.size() << ','
                            << result.background_rate_hz << '\n';

                append_window_rows(
                    windows_out,
                    loaded.run_id,
                    seg.segment_name,
                    hold_time_s,
                    "background",
                    result.background_window_summaries
                );

                append_window_rows(
                    windows_out,
                    loaded.run_id,
                    seg.segment_name,
                    hold_time_s,
                    "signal",
                    result.signal_window_summaries
                );

                append_window_rows(
                    windows_out,
                    loaded.run_id,
                    seg.segment_name,
                    hold_time_s,
                    "end",
                    result.end_window_summaries
                );

                summary_out.flush();
                pulses_out.flush();
                windows_out.flush();
            }

            ++n_runs_processed;
        } catch (const std::exception& e) {
            ++n_runs_failed;
            std::cerr << "Skipping run " << run << " because: " << e.what() << '\n';
        }
    }

    std::cout << "Finished shard " << cfg_.shard_index << " / " << cfg_.num_shards << ".\n"
              << "  Runs processed: " << n_runs_processed << '\n'
              << "  Runs failed:    " << n_runs_failed << '\n'
              << "  Summary:        " << summary_path << '\n'
              << "  Pulses:         " << pulses_path << '\n'
              << "  Windows:        " << windows_path << '\n';
}

} // namespace ucn::app