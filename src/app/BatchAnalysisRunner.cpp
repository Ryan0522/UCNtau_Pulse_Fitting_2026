#include "ucn/app/BatchAnalysisRunner.hpp"
#include "ucn/app/WindowedPulseProcessor.hpp"
#include "ucn/inference/GreedyLRTFitter.hpp"
#include "ucn/io/RootRunLoader.hpp"
#include "ucn/templates/GaussianTripPulseTemplate.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace ucn::app {

namespace {

bool run_is_allowed(const io::AnalysisConfig& cfg, int run_number) {
    if (!cfg.restrict_to_good_runs || cfg.good_runs.empty()) {
        return true;
    }
    return cfg.good_runs.count(std::to_string(run_number)) > 0;
}

void write_summary_header(std::ofstream& out) {
    out << "run,segment,signal_start_us,signal_end_us,background_start_us,"
           "n_signal_pulses,n_background_pulses,n_windows,background_rate_hz\n";
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
    fs::create_directories(cfg_.output_folder);

    const fs::path metadata_path = fs::path(cfg_.output_folder) / "analysis_metadata.json";
    std::ofstream meta_out(metadata_path);
    meta_out << "{\n"
            << "  \"start_run\": " << cfg_.start_run << ",\n"
            << "  \"end_run\": " << cfg_.end_run << ",\n"
            << "  \"year\": " << cfg_.year << "\n"
            << "}\n";

    const fs::path summary_path = fs::path(cfg_.output_folder) / "run_segment_summary.csv";
    const fs::path pulses_path  = fs::path(cfg_.output_folder) / "all_pulses.csv";
    const fs::path windows_path = fs::path(cfg_.output_folder) / "all_windows.csv";

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

    for (int run = cfg_.start_run; run <= cfg_.end_run; ++run) {
        if (!run_is_allowed(cfg_, run)) {
            continue;
        }

        try {
            const io::LoadedRun loaded = io::load_root_run(cfg_, run);
            const io::RunWindow window = io::resolve_run_window(cfg_, run);

            std::cout << "Run " << loaded.run_id
                      << " loaded. Segments = " << loaded.segments.size() << '\n';

            for (const io::LoadedSegment& seg : loaded.segments) {
                if (seg.hits.empty()) {
                    continue;
                }

                RegionResult result = processor.analyze(
                    seg.hits,
                    window.signal_start_us,
                    window.signal_end_us,
                    window.background_start_us,
                    cfg_.region_settings,
                    cfg_.fit_settings
                );

                summary_out << loaded.run_id << ','
                            << seg.segment_name << ','
                            << std::setprecision(17) << window.signal_start_us << ','
                            << window.signal_end_us << ','
                            << window.background_start_us << ','
                            << result.signal_pulses.size() << ','
                            << result.background_pulses.size() << ','
                            << result.window_summaries.size() << ','
                            << result.background_rate_hz << '\n';

                append_tagged_pulses(
                    pulses_out, loaded.run_id, seg.segment_name, "signal", result.signal_pulses
                );
                append_tagged_pulses(
                    pulses_out, loaded.run_id, seg.segment_name, "background", result.background_pulses
                );
                append_window_rows(
                    windows_out, loaded.run_id, seg.segment_name, result.window_summaries
                );
            }
        } catch (const std::exception& e) {
            std::cerr << "Skipping run " << run << " because: " << e.what() << '\n';
        }
    }

    std::cout << "Finished.\n"
              << "  Summary: " << summary_path << '\n'
              << "  Pulses:  " << pulses_path << '\n'
              << "  Windows: " << windows_path << '\n';
}

} // namespace ucn::app