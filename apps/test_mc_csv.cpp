#include "ucn/app/WindowedPulseProcessor.hpp"
#include "ucn/inference/GreedyLRTFitter.hpp"
#include "ucn/templates/GaussianTripPulseTemplate.hpp"
#include "ucn/templates/GaussianQuadPulseTemplate.hpp"
#include "ucn/io/AnalysisConfig.hpp"
#include "ucn/debug/DebugCsvWriter.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

std::vector<ucn::Hit> load_mc_hits(const std::string& path) {
    std::vector<ucn::Hit> hits;
    std::ifstream file(path);
    std::string line;

    std::getline(file, line);  // header

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string t_str, c_str;

        if (std::getline(ss, t_str, ',') && std::getline(ss, c_str, ',')) {
            ucn::Hit h;
            h.time_us = std::stod(t_str);
            h.channel = std::stoi(c_str);
            hits.push_back(h);
        }
    }

    std::sort(hits.begin(), hits.end(),
              [](const ucn::Hit& a, const ucn::Hit& b) {
                  return a.time_us < b.time_us;
              });

    return hits;
}

std::vector<ucn::debug::TruthPulse> load_mc_truth(const std::string& path) {
    std::vector<ucn::debug::TruthPulse> truth;

    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open MC truth CSV: " + path);
    }

    std::string line;
    std::getline(file, line); // header

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string t_str;
        std::string amp_str;

        if (!std::getline(ss, t_str, ',')) continue;
        if (!std::getline(ss, amp_str, ',')) amp_str = "0.0";

        ucn::debug::TruthPulse p;
        p.time_us = std::stod(t_str);
        p.amplitude_pe = std::stod(amp_str);
        truth.push_back(p);
    }

    std::sort(truth.begin(), truth.end(),
              [](const auto& a, const auto& b) {
                  return a.time_us < b.time_us;
              });

    return truth;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: test_mc_csv <config.json> <input_hits.csv> <truth.csv> <output postfix>\n";
        return 1;
    }

    try {
        ucn::io::AnalysisConfig cfg = ucn::io::load_analysis_config(argv[1]);

        ucn::GaussianQuadPulseTemplate pulse_template(
            cfg.template_config.native_bin_width_us,
            cfg.template_config.support_end_us,
            cfg.template_config.use_smooth_tail_onset,
            cfg.template_config.baseline,
            cfg.template_config.gauss_amp,
            cfg.template_config.gauss_mu,
            cfg.template_config.gauss_sigma,
            cfg.template_config.tail_start_us,
            cfg.template_config.tail_width_us,
            cfg.template_config.a1, cfg.template_config.tau1,
            cfg.template_config.a2, cfg.template_config.tau2,
            cfg.template_config.a3, cfg.template_config.tau3,
            cfg.template_config.a4, cfg.template_config.tau4
        );

        ucn::GreedyLRTFitter fitter(pulse_template);
        ucn::WindowedPulseProcessor processor(pulse_template, fitter);

        std::shared_ptr<ucn::debug::DebugCsvWriter> debug_writer;

        std::vector<ucn::Hit> hits = load_mc_hits(argv[2]);
        std::vector<ucn::debug::TruthPulse> truth = load_mc_truth(argv[3]);
        std::string postfix = argv[4];

        if (cfg.debug_max_windows > 0 && cfg.shard_index == 0) {
            const std::filesystem::path debug_dir =
                std::filesystem::path("test") / "debug" / postfix / "task_000";

            debug_writer = std::make_shared<ucn::debug::DebugCsvWriter>(
                debug_dir,
                cfg.debug_max_windows,
                pulse_template
            );

            processor.set_debug_writer(debug_writer);
        }

        if (hits.empty()) {
            std::cerr << "No hits found.\n";
            return 1;
        }

        std::filesystem::create_directories("test");

        std::string pulse_file = "test/fit_results_seg" + postfix + ".csv";
        std::string window_file = "test/win_summaries_seg" + postfix + ".csv";

        std::ofstream out(pulse_file);
        std::ofstream win_out(window_file);

        out << std::fixed << std::setprecision(3);
        win_out << std::fixed << std::setprecision(3);

        out << "window_index,final_nll,seed_count,pulse_count,"
               "observed_count,expected_count,start_time_us,end_time_us,"
               "fitted_pe_sum,fit_expected_sum,fixed_expected_sum,"
               "background_expected_sum,model_expected_sum,"
               "template_mass_in_window,pe_per_observed_count,"
               "background_fraction,fit_fraction,"
               "chunk_index\n";

        // MC signal starts at 10 s.
        const double start_us = 10.0e6;

        // Your current long test was effectively 1050 s.
        // For a 1000 s generated MC, set this to 1000.0.
        const double total_signal_s = 1050.0;

        // Safeguard chunk size.
        const double chunk_s = 60.0;

        const double hold_time_s = -1.0;

        const int n_chunks = static_cast<int>(
            std::ceil(total_signal_s / chunk_s)
        );

        int total_pulses = 0;
        int total_windows = 0;
        int global_window_offset = 0;

        for (int chunk = 0; chunk < n_chunks; ++chunk) {
            const double chunk_start_us = start_us + chunk * chunk_s * 1.0e6;
            const double chunk_end_us = std::min(
                start_us + total_signal_s * 1.0e6,
                chunk_start_us + chunk_s * 1.0e6
            );

            if (chunk_end_us <= chunk_start_us) {
                continue;
            }

            const double background_start = std::max(0.0, chunk_start_us - 60.0e6);
            const double background_end   = chunk_start_us;

            const double signal_start = chunk_start_us;
            const double signal_end   = chunk_end_us;

            const double end_start = chunk_start_us;
            const double end_end   = chunk_start_us;

            std::cout << "\n[chunk " << chunk + 1 << "/" << n_chunks << "] "
                      << "signal = "
                      << (signal_start - start_us) * 1.0e-6 << " to "
                      << (signal_end - start_us) * 1.0e-6 << " s\n";

            ucn::RegionResult result = processor.analyze(
                hits,
                background_start,
                background_end,
                signal_start,
                signal_end,
                end_start,
                end_end,
                cfg.region_settings,
                cfg.fit_settings,
                &truth,
                chunk,
                global_window_offset,
                hold_time_s
            );

            for (const auto& p : result.signal_pulses) {
                out << p.time_us << ","
                    << p.amplitude_pe << ","
                    << (p.is_pileup ? 1 : 0) << ","
                    << (p.window_index + global_window_offset)
                    << "\n";
            }

            int max_local_window_index = -1;

            for (const auto& w : result.signal_window_summaries) {
                const int global_window_index = w.window_index + global_window_offset;

                win_out << global_window_index << ","
                        << w.final_nll << ","
                        << w.seed_count << ","
                        << w.pulse_count << ","
                        << w.observed_count << ","
                        << w.expected_count << ","
                        << w.start_time_us << ","
                        << w.end_time_us << ","
                        << w.fitted_pe_sum << ","
                        << w.fit_expected_sum << ","
                        << w.fixed_expected_sum << ","
                        << w.background_expected_sum << ","
                        << w.template_mass_in_window << ","
                        << w.pe_per_observed_count << ","
                        << w.background_fraction << ","
                        << w.fit_fraction << ","
                        << chunk
                        << "\n";

                max_local_window_index = std::max(max_local_window_index, w.window_index);
            }

            total_pulses += static_cast<int>(result.signal_pulses.size());
            total_windows += static_cast<int>(result.signal_window_summaries.size());

            if (max_local_window_index >= 0) {
                global_window_offset += max_local_window_index + 1;
            }

            // Safeguard: write to disk after every 60 s chunk.
            out.flush();
            win_out.flush();

            std::cout << "[chunk " << chunk + 1 << "] found "
                      << result.signal_pulses.size() << " pulses, "
                      << result.signal_window_summaries.size() << " windows. "
                      << "Cumulative pulses = " << total_pulses << "\n";
        }

        std::cout << "\nAnalysis complete.\n";
        std::cout << "Total pulses: " << total_pulses << "\n";
        std::cout << "Total windows: " << total_windows << "\n";
        std::cout << "Saved pulses to: " << pulse_file << "\n";
        std::cout << "Saved window summaries to: " << window_file << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }

    return 0;
}