#include "ucn/app/WindowedPulseProcessor.hpp"
#include "ucn/inference/GreedyLRTFitter.hpp"
#include "ucn/templates/GaussianTripPulseTemplate.hpp"
#include "ucn/io/AnalysisConfig.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>

std::vector<ucn::Hit> load_mc_hits(const std::string& path) {
    std::vector<ucn::Hit> hits;
    std::ifstream file(path);
    std::string line;
    // skip header (Header)
    std::getline(file, line); 
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
    return hits;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: test_mc_csv <config.json> <input_hits.csv> <output postfix (12, 34, etc.)>\n";
        return 1;
    }

    try {
        ucn::io::AnalysisConfig cfg = ucn::io::load_analysis_config(argv[1]);
        
        ucn::GaussianTripPulseTemplate pulse_template(
            cfg.template_config.native_bin_width_us,
            cfg.template_config.support_end_us,
            cfg.template_config.baseline,
            cfg.template_config.gauss_amp,
            cfg.template_config.gauss_mu,
            cfg.template_config.gauss_sigma,
            cfg.template_config.tail_start_us,
            cfg.template_config.a1, cfg.template_config.tau1,
            cfg.template_config.a2, cfg.template_config.tau2,
            cfg.template_config.a3, cfg.template_config.tau3
        );

        ucn::GreedyLRTFitter fitter(pulse_template);
        ucn::WindowedPulseProcessor processor(pulse_template, fitter);
        
        // debug
        bool debug = false;
        if (debug) {
            processor.set_debug_max_windows(3);
            cfg.region_settings.debug = true;
            cfg.fit_settings.debug = true;
        } else {
            processor.set_debug_max_windows(-1);
            cfg.region_settings.debug = false;
            cfg.fit_settings.debug = false;
        }
        
        // 3. load MC data
        std::vector<ucn::Hit> hits = load_mc_hits(argv[2]);
        std::cout << "Loaded " << hits.size() << " hits from MC CSV.\n";

        // 4. analyze
        double signal_start = 10 * 1e6;
        double signal_end = 31 * 1e6;
        double bg_start = 0.0;
        ucn::RegionResult result = processor.analyze(
            hits, 
            signal_start, signal_end,  // Signal Window
            bg_start,               // Disable background fit if not needed
            cfg.region_settings,
            cfg.fit_settings
        );

        std::cout << "Analysis complete. Found " << result.signal_pulses.size() << " signal pulses.\n";

        // 5. output for Python test (all_pulses.csv format)
        std::string postfix = argv[3];
        std::string pulse_file = "test/fit_results_seg" + postfix + ".csv";
        std::string window_file = "test/win_summaries_seg" + postfix + ".csv";
        
        std::ofstream out(pulse_file);
        out << std::fixed << std::setprecision(3);
        out << "time_us,amplitude_pe,is_pileup, window_index\n";
        for (const auto& p : result.signal_pulses) {
            out << p.time_us << "," << p.amplitude_pe << "," 
                << (p.is_pileup ? 1 : 0) << "," << p.window_index << "\n";
        }

        std::ofstream win_out(window_file);
        win_out << std::fixed << std::setprecision(3);
        win_out << "window_index,final_nll,seed_count,pulse_count,observed_count,expected_count\n";
        for (const auto& w : result.window_summaries) {
            win_out << w.window_index << "," << w.final_nll << "," << w.seed_count << "," 
                    << w.pulse_count << "," << w.observed_count << "," << w.expected_count << "\n";
        }

        std::cout << "Found " << result.signal_pulses.size() << " pulses. Output saved.\n";
        std::cout << "Saved pulses to: " << pulse_file << "\n";
        std::cout << "Saved window summaries to: " << window_file << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
    return 0;
}