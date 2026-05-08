#include "ucn/app/BatchAnalysisRunner.hpp"
#include "ucn/app/WindowedPulseProcessor.hpp"
#include "ucn/inference/GreedyLRTFitter.hpp"
#include "ucn/inference/CoincidenceFitter.hpp"
#include "ucn/io/RootRunLoader.hpp"
#include "ucn/templates/GaussianTripPulseTemplate.hpp"
#include "ucn/templates/GaussianQuadPulseTemplate.hpp"
#include "ucn/templates/EmpiricalPulseTemplate.hpp"
#include "ucn/debug/DebugCsvWriter.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <cstdint>
#include <map>
#include <tuple>
#include <memory>

namespace fs = std::filesystem;

namespace ucn::app {

namespace {

bool run_is_allowed(const io::AnalysisConfig& cfg_, int run_number) {
    if (!cfg_.restrict_to_good_runs || cfg_.good_runs.empty()) {
        return true;
    }
    return cfg_.good_runs.count(std::to_string(run_number)) > 0;
}

bool run_is_production(const io::AnalysisConfig& cfg_, int run_number) {
    if (cfg_.runinfo_json.empty()) return true;

    const std::string run_key = std::to_string(run_number);
    if (!cfg_.runinfo_json.contains(run_key)) return false;

    const auto& r = cfg_.runinfo_json.at(run_key);
    if (!r.contains("run_type")) return false;

    return r.at("run_type").get<std::string>() == "production";
}

double get_runinfo_number_or(const io::AnalysisConfig& cfg_,
                             int run_number,
                             const std::vector<std::string>& keys,
                             double fallback) 
{
    if (cfg_.runinfo_json.empty()) return fallback;

    const std::string run_key = std::to_string(run_number);
    if (!cfg_.runinfo_json.contains(run_key)) return fallback;

    const auto& r = cfg_.runinfo_json.at(run_key);
    for (const std::string& key : keys) {
        if (r.contains(key) && r.at(key).is_number()) return r.at(key).get<double>();
    }

    return fallback;
}

std::vector<int> build_selected_run_list(const io::AnalysisConfig& cfg_) {
    std::vector<int> runs;
    
    for (int run = cfg_.start_run; run <= cfg_.end_run; ++run) {
        if (!run_is_allowed(cfg_, run)) continue;
        if (!run_is_production(cfg_, run)) continue;

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

std::string make_array_subdir_name(const io::AnalysisConfig& cfg_) {
    if (!cfg_.array_output_subdir.empty()) {
        return cfg_.array_output_subdir;
    }

    const double pe = cfg_.fit_settings.min_amplitude_pe;
    const double nll = cfg_.fit_settings.delta_nll_cut;
    const double wt = cfg_.region_settings.min_gap_us;
    const double wc = cfg_.region_settings.coincidence_window_us;

    std::ostringstream ss;
    ss << "array"
       << "_" << static_cast<int>(std::llround(pe)) << "PE"
       << "_" << static_cast<int>(std::llround(nll)) << "nll"
       << "_wc" << static_cast<int>(std::llround(wc * 1000)) << "ns"
       << "_wt" << static_cast<int>(std::llround(wt)) << "us";

    return ss.str();
}

fs::path make_output_dir(const io::AnalysisConfig& cfg_) {
    fs::path out_dir = fs::path(cfg_.output_folder);

    out_dir /= make_array_subdir_name(cfg_);
    if (cfg_.num_shards > 1) {
        out_dir /= shard_name(cfg_.shard_index);
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
    out << "run,segment,hold_time_s,region,time_us,amplitude_pe,"
           "window_index,width_us,is_pileup,uses_fine_bins,"
           "background_rate_hz\n";
}

void write_coincidence_header(std::ofstream& out) {
    out << "run,segment,hold_time_s,region, time_us, amplitude_pe,"
           "start_time_us,end_time_us,length_us,"
           "n_pe,n_pileup,free_pe_interval_us,"
           "passes_raw_threshold,passes_pileup_threshold\n";
}

void write_window_header(std::ofstream& out) {
    out << "run,segment,hold_time_s,region,window_index,"
           "start_time_us,end_time_us,width_us,bin_width_us,"
           "pulse_count,seed_count,observed_count,expected_count,final_nll,"
           "fitted_pe_sum,fit_expected_sum,fixed_expected_sum,"
           "background_expected_sum,"
           "template_mass_in_window,pe_per_observed_count,"
           "background_fraction,fit_fraction\n";
}

void write_tail_pulse_header(std::ofstream& out) {
    out << "tail_id,run,segment,hold_time_s,region,pulse_index_in_region,"
           "pulse_time_us,amplitude_pe,window_index,width_us,"
           "is_pileup,uses_fine_bins,bin_width_us,window_start_us,"
           "window_end_us,n_raw_hits\n";
}

constexpr double kTailPeBinWidth = 1.0;
using TailSumKey = std::tuple<std::string, std::string, int>; // segment, region, pe_bin
struct TailSumRow {
    long long n_events = 0;
    long long n_raw_hits = 0;
    double amp_sum = 0.0;
    std::vector<std::uint64_t> counts;
};
using TailSumMap = std::map<TailSumKey, TailSumRow>;

void write_tail_sum_header(std::ofstream& out, int n_bins) {
    out << "segment,region,pe_bin,pe_low,pe_high,"
           "n_events,n_raw_hits,amp_sum,amp_mean";

    const char old_fill = out.fill();

    for (int b = 0; b < n_bins; ++b) {
        out << ",b" << std::setw(4) << std::setfill('0') << b;
    }

    out << std::setfill(old_fill) << '\n';
}

void write_tail_sums(std::ofstream& out,
                     const TailSumMap& sums,
                     int n_bins) {
    for (const auto& kv : sums) {
        const auto& key = kv.first;
        const TailSumRow& row = kv.second;

        const std::string& segment = std::get<0>(key);
        const std::string& region = std::get<1>(key);
        const int pe_bin = std::get<2>(key);

        const double pe_low = static_cast<double>(pe_bin) * kTailPeBinWidth;
        const double pe_high = static_cast<double>(pe_bin + 1) * kTailPeBinWidth;
        const double amp_mean =
            row.n_events > 0 ? row.amp_sum / static_cast<double>(row.n_events) : 0.0;
        
        out << segment << ','
            << region << ','
            << pe_bin << ','
            << pe_low << ','
            << pe_high << ','
            << row.n_events << ','
            << row.n_raw_hits << ','
            << std::setprecision(17) << row.amp_sum << ','
            << amp_mean;

        for (int b = 0; b < n_bins; ++b) {
            const std::uint64_t c =
                b < static_cast<int>(row.counts.size()) ? row.counts[static_cast<std::size_t>(b)] : 0;
            out << ',' << c;
        }

        out << '\n';
    }
}

void append_tagged_pulses(std::ofstream& out,
                          const std::string& run_id,
                          const std::string& segment,
                          double hold_time_s,
                          const std::string& region_name,
                          const std::vector<TaggedPulse>& pulses,
                          double background_rate_hz) {
    for (const TaggedPulse& p : pulses) {
        out << run_id << ','
            << segment << ','
            << std::setprecision(17) << hold_time_s << ','
            << region_name << ','
            << p.time_us << ','
            << p.amplitude_pe << ','
            << p.window_index << ','
            << p.width_us << ','
            << (p.is_pileup ? 1 : 0) << ','
            << (p.uses_fine_bins ? 1 : 0) << ','
            << background_rate_hz << '\n';
    }
}

void append_window_rows(std::ofstream& out,
                        const std::string& run_id,
                        const std::string& segment,
                        double hold_time_s,
                        const std::string& region_name,
                        const std::vector<WindowSummary>& windows) {
    for (const WindowSummary& w : windows) {
        out << run_id << ','
            << segment << ','
            << std::setprecision(17) << hold_time_s << ','
            << region_name << ','
            << w.window_index << ','
            << w.start_time_us << ','
            << w.end_time_us << ','
            << w.width_us << ','
            << w.bin_width_us << ','
            << w.pulse_count << ','
            << w.seed_count << ','
            << w.observed_count << ','
            << w.expected_count << ','
            << w.final_nll << ','
            << w.fitted_pe_sum << ','
            << w.fit_expected_sum << ','
            << w.fixed_expected_sum << ','
            << w.background_expected_sum << ','
            << w.template_mass_in_window << ','
            << w.pe_per_observed_count << ','
            << w.background_fraction << ','
            << w.fit_fraction << '\n';
    }
}

std::string classify_region(double t_us, const ucn::io::RunWindow& w) {
    if (t_us >= w.background_start_us && t_us < w.background_end_us) {
        return "background";
    }
    if (t_us >= w.signal_start_us && t_us < w.signal_end_us) {
        return "signal";
    }
    if (t_us >= w.end_start_us && t_us < w.end_end_us) {
        return "end";
    }
    return "";
}

void append_coincidence_rows(std::ofstream& out,
                             const std::string& run_id,
                             const std::string& segment,
                             double hold_time_s,
                             const ucn::io::RunWindow& window,
                             const std::vector<ucn::CoincidenceEvent>& events) {
    for (const auto& ev : events) {
        const std::string region = classify_region(ev.start_time_us, window);
        if (region.empty()) {
            continue;
        }

        out << run_id << ','
            << segment << ','
            << std::setprecision(17) << hold_time_s << ','
            << region << ','
            << ev.start_time_us << ','
            << ev.n_pe << ','
            << ev.start_time_us << ','
            << ev.end_time_us << ','
            << ev.length_us << ','
            << ev.n_pe << ','
            << ev.n_pileup << ','
            << ev.free_pe_interval_us << ','
            << (ev.passes_raw_threshold ? 1 : 0) << ','
            << (ev.passes_pileup_threshold ? 1 : 0) << '\n';
    }
}

bool tail_region_enabled(const io::TailExtractionSettings& settings,
                         const std::string& region_name) {
    if (settings.regions.empty()) return true;
    return std::find(settings.regions.begin(), settings.regions.end(), region_name) != settings.regions.end();
}

bool has_nearby_fitted_pulse(const std::vector<TaggedPulse>& pulses, std::size_t i, double min_separation_us) {
    if (min_separation_us <= 0.0) return false;
    
    const double ti = pulses[i].time_us;
    for (std::size_t j = 0; j < pulses.size(); ++j) {
        if (j == i) continue;
        if (std::abs(pulses[j].time_us - ti) < min_separation_us) return true;
    }
    return false;
}

void accumulate_tail_waveforms(std::ofstream& tail_pulses_out,
                               TailSumMap& tail_sums,
                               const std::string& run_id,
                               const std::string& segment,
                               double hold_time_s,
                               const std::string& region_name,
                               const std::vector<Hit>& raw_hits,
                               const std::vector<TaggedPulse>& pulses,
                               const io::TailExtractionSettings& settings) {
    if (!settings.enable) return;
    if (!tail_region_enabled(settings, region_name)) return;
    if (settings.bin_width_us <= 0.0 || settings.window_us <= 0.0) return;

    const int n_bins = static_cast<int>(std::ceil(settings.window_us / settings.bin_width_us));
    if (n_bins <= 0) return;

    for (std::size_t ip = 0; ip < pulses.size(); ++ip) {
        const TaggedPulse& p = pulses[ip];

        if (settings.only_non_pileup && p.is_pileup) continue;
        if (p.amplitude_pe < settings.min_amplitude_pe) continue;
        if (p.amplitude_pe > settings.max_amplitude_pe) continue;
        if (has_nearby_fitted_pulse(pulses, ip, settings.min_neighbor_separation_us)) continue;

        const double window_start_us = p.time_us - settings.pretrigger_us;
        const double window_end_us = window_start_us + static_cast<double>(n_bins) * settings.bin_width_us;

        const int pe_bin = static_cast<int>(std::floor(p.amplitude_pe / kTailPeBinWidth));
        if (pe_bin < 0) continue;

        TailSumKey key{segment, region_name, pe_bin};
        TailSumRow& row = tail_sums[key];

        if (row.counts.empty()) {
            row.counts.assign(static_cast<std::size_t>(n_bins), 0);
        } else if (static_cast<int>(row.counts.size()) != n_bins) {
            throw std::runtime_error("Tail sum row has inconsistent n_bins.");
        }
        
        auto first = std::lower_bound(
            raw_hits.begin(), raw_hits.end(), window_start_us, [](const Hit& h, double t) { return h.time_us < t; }
        );

        int n_raw_hits_this_tail = 0;
        for (auto it = first; it != raw_hits.end() && it->time_us < window_end_us; ++it) {
            const int bin = static_cast<int>((it->time_us - window_start_us) / settings.bin_width_us);
            if (bin >= 0 && bin < n_bins) {
                row.counts[static_cast<std::size_t>(bin)] += 1;
                ++n_raw_hits_this_tail;
            }
        }

        row.n_events += 1;
        row.n_raw_hits += n_raw_hits_this_tail;
        row.amp_sum += p.amplitude_pe;

        std::ostringstream id;
        id << run_id << '_' << segment << '_' << region_name << '_'
           << ip << '_'
           << static_cast<long long>(std::llround(p.time_us * 1000.0));
        const std::string tail_id = id.str();

        tail_pulses_out << tail_id << ','
                        << run_id << ','
                        << segment << ','
                        << std::setprecision(17) << hold_time_s << ','
                        << region_name << ','
                        << ip << ','
                        << p.time_us << ','
                        << p.amplitude_pe << ','
                        << p.window_index << ','
                        << p.width_us << ','
                        << (p.is_pileup ? 1 : 0) << ','
                        << (p.uses_fine_bins ? 1 : 0) << ','
                        << settings.bin_width_us << ','
                        << window_start_us << ','
                        << window_end_us << ','
                        << n_raw_hits_this_tail << '\n';
    }
}

} // namespace

BatchAnalysisRunner::BatchAnalysisRunner(const io::AnalysisConfig& cfg_)
    : cfg_(cfg_) {}

void BatchAnalysisRunner::run() const {
    const fs::path out_dir = make_output_dir(cfg_);

    const std::vector<int> all_runs = build_selected_run_list(cfg_);
    const std::vector<int> my_runs = select_shard(
        all_runs,
        cfg_.shard_index,
        cfg_.num_shards
    );

    const fs::path metadata_path     = out_dir / "analysis_metadata.json";
    const fs::path summary_path      = out_dir / "run_segment_summary.csv";
    const fs::path pulses_path       = out_dir / "all_pulses.csv";
    const fs::path windows_path      = out_dir / "all_windows.csv";
    const fs::path coincidences_path = out_dir / "all_coincidences.csv";
    const fs::path tail_pulses_path  = out_dir / "all_tail_pulses.csv";
    const fs::path tail_sums_path    = out_dir / "all_tail_sums_by_pe.csv";

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
    std::ofstream coincidences_out(coincidences_path);
    std::ofstream tail_pulses_out;
    TailSumMap tail_sums;
    int tail_n_bins = 0;

    if (cfg_.tail_extraction.enable) {
        if (cfg_.tail_extraction.bin_width_us <= 0.0 || cfg_.tail_extraction.window_us <= 0.0) {
            throw std::runtime_error(
                "Tail extraction enabled, but bin_width_us or window_us is non-positive."
            );
        }

        tail_n_bins = static_cast<int>(
            std::ceil(cfg_.tail_extraction.window_us /
                    cfg_.tail_extraction.bin_width_us)
        );

        if (tail_n_bins <= 0) {
            throw std::runtime_error("Tail extraction produced non-positive n_bins.");
        }

        tail_pulses_out.open(tail_pulses_path);

        if (!tail_pulses_out.is_open()) {
            throw std::runtime_error("Could not open tail pulse metadata CSV file.");
        }

        write_tail_pulse_header(tail_pulses_out);
    }

    if (!summary_out.is_open() || !pulses_out.is_open() || !windows_out.is_open() || !coincidences_out.is_open()) {
        throw std::runtime_error("Could not open one or more output CSV files.");
    }

    write_summary_header(summary_out);
    write_pulse_header(pulses_out);
    write_window_header(windows_out);
    write_coincidence_header(coincidences_out);

    std::unique_ptr<ucn::PulseTemplate> pulse_template;
    if (cfg_.template_config.type == "empirical") {
        pulse_template = std::make_unique<ucn::EmpiricalPulseTemplate>(
            cfg_.template_config.native_bin_width_us,
            cfg_.template_config.support_end_us,
            cfg_.template_config.empirical_csv_path,
            cfg_.template_config.empirical_pretrigger_us
        );
    } else {
        pulse_template = std::make_unique<ucn::GaussianQuadPulseTemplate>(
            cfg_.template_config.native_bin_width_us,
            cfg_.template_config.support_end_us,
            cfg_.template_config.use_smooth_tail_onset,
            cfg_.template_config.baseline,
            cfg_.template_config.gauss_amp,
            cfg_.template_config.gauss_mu,
            cfg_.template_config.gauss_sigma,
            cfg_.template_config.tail_start_us,
            cfg_.template_config.tail_width_us,
            cfg_.template_config.a1, cfg_.template_config.tau1,
            cfg_.template_config.a2, cfg_.template_config.tau2,
            cfg_.template_config.a3, cfg_.template_config.tau3,
            cfg_.template_config.a4, cfg_.template_config.tau4
        );
    }

    const auto pmf = pulse_template->pmf();
    const auto it = std::max_element(pmf.begin(), pmf.end());
    const auto idx = std::distance(pmf.begin(), it);
    const double sum = std::accumulate(pmf.begin(), pmf.end(), 0.0);

    std::cout << "[TEMPLATE]"
              << " type=" << cfg_.template_config.type
              << " n_bins=" << pmf.size()
              << " bin_width_us=" << pulse_template->native_bin_width_us()
              << " peak_local_time_us="
              << (static_cast<double>(idx) + 0.5) * pulse_template->native_bin_width_us()
              << " empirical_pretrigger_us="
              << cfg_.template_config.empirical_pretrigger_us
              << " sum=" << sum
              << "\n";

    GreedyLRTFitter fitter(*pulse_template);
    WindowedPulseProcessor processor(*pulse_template, fitter);
    
    std::shared_ptr<debug::DebugCsvWriter> debug_writer;

    if (cfg_.debug_max_windows > 0 && cfg_.shard_index == 0) {
        const fs::path debug_dir = out_dir / "debug";

        debug_writer = std::make_shared<debug::DebugCsvWriter>(
            debug_dir,
            cfg_.debug_max_windows,
            *pulse_template
        );

        processor.set_debug_writer(debug_writer);
    }
    
    CoincidenceFitter coincidence_fitter(cfg_.coincidence_settings);

    std::cout << "Selected " << all_runs.size() << " production/good runs total.\n"
              << "Shard " << cfg_.shard_index << " / " << cfg_.num_shards
              << " will process " << my_runs.size() << " runs.\n"
              << "Output directory: " << out_dir << '\n';

    int n_runs_processed = 0;
    int n_runs_failed = 0;

    for (int run : my_runs) {
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
                    cfg_.fit_settings,
                    nullptr, -1, 0, hold_time_s
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

                append_tagged_pulses(
                    pulses_out,
                    loaded.run_id,
                    seg.segment_name,
                    hold_time_s,
                    "background",
                    result.background_pulses,
                    result.background_rate_hz
                );

                append_tagged_pulses(
                    pulses_out,
                    loaded.run_id,
                    seg.segment_name,
                    hold_time_s,
                    "signal",
                    result.signal_pulses,
                    result.background_rate_hz
                );

                append_tagged_pulses(
                    pulses_out,
                    loaded.run_id,
                    seg.segment_name,
                    hold_time_s,
                    "end",
                    result.end_pulses,
                    result.background_rate_hz
                );

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

                if (cfg_.tail_extraction.enable) {
                    accumulate_tail_waveforms(
                        tail_pulses_out,
                        tail_sums,
                        loaded.run_id,
                        seg.segment_name,
                        hold_time_s,
                        "background",
                        seg.hits,
                        result.background_pulses,
                        cfg_.tail_extraction
                    );

                    accumulate_tail_waveforms(
                        tail_pulses_out,
                        tail_sums,
                        loaded.run_id,
                        seg.segment_name,
                        hold_time_s,
                        "signal",
                        seg.hits,
                        result.signal_pulses,
                        cfg_.tail_extraction
                    );

                    accumulate_tail_waveforms(
                        tail_pulses_out,
                        tail_sums,
                        loaded.run_id,
                        seg.segment_name,
                        hold_time_s,
                        "end",
                        seg.hits,
                        result.end_pulses,
                        cfg_.tail_extraction
                    );

                    tail_pulses_out.flush();
                }

                summary_out.flush();
                pulses_out.flush();
                windows_out.flush();

                if (cfg_.enable_coincidence_output) {
                    const CoincidenceFitResult coinc_result = coincidence_fitter.find(seg.hits);
                    
                    append_coincidence_rows(
                        coincidences_out,
                        loaded.run_id,
                        seg.segment_name,
                        hold_time_s,
                        window,
                        coinc_result.events
                    );

                    coincidences_out.flush();
                }
            }

            ++n_runs_processed;
        } catch (const std::exception& e) {
            ++n_runs_failed;
            std::cerr << "Skipping run " << run << " because: " << e.what() << '\n';
        }
    }

    if (cfg_.tail_extraction.enable) {
        std::ofstream tail_sums_out(tail_sums_path);
        if (!tail_sums_out.is_open()) {
            throw std::runtime_error("Could not open tail summed waveform CSV file.");
        }

        write_tail_sum_header(tail_sums_out, tail_n_bins);
        write_tail_sums(tail_sums_out, tail_sums, tail_n_bins);
    }

    std::cout << "Finished shard " << cfg_.shard_index << " / " << cfg_.num_shards << ".\n"
              << "  Runs processed: " << n_runs_processed << '\n'
              << "  Runs failed:    " << n_runs_failed << '\n'
              << "  Summary:        " << summary_path << '\n'
              << "  Pulses:         " << pulses_path << '\n'
              << "  Windows:        " << windows_path << '\n'
              << "  Tail pulses:    " << tail_pulses_path << '\n'
              << "  Tail sums:      " << tail_sums_path << '\n';
}

} // namespace ucn::app