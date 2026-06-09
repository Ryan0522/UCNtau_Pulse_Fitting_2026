#include "ucn/io/AnalysisConfig.hpp"
#include "ucn/io/RootRunLoader.hpp"
#include "ucn/inference/CoincidenceFitter.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <array>

namespace fs = std::filesystem;

namespace {

constexpr double kLastSignalSeconds = 35.0;
constexpr double kMinCoincidenceSeparationUs = 1000.0; // 1 ms
constexpr double kPeBinWidth = 1.0;

using SumKey = std::tuple<int, std::string, std::string, int>; // segment, region, pe_bin

struct SumRow {
    long long n_events = 0;
    long long n_raw_hits = 0;
    double amp_sum = 0.0;
    std::vector<std::uint64_t> counts;
};

using SumMap = std::map<SumKey, SumRow>;

bool run_is_allowed(const ucn::io::AnalysisConfig& cfg, int run_number) {
    if (!cfg.restrict_to_good_runs || cfg.good_runs.empty()) return true;
    return cfg.good_runs.count(std::to_string(run_number)) > 0;
}

bool run_is_production(const ucn::io::AnalysisConfig& cfg, int run_number) {
    if (cfg.runinfo_json.empty()) return true;

    const std::string run_key = std::to_string(run_number);
    if (!cfg.runinfo_json.contains(run_key)) return false;

    const auto& r = cfg.runinfo_json.at(run_key);
    if (!r.contains("run_type")) return false;

    return r.at("run_type").get<std::string>() == "production";
}

double get_runinfo_number_or(const ucn::io::AnalysisConfig& cfg,
                             int run_number,
                             const std::vector<std::string>& keys,
                             double fallback) {
    if (cfg.runinfo_json.empty()) return fallback;

    const std::string run_key = std::to_string(run_number);
    if (!cfg.runinfo_json.contains(run_key)) return fallback;

    const auto& r = cfg.runinfo_json.at(run_key);
    for (const std::string& key : keys) {
        if (r.contains(key) && r.at(key).is_number()) {
            return r.at(key).get<double>();
        }
    }

    return fallback;
}

std::vector<int> build_selected_run_list(const ucn::io::AnalysisConfig& cfg) {
    std::vector<int> runs;

    for (int run = cfg.start_run; run <= cfg.end_run; ++run) {
        if (!run_is_allowed(cfg, run)) continue;
        if (!run_is_production(cfg, run)) continue;
        runs.push_back(run);
    }

    return runs;
}

std::vector<int> select_shard(const std::vector<int>& all_runs,
                              int shard_index,
                              int num_shards) {
    if (num_shards <= 0) {
        throw std::runtime_error("num_shards must be positive.");
    }
    if (shard_index < 0 || shard_index >= num_shards) {
        throw std::runtime_error("shard_index must satisfy 0 <= shard_index < num_shards.");
    }

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

fs::path make_output_dir(const ucn::io::AnalysisConfig& cfg,
                         const std::string& optional_out_dir) {
    fs::path out_dir;

    if (!optional_out_dir.empty()) {
        out_dir = fs::path(optional_out_dir);
    } else {
        out_dir = fs::path(cfg.output_folder);

        if (!cfg.array_output_subdir.empty()) {
            out_dir /= cfg.array_output_subdir + "_coinc_pulse_shape";
        } else {
            out_dir /= "coinc_pulse_shape";
        }

        if (cfg.num_shards > 1) {
            out_dir /= shard_name(cfg.shard_index);
        }
    }

    fs::create_directories(out_dir);
    return out_dir;
}

bool event_is_in_signal(const ucn::CoincidenceEvent& ev,
                        const ucn::io::RunWindow& w) {
    return ev.start_time_us >= w.signal_start_us &&
           ev.start_time_us <  w.signal_end_us;
}

bool event_is_in_last_signal_seconds(const ucn::CoincidenceEvent& ev,
                                     const ucn::io::RunWindow& w,
                                     double last_seconds) {
    const double late_start_us =
        std::max(w.signal_start_us, w.signal_end_us - last_seconds * 1.0e6);

    return ev.start_time_us >= late_start_us &&
           ev.start_time_us <  w.signal_end_us;
}

bool event_passes_threshold(const ucn::CoincidenceEvent& ev,
                            const ucn::CoincidenceSettings& settings) {
    if (settings.threshold_on_pileup_corrected) {
        return ev.passes_pileup_threshold;
    }
    return ev.passes_raw_threshold;
}

std::vector<ucn::CoincidenceEvent> signal_events_sorted(
    const std::vector<ucn::CoincidenceEvent>& events,
    const ucn::io::RunWindow& w
) {
    std::vector<ucn::CoincidenceEvent> out;
    out.reserve(events.size());

    for (const auto& ev : events) {
        if (event_is_in_signal(ev, w)) {
            out.push_back(ev);
        }
    }

    std::sort(out.begin(), out.end(),
              [](const auto& a, const auto& b) {
                  return a.start_time_us < b.start_time_us;
              });

    return out;
}

bool is_isolated_from_neighboring_coincidences(
    const std::vector<ucn::CoincidenceEvent>& signal_events,
    std::size_t i,
    double min_sep_us,
    double& prev_dt_us,
    double& next_dt_us
) {
    prev_dt_us = std::numeric_limits<double>::infinity();
    next_dt_us = std::numeric_limits<double>::infinity();

    const double t = signal_events[i].start_time_us;

    if (i > 0) {
        prev_dt_us = t - signal_events[i - 1].start_time_us;
    }

    if (i + 1 < signal_events.size()) {
        next_dt_us = signal_events[i + 1].start_time_us - t;
    }

    return prev_dt_us >= min_sep_us && next_dt_us >= min_sep_us;
}

void write_event_header(std::ofstream& out) {
    out << "event_id,run,segment,hold_time_s,region,"
           "coinc_index_in_signal,time_us,n_pe,n_pileup,length_us,"
           "free_pe_interval_us,prev_dt_us,next_dt_us,"
           "bin_width_us,window_start_us,window_end_us,n_raw_hits\n";
}

void write_sum_header(std::ofstream& out, int n_bins) {
    out << "hold_time_s,segment,region,pe_bin,pe_low,pe_high,"
           "n_events,n_raw_hits,amp_sum,amp_mean";

    const char old_fill = out.fill();

    for (int b = 0; b < n_bins; ++b) {
        out << ",b" << std::setw(4) << std::setfill('0') << b;
    }

    out << std::setfill(old_fill) << '\n';
}

void write_sums(std::ofstream& out,
                const SumMap& sums,
                int n_bins) {
    for (const auto& kv : sums) {
        const auto& key = kv.first;
        const SumRow& row = kv.second;

        const int hold_time_s = std::get<0>(key);
        const std::string& segment = std::get<1>(key);
        const std::string& region = std::get<2>(key);
        const int pe_bin = std::get<3>(key);

        double pe_low = -1.0;
        double pe_high = -1.0;
        if (pe_bin >= 0) {
            pe_low = static_cast<double>(pe_bin) * kPeBinWidth;
            pe_high = static_cast<double>(pe_bin + 1) * kPeBinWidth;
        }

        const double amp_mean =
            row.n_events > 0 ? row.amp_sum / static_cast<double>(row.n_events) : 0.0;

        out << hold_time_s << ','
            << segment << ','
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
                b < static_cast<int>(row.counts.size())
                    ? row.counts[static_cast<std::size_t>(b)]
                    : 0;
            out << ',' << c;
        }

        out << '\n';
    }
}

void accumulate_into_key(SumMap& sums,
                         int hold_time_s,
                         const std::string& segment,
                         const std::string& region,
                         int pe_bin,
                         int n_bins,
                         int n_raw_hits_this_event,
                         int n_pe,
                         const std::vector<int>& local_counts) {
    SumKey key{hold_time_s, segment, region, pe_bin};
    SumRow& row = sums[key];

    if (row.counts.empty()) {
        row.counts.assign(static_cast<std::size_t>(n_bins), 0);
    } else if (static_cast<int>(row.counts.size()) != n_bins) {
        throw std::runtime_error("Sum row has inconsistent n_bins.");
    }

    for (int b = 0; b < n_bins; ++b) {
        row.counts[static_cast<std::size_t>(b)] +=
            static_cast<std::uint64_t>(local_counts[static_cast<std::size_t>(b)]);
    }

    row.n_events += 1;
    row.n_raw_hits += n_raw_hits_this_event;
    row.amp_sum += static_cast<double>(n_pe);
}

void accumulate_coincidence_waveform(std::ofstream& event_out,
                                     SumMap& sums,
                                     const std::string& run_id,
                                     const std::string& segment,
                                     double hold_time_s,
                                     const std::vector<ucn::Hit>& raw_hits,
                                     const ucn::CoincidenceEvent& ev,
                                     std::size_t coinc_index_in_signal,
                                     double prev_dt_us,
                                     double next_dt_us,
                                     double bin_width_us,
                                     double window_us,
                                     double pretrigger_us) {
    const int n_bins = static_cast<int>(std::ceil(window_us / bin_width_us));
    if (n_bins <= 0) return;

    const double window_start_us = ev.start_time_us - pretrigger_us;
    const double window_end_us =
        window_start_us + static_cast<double>(n_bins) * bin_width_us;

    std::vector<int> local_counts(static_cast<std::size_t>(n_bins), 0);

    auto first = std::lower_bound(
        raw_hits.begin(),
        raw_hits.end(),
        window_start_us,
        [](const ucn::Hit& h, double t) {
            return h.time_us < t;
        }
    );

    int n_raw_hits_this_event = 0;
    for (auto it = first; it != raw_hits.end() && it->time_us < window_end_us; ++it) {
        const int bin = static_cast<int>((it->time_us - window_start_us) / bin_width_us);
        if (bin >= 0 && bin < n_bins) {
            local_counts[static_cast<std::size_t>(bin)] += 1;
            ++n_raw_hits_this_event;
        }
    }

    const int pe_bin = static_cast<int>(std::floor(static_cast<double>(ev.n_pe) / kPeBinWidth));
    if (pe_bin < 0) return;

    const int hold_time_bin_s = static_cast<int>(std::llround(hold_time_s));
    const std::string region = "signal_last35s";

    // PE-binned average, similar to existing tail sums.
    accumulate_into_key(
        sums,
        hold_time_bin_s,
        segment,
        region,
        pe_bin,
        n_bins,
        n_raw_hits_this_event,
        ev.n_pe,
        local_counts
    );

    // Also write an all-PE average per segment.
    accumulate_into_key(
        sums,
        hold_time_bin_s,
        segment,
        region,
        -1,
        n_bins,
        n_raw_hits_this_event,
        ev.n_pe,
        local_counts
    );

    std::ostringstream id;
    id << run_id << '_' << segment << "_coinc_"
       << coinc_index_in_signal << '_'
       << static_cast<long long>(std::llround(ev.start_time_us * 1000.0));

    event_out << id.str() << ','
              << run_id << ','
              << segment << ','
              << std::setprecision(17) << hold_time_s << ','
              << region << ','
              << coinc_index_in_signal << ','
              << ev.start_time_us << ','
              << ev.n_pe << ','
              << ev.n_pileup << ','
              << ev.length_us << ','
              << ev.free_pe_interval_us << ','
              << prev_dt_us << ','
              << next_dt_us << ','
              << bin_width_us << ','
              << window_start_us << ','
              << window_end_us << ','
              << n_raw_hits_this_event << '\n';
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr
            << "Usage: extract_coincidence_pulse_shape <config.json> [output_dir]\n\n"
            << "Uses coincidence timestamps, keeps only events in the last 35 s of the signal window,\n"
            << "requires neighboring coincidence timestamps to be at least 1 ms away,\n"
            << "and accumulates raw-hit pulse shapes like tail extraction.\n";
        return 1;
    }

    try {
        const std::string config_path = argv[1];
        const std::string optional_out_dir = argc == 3 ? argv[2] : "";

        const ucn::io::AnalysisConfig cfg = ucn::io::load_analysis_config(config_path);

        double bin_width_us = 0.5;
        double window_us = 1000.0;
        double pretrigger_us = 10.0;

        const int n_bins = static_cast<int>(std::ceil(window_us / bin_width_us));

        const fs::path out_dir = make_output_dir(cfg, optional_out_dir);
        const fs::path event_path = out_dir / "coinc_pulse_shape_events.csv";
        const fs::path sum_path = out_dir / "coinc_pulse_shape_sums_by_hold_and_pe.csv";
        const fs::path meta_path = out_dir / "coinc_pulse_shape_metadata.json";

        std::ofstream event_out(event_path);
        if (!event_out.is_open()) {
            throw std::runtime_error("Could not open event output CSV.");
        }
        write_event_header(event_out);

        std::ofstream meta_out(meta_path);
        if (!meta_out.is_open()) {
            throw std::runtime_error("Could not open metadata output JSON.");
        }

        const std::vector<int> all_runs = build_selected_run_list(cfg);
        const std::vector<int> my_runs = select_shard(
            all_runs,
            cfg.shard_index,
            cfg.num_shards
        );

        meta_out << "{\n"
                 << "  \"config_path\": \"" << config_path << "\",\n"
                 << "  \"last_signal_seconds\": " << kLastSignalSeconds << ",\n"
                 << "  \"min_coincidence_separation_us\": " << kMinCoincidenceSeparationUs << ",\n"
                 << "  \"bin_width_us\": " << bin_width_us << ",\n"
                 << "  \"window_us\": " << window_us << ",\n"
                 << "  \"pretrigger_us\": " << pretrigger_us << ",\n"
                 << "  \"n_bins\": " << n_bins << ",\n"
                 << "  \"start_run\": " << cfg.start_run << ",\n"
                 << "  \"end_run\": " << cfg.end_run << ",\n"
                 << "  \"restrict_to_good_runs\": "
                 << (cfg.restrict_to_good_runs ? "true" : "false") << ",\n"
                 << "  \"shard_index\": " << cfg.shard_index << ",\n"
                 << "  \"num_shards\": " << cfg.num_shards << ",\n"
                 << "  \"n_selected_runs_total\": " << all_runs.size() << ",\n"
                 << "  \"n_selected_runs_this_shard\": " << my_runs.size() << "\n"
                 << "}\n";

        ucn::CoincidenceFitter coincidence_fitter(cfg.coincidence_settings);

        SumMap sums;

        long long n_runs_processed = 0;
        long long n_runs_failed = 0;
        long long n_segments_processed = 0;
        long long n_signal_coincidences = 0;
        long long n_late_signal_coincidences = 0;
        long long n_selected_events = 0;
        long long n_rejected_threshold = 0;
        long long n_rejected_isolation = 0;

        std::cout << "Selected " << all_runs.size() << " production/good runs total.\n"
                  << "Shard " << cfg.shard_index << " / " << cfg.num_shards
                  << " will process " << my_runs.size() << " runs.\n"
                  << "Output directory: " << out_dir << '\n'
                  << "bin_width_us = " << bin_width_us
                  << ", window_us = " << window_us
                  << ", pretrigger_us = " << pretrigger_us
                  << ", n_bins = " << n_bins << '\n';

        for (int run : my_runs) {
            try {
                const ucn::io::LoadedRun loaded = ucn::io::load_root_run(cfg, run);
                const ucn::io::RunWindow window = ucn::io::resolve_run_window(cfg, run);

                const double hold_time_s = get_runinfo_number_or(
                    cfg,
                    run,
                    {"hold_time", "Holding Time", "holding_time"},
                    -1.0
                );

                std::cout << "Run " << loaded.run_id
                          << " loaded. Segments = " << loaded.segments.size()
                          << ", hold_time_s = " << hold_time_s << '\n';

                for (const auto& seg : loaded.segments) {
                    std::vector<ucn::Hit> raw_hits = seg.hits;
                    std::sort(raw_hits.begin(), raw_hits.end(),
                              [](const auto& a, const auto& b) {
                                  return a.time_us < b.time_us;
                              });

                    const ucn::CoincidenceFitResult result =
                        coincidence_fitter.find(raw_hits);

                    const std::vector<ucn::CoincidenceEvent> signal_events =
                        signal_events_sorted(result.events, window);

                    n_signal_coincidences += static_cast<long long>(signal_events.size());

                    for (std::size_t i = 0; i < signal_events.size(); ++i) {
                        const auto& ev = signal_events[i];

                        if (!event_is_in_last_signal_seconds(ev, window, kLastSignalSeconds)) {
                            continue;
                        }
                        ++n_late_signal_coincidences;

                        if (!event_passes_threshold(ev, cfg.coincidence_settings)) {
                            ++n_rejected_threshold;
                            continue;
                        }

                        double prev_dt_us = 0.0;
                        double next_dt_us = 0.0;
                        if (!is_isolated_from_neighboring_coincidences(
                                signal_events,
                                i,
                                kMinCoincidenceSeparationUs,
                                prev_dt_us,
                                next_dt_us)) {
                            ++n_rejected_isolation;
                            continue;
                        }

                        accumulate_coincidence_waveform(
                            event_out,
                            sums,
                            loaded.run_id,
                            seg.segment_name,
                            hold_time_s,
                            raw_hits,
                            ev,
                            i,
                            prev_dt_us,
                            next_dt_us,
                            bin_width_us,
                            window_us,
                            pretrigger_us
                        );

                        ++n_selected_events;
                    }

                    ++n_segments_processed;
                    event_out.flush();
                }

                ++n_runs_processed;
            } catch (const std::exception& e) {
                ++n_runs_failed;
                std::cerr << "Skipping run " << run << " because: " << e.what() << '\n';
            }
        }

        std::ofstream sum_out(sum_path);
        if (!sum_out.is_open()) {
            throw std::runtime_error("Could not open sum output CSV.");
        }

        write_sum_header(sum_out, n_bins);
        write_sums(sum_out, sums, n_bins);
        sum_out.close();

        std::cout << "\nCoincidence pulse-shape extraction complete.\n"
                  << "Runs processed: " << n_runs_processed << '\n'
                  << "Runs failed: " << n_runs_failed << '\n'
                  << "Segments processed: " << n_segments_processed << '\n'
                  << "Signal coincidences: " << n_signal_coincidences << '\n'
                  << "Late signal coincidences: " << n_late_signal_coincidences << '\n'
                  << "Rejected by threshold: " << n_rejected_threshold << '\n'
                  << "Rejected by 1 ms isolation: " << n_rejected_isolation << '\n'
                  << "Selected events: " << n_selected_events << '\n'
                  << "Wrote: " << event_path << '\n'
                  << "Wrote: " << sum_path << '\n'
                  << "Wrote: " << meta_path << '\n';

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << '\n';
        return 2;
    }

    return 0;
}