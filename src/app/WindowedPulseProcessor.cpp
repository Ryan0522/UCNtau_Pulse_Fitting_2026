#include "ucn/app/WindowedPulseProcessor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>
#include <memory>

namespace ucn {
namespace {

struct LocalBgEstimate {
    double rate_hz = 0.0;
    double free_pe_interval_us = 0.0;
    int n_free = 0;
    bool valid = false;
};

struct RunningFreePeBg {
    double last_free_time_us = 0.0;
    double free_pe_interval_us = 0.0;
    std::size_t n_free = 0;
    double fallback_rate_hz = 0.0;
    explicit RunningFreePeBg(double initial_rate_hz = 0.0)
        : fallback_rate_hz(std::max(0.0, initial_rate_hz)) {}

    void update(double free_time_us) {
        if (n_free == 0) {
            last_free_time_us = free_time_us;
            n_free = 1;
            return;
        }
        const double dt_us = free_time_us - last_free_time_us;
        last_free_time_us = free_time_us;

        if (dt_us <= 0.0) {
            ++n_free;
            return;
        }

        if (n_free == 1) {
            free_pe_interval_us = dt_us;
        } else if (n_free < 100) {
            const double nf = static_cast<double>(n_free);
            free_pe_interval_us = 
                ((nf - 1.0) / nf) * free_pe_interval_us
                + (1.0 / nf) * dt_us;
        } else {
            free_pe_interval_us = 
                0.99 * free_pe_interval_us + 0.01 * dt_us;
        }
        ++n_free;
    }

    bool ready() const {
        return free_pe_interval_us > 0.0;
    }

    double rate_hz() const {
        if (ready()) {
            return 1.0e6 / free_pe_interval_us;
        }
        return fallback_rate_hz;
    }

    LocalBgEstimate snapshot() const {
        LocalBgEstimate out;
        out.rate_hz = rate_hz();
        out.free_pe_interval_us = free_pe_interval_us;
        out.n_free = static_cast<int>(
            std::min<std::size_t>(
                n_free,
                static_cast<std::size_t>(std::numeric_limits<int>::max())
            )
        );
        out.valid = ready();
        return out;
    }
};

void update_free_pe_bg_from_hit_range(
    const std::vector<ucn::Hit>& hits,
    int begin_index, int end_index,
    RunningFreePeBg& bg
) {
    const int n = static_cast<int>(hits.size());
    begin_index = std::max(0, begin_index);
    end_index = std::min(end_index, n);

    for (int k = begin_index; k < end_index; ++k) {
        bg.update(hits[static_cast<std::size_t>(k)].time_us);
    }
}

double sum_vector(const std::vector<double>& xs) {
    return std::accumulate(xs.begin(), xs.end(), 0.0);
}

double compute_log_likelihood(const std::vector<double>& counts,
                              const std::vector<double>& expected) {
    double log_l = 0.0;
    for (std::size_t i = 0; i < counts.size(); ++i) {
        double mu = expected[i];
        if (mu < 1.0e-12) {
            mu = 1.0e-12;
        }
        log_l += counts[i] * std::log(mu) - mu - std::lgamma(counts[i] + 1.0);
    }
    return log_l;
}

std::vector<debug::TruthPulse> select_truth_in_window(
    const std::vector<debug::TruthPulse>* truth,
    double start_us,
    double model_end_us
) {
    std::vector<debug::TruthPulse> out;
    if (!truth) return out;

    for (const auto& p : *truth) {
        if (p.time_us >= start_us && p.time_us < model_end_us) {
            out.push_back(p);
        }
    }

    return out;
}

debug::DebugCaseType classify_debug_case(
    int n_truth, int n_seeds
) {
    if (n_truth == 1 && n_seeds == 1) {
        return debug::DebugCaseType::SingleNeutronSingleSeed;
    }

    if (n_truth == 1 && n_seeds > 1) {
        return debug::DebugCaseType::SingleNeutronMultiSeed;
    }

    if (n_truth >= 2) {
        return debug::DebugCaseType::MultiNeutron;
    }

    return debug::DebugCaseType::Unknown;
}

debug::DebugCaseType classify_observed_debug_case(
    int n_seeds,
    int n_fit_pulses
) {
    if (n_seeds == 1 && n_fit_pulses == 1) {
        return debug::DebugCaseType::ObservedOneSeedOneFit;
    }

    if (n_seeds == 1 && n_fit_pulses > 1) {
        return debug::DebugCaseType::ObservedOneSeedMultiFit;
    }

    if (n_seeds > 1) {
        return debug::DebugCaseType::ObservedMultiSeed;
    }

    return debug::DebugCaseType::Unknown;
}

std::string classify_seed_fit_topology(int n_seeds, int n_fit_pulses) {
    if (n_seeds == 1 && n_fit_pulses == 1) {
        return "single_seed_single_fit";
    }
    if (n_seeds == 1 && n_fit_pulses > 1) {
        return "single_seed_multi_fit";
    }
    if (n_seeds > 1 && n_fit_pulses == 1) {
        return "multi_seed_single_fit";
    }
    if (n_seeds > 1 && n_fit_pulses > 1) {
        return "multi_seed_multi_fit";
    }
    return "unknown";
}

int pulse_rank_by_time(const std::vector<PulseCandidate>& pulses, std::size_t idx) {
    if (idx >= pulses.size()) return -1;

    int rank = 0;
    const double t = pulses[idx].time_us;

    for (std::size_t j = 0; j < pulses.size(); ++j) {
        if (j == idx) continue;
        if (pulses[j].time_us < t) {
            ++rank;
        }
    }

    return rank;
}

double nearest_fit_neighbor_dt_us(const std::vector<PulseCandidate>& pulses, std::size_t idx) {
    if (idx >= pulses.size() || pulses.size() < 2) {
        return -1.0;
    }

    double best = std::numeric_limits<double>::infinity();
    const double t = pulses[idx].time_us;

    for (std::size_t j = 0; j < pulses.size(); ++j) {
        if (j == idx) continue;
        const double dt = std::abs(pulses[j].time_us - t);
        if (dt < best) {
            best = dt;
        }
    }

    return std::isfinite(best) ? best : -1.0;
}

} // namespace

WindowedPulseProcessor::WindowedPulseProcessor(const PulseTemplate& pulse_template,
                                               const GreedyLRTFitter& fitter)
    : pulse_template_(pulse_template),
      fitter_(fitter) {
}

void WindowedPulseProcessor::set_debug_writer(
    std::shared_ptr<debug::DebugCsvWriter> writer
) {
    debug_writer_ = std::move(writer);
}

std::vector<Hit> WindowedPulseProcessor::select_hits(const std::vector<Hit>& hits,
                                                     double start_us,
                                                     double end_us) const {
    std::vector<Hit> selected;
    for (const Hit& hit : hits) {
        if (hit.time_us >= start_us && hit.time_us < end_us) {
            selected.push_back(hit);
        }
    }
    std::sort(selected.begin(), selected.end(), [](const Hit& a, const Hit& b) {
        return a.time_us < b.time_us;
    });
    return selected;
}

bool WindowedPulseProcessor::build_window_histogram(
    const std::vector<Hit>& hits,
    int start_index,
    double stream_start_us,
    double bin_width_us,
    const RegionSettings& region_settings,
    int& next_index,
    double& start_time_us,
    double& end_time_us,
    double& model_end_time_us,
    Histogram& histogram
) const {
    int n_hits = static_cast<int>(hits.size());
    if (start_index >= n_hits) {
        next_index = start_index;
        return false;
    }

    int seed_index = -1;

    for (int i = start_index; i < n_hits; ++i) {
        const double t0 = hits[i].time_us;
        const int ch0 = hits[i].channel;

        bool armed = false;
        int total = 1;

        for (int k = i + 1; k < n_hits; ++k) {
            const double dt = hits[k].time_us - t0;
            if (dt > region_settings.coincidence_window_us) break;
            if (hits[k].channel != ch0) armed = true;
            ++total;
        }

        if (armed && total >= region_settings.coincidence_min_hits) {
            seed_index = i;
            break;
        }
    }

    if (seed_index < 0) {
        next_index = n_hits;
        return false;
    }

    const double seed_time_us = hits[seed_index].time_us;

    int recovered_start_index = seed_index;
    if (region_settings.recover_preseed_pile) {
        while (recovered_start_index > start_index) {
            const double previous_time_us = hits[recovered_start_index - 1].time_us;
            const double current_time_us = hits[recovered_start_index].time_us;
            const double gap_us = current_time_us - previous_time_us;

            if (gap_us >= region_settings.min_gap_us) break;

            const double lookback_us = seed_time_us - previous_time_us;
            if (region_settings.max_preseed_lookback_us > 0.0 && lookback_us > region_settings.max_preseed_lookback_us) break;
            
            --recovered_start_index;
        }
    }

    const double recovered_start_time_us = hits[recovered_start_index].time_us;

    double left_partition_us = stream_start_us;
    if (start_index > 0) {
        const double previous_owned_hit_us = hits[start_index - 1].time_us;
        const double first_unconsumed_hit_us = hts[start_index].time_us;
        left_partition_us = std::max(stream_start_us, 0.5 * (previous_owned_hit_u + first_unconsumed_hit_us));
    }

    int j = seed_index + 1;
    double last_hit_time_us = seed_time_us;

    if (region_settings.window_mode == "fixed_seed_window") {
        const double prepad_us = std::max(0.0, region_settings.fixed_seed_pretrigger_us);
        const double fixed_window_us = std::max(bin_width_us, region_settings.fixed_seed_window_us);
    
        double current_window_end_us = seed_time_us + fixed_window_us;
        while (j < n_hits && hits[j].time_us < current_window_end_us) {
            const double t0 = hits[j].time_us;
            const int ch0 = hits[j].channel;

            bool armed = false;
            int total = 1;

            for (int k = j + 1; k < n_hits; ++k) {
                const double dt = hits[k].time_us - t0;
                if (dt > region_settings.coincidence_window_us) {
                    break;
                }
                if (hits[k].channel != ch0) {
                    armed = true;
                }
                ++total;
            }
            if (armed && total >= region_settings.coincidence_min_hits) {
                current_window_end_us = t0 + fixed_window_us;
            }
            ++j;
        }

        last_hit_time_us = hits[j - 1].time_us;
        double right_partition_us = std::numeric_limits<double>::infinity():
        
        if (j < n_hits)
            right_partition_us = 0.5 * (hits[j - 1].time_us + hits[j].time_us);

        start_time_us = std::max(left_partition_us, recovered_start_time_us - prepad_us);
        end_time_us = std::min(current_window_end_us, right_partition_us);
        model_end_time_us = current_window_end_us;
        next_index = j;

    } else {
        while (j < n_hits) {
            const double gap = hits[j].time_us - hits[j - 1].time_us;
            if (gap > region_settings.min_gap_us) break;
            ++j;
        }

        last_hit_time_us = hits[j - 1].time_us;

        const double prepad_us =  std::max(0.0, region_settings.fit_start_padding_us);
        const double postpad_us = std::max(0.0, region_settings.fit_end_padding_us);

        double right_partition_us = std::numeric_limits<double>::infinity();
        if (j < n_hits) 
            right_partition_us = 0.5 * (hits[j - 1].time_us + hits[j].time_us);

        start_time_us = std::max(left_partition_us, recovered_start_time_us - prepad_us);
        end_time_us = last_hit_time_us;
        model_end_time_us = std::min(last_hit_time_us + postpad_us, right_partition_us);

        next_index = j;
    }

    start_time_us = std::max(start_tims_us, stream_start_us);
    if (!std::isfinite(start_time_us) || !std::isfinite(model_end_time_us) || model_end_time_us <= start_time_us) return false;
    end_time_us = std::min(end_time_us, model_end_time_us);

    const double model_width_us = model_end_time_us - start_time_us;
    if (model_width_us < bin_width_us) {
        return false;
    }
    
    const int n_bins = static_cast<int>(std::ceil(model_width_us / bin_width_us));
    if (n_bins <= 0) {
        return false;
    }

    histogram.bin_edges_us.clear();
    histogram.bin_edges_us.reserve(static_cast<std::size_t>(n_bins + 1));
    histogram.counts.assign(static_cast<std::size_t>(n_bins), 0.0);
    
    for (int b = 0; b <= n_bins; ++b) {
        histogram.bin_edges_us.push_back(start_time_us + static_cast<double>(b) * bin_width_us);
    }

    auto first_it = std::lower_bound(
        hits.begin() + start_index,
        hits.begin() + j,
        start_time_us,
        [](const Hit& h, double t) {
            return h.time_us < t;
        }
    );

    for (auto it = first_it; it != hits.begin() + j; ++it) {
        if (it->time_us < start_time_us) continue;
        if (it->time_us >= model_end_time_us) break;

        const double dt_us = it->time_us - start_time_us;
        const int bin = static_cast<int>(dt_us / bin_width_us);

        if (bin >= 0 && bin < n_bins) {
            histogram.counts[static_cast<std::size_t>(bin)] += 1.0;
        }
    }

    return true;
}

std::vector<double> WindowedPulseProcessor::find_coincidence_seeds(const std::vector<Hit>& hits,
                                                                   double window_start_us,
                                                                   double window_end_us,
                                                                   double bin_width_us,
                                                                   const RegionSettings& region_settings) const {
    std::vector<double> seeds;
    double next_allowed_seed_time_us = -1.0e99;

    for (std::size_t i = 0; i < hits.size(); ++i) {
        double t0 = hits[i].time_us;
        if (t0 < window_start_us || t0 >= window_end_us) {
            continue;
        }
        if (t0 < next_allowed_seed_time_us) {
            continue;
        }

        int channel0 = hits[i].channel;
        bool armed = false;
        int total = 1;
        for (std::size_t j = i + 1; j < hits.size(); ++j) {
            double tj = hits[j].time_us;
            if (tj >= window_end_us) {
                break;
            }
            if (std::abs(tj - t0) > region_settings.coincidence_window_us) {
                break;
            }
            if (hits[j].channel != channel0) {
                armed = true;
            }
            ++total;
        }

        if (armed && total >= region_settings.coincidence_min_hits) {
            seeds.push_back(t0);
            next_allowed_seed_time_us = t0 + region_settings.seed_veto_window_us;
        }
    }

    return seeds;
}

std::vector<double> WindowedPulseProcessor::build_carry_expected(const std::vector<PulseCandidate>& carry_pulses,
                                                                 const Histogram& histogram) const {
    std::vector<double> carry(histogram.counts.size(), 0.0);
    if (histogram.bin_edges_us.size() < 2) {
        return carry;
    }

    double window_start_us = histogram.bin_edges_us.front();

    for (const PulseCandidate& pulse : carry_pulses) {
        if (pulse.time_us >= window_start_us) continue;

        std::vector<double> component = pulse_template_.shifted_to_histogram(pulse.time_us, histogram.bin_edges_us);
        if (component.size() != carry.size()) continue;

        double mass = 0.0;
        for (double x : component) mass += x;
        if (mass <= 1.0e-12) continue;
        
        for (std::size_t i = 0; i < carry.size(); ++i) {
            carry[i] += pulse.amplitude_pe * component[i];
        }
    }

    return carry;
}

void WindowedPulseProcessor::fit_stream(
    const std::vector<Hit>& hits,
    const RegionSettings& region_settings,
    const FitSettings& fit_settings,
    double background_rate_hz,
    double stream_start_us,
    std::vector<TaggedPulse>& output_pulses,
    std::vector<WindowSummary>& output_summaries,
    const std::vector<debug::TruthPulse>* truth_pulses,
    int chunk_index,
    int global_window_offset,
    const std::string& region_name,
    double hold_time_s
) const {
    std::vector<PulseCandidate> carry_pulses;

    int i = 0;
    int window_index = 0;
    double running_background_rate_hz = background_rate_hz;
    RunningFreePeBg free_pe_bg(background_rate_hz);
    double last_model_end_time_us = stream_start_us;
    
    while (i < static_cast<int>(hits.size())) {
        Histogram coarse_histogram;
        double start_time_us = 0.0;
        double end_time_us = 0.0;
        double model_end_time_us = 0.0;
        int next_index = i;

        bool ok = build_window_histogram(hits,
                                         i,
                                         stream_start_us,
                                         region_settings.coarse_bin_width_us,
                                         region_settings,
                                         next_index,
                                         start_time_us,
                                         end_time_us,
                                         model_end_time_us,
                                         coarse_histogram);
        if (!ok) {
            i = std::max(next_index, i + 1);
            continue;
        }

        Histogram histogram = coarse_histogram;
        double bin_width_us = region_settings.coarse_bin_width_us;
        bool uses_fine_bins = false;

        if (histogram.counts.size() < 2) {
            Histogram fine_histogram;
            ok = build_window_histogram(hits,
                                        i,
                                        region_settings.fine_bin_width_us,
                                        region_settings,
                                        next_index,
                                        start_time_us,
                                        end_time_us,
                                        model_end_time_us,
                                        fine_histogram);
            if (!ok) {
                i = std::max(next_index, i + 1);
                continue;
            }
            histogram = fine_histogram;
            bin_width_us = region_settings.fine_bin_width_us;
            uses_fine_bins = true;
        }

        std::vector<double> fixed_expected = build_carry_expected(carry_pulses, histogram);
        // std::vector<double> fixed_expected(histogram.counts.size(), 0.0);
        std::vector<double> seeds = find_coincidence_seeds(
            hits, start_time_us, end_time_us, bin_width_us, region_settings
        );
        
        if (seeds.empty()) {
            if (region_settings.enable_local_background && region_name == "signal") {
                update_free_pe_bg_from_hit_range(hits, i, next_index, free_pe_bg);
            }
            
            last_model_end_time_us = std::max(last_model_end_time_us, model_end_time_us);
            i = next_index;
            ++window_index;
            continue;
        }

        const bool has_truth = (truth_pulses != nullptr);
        const bool allow_debug_this_region = (region_name == "signal");

        std::vector<debug::TruthPulse> truth_here;
        debug::DebugCaseType prefit_case_type = debug::DebugCaseType::Unknown;

        bool mc_capture_debug = false;
        bool root_candidate_debug = false;

        std::string prefit_case_id;

        std::unique_ptr<debug::BufferedDebugSink> buffered_sink;
        debug::DebugSink* sink_for_fit = nullptr;

        if (allow_debug_this_region && debug_writer_ && debug_writer_->enabled()) {
            if (has_truth) {
                truth_here = select_truth_in_window(
                    truth_pulses, start_time_us, model_end_time_us
                ); 

                prefit_case_type = classify_debug_case(
                    static_cast<int>(truth_here.size()), static_cast<int>(seeds.size())
                );

                if (prefit_case_type != debug::DebugCaseType::Unknown &&
                    debug_writer_->can_capture(prefit_case_type, hold_time_s)) {
                    mc_capture_debug = true;
                    prefit_case_id = debug_writer_->next_case_id(prefit_case_type, hold_time_s);
                    sink_for_fit = debug_writer_.get();
                }
            } else if (debug_writer_->can_capture_any_observed(hold_time_s)) {
                root_candidate_debug = true;
                buffered_sink = std::make_unique<debug::BufferedDebugSink>();
                sink_for_fit = buffered_sink.get();
            }
        }

        double window_background_rate_hz = background_rate_hz;
        LocalBgEstimate gap_bg;

        if (region_settings.enable_local_background && region_name == "signal") {
            gap_bg = free_pe_bg.snapshot();

            double measured_rate_hz = gap_bg.rate_hz;
            const double max_rate_hz = region_settings.local_bg_max_scale * std::max(1.0, background_rate_hz);
            measured_rate_hz = std::min(measured_rate_hz, max_rate_hz);

            running_background_rate_hz = measured_rate_hz;
            window_background_rate_hz = running_background_rate_hz;
        }

        FitSettings window_fit_settings = fit_settings;
        window_fit_settings.background_per_bin = window_background_rate_hz * bin_width_us * 1.0e-6;
        window_fit_settings.fixed_expected = fixed_expected;

        FitResult fit = fitter_.fit(
            histogram, seeds, window_fit_settings, sink_for_fit, prefit_case_id
        );

        if (fit.pulses.empty()) {
            if (region_settings.enable_local_background && region_name == "signal") {
                update_free_pe_bg_from_hit_range(hits, i, next_index, free_pe_bg);
            }

            last_model_end_time_us = std::max(last_model_end_time_us, model_end_time_us);
            i = next_index;
            ++window_index;
            continue;
        }

        if (mc_capture_debug) {
            debug::WindowDebugContext ctx;
            ctx.case_id = prefit_case_id;
            ctx.case_type = prefit_case_type;
            ctx.region = region_name;
            ctx.chunk_index = chunk_index;
            ctx.local_window_index = window_index;
            ctx.global_window_index = global_window_offset + window_index;
            ctx.start_time_us = start_time_us;
            ctx.end_time_us = end_time_us;
            ctx.model_end_time_us = model_end_time_us;
            ctx.bin_width_us = bin_width_us;
            ctx.seeds = seeds;
            ctx.truth_pulses = truth_here;

            debug_writer_->write_window(
                ctx,
                histogram,
                fit,
                window_fit_settings.fixed_expected,
                window_fit_settings.background_per_bin
            );
        }

        if (!has_truth && root_candidate_debug && buffered_sink) {
            const debug::DebugCaseType observed_case_type =
                classify_observed_debug_case(
                    static_cast<int>(seeds.size()),
                    static_cast<int>(fit.pulses.size())
                );

            if (observed_case_type != debug::DebugCaseType::Unknown &&
                debug_writer_->can_capture(observed_case_type, hold_time_s)) {
                const std::string observed_case_id =
                    debug_writer_->next_case_id(observed_case_type, hold_time_s);

                debug::WindowDebugContext ctx;
                ctx.case_id = observed_case_id;
                ctx.case_type = observed_case_type;
                ctx.region = region_name;
                ctx.hold_time_s = hold_time_s;
                ctx.chunk_index = chunk_index;
                ctx.local_window_index = window_index;
                ctx.global_window_index = global_window_offset + window_index;
                ctx.start_time_us = start_time_us;
                ctx.end_time_us = end_time_us;
                ctx.model_end_time_us = model_end_time_us;
                ctx.bin_width_us = bin_width_us;
                ctx.seeds = seeds;

                debug_writer_->write_window(
                    ctx,
                    histogram,
                    fit,
                    window_fit_settings.fixed_expected,
                    window_fit_settings.background_per_bin
                );

                debug_writer_->write_lrt_trials_for_case(
                    observed_case_id,
                    buffered_sink->trials
                );

                debug_writer_->write_lrt_accepts_for_case(
                    observed_case_id,
                    buffered_sink->accepts
                );
            }
        }

        std::vector<double> full_expected = fit.expected_total;
        if (window_fit_settings.background_per_bin > 0.0) {
            for (double& value : full_expected) {
                value += window_fit_settings.background_per_bin;
            }
        }
        if (!window_fit_settings.fixed_expected.empty()) {
            for (std::size_t j = 0; j < full_expected.size(); ++j) {
                full_expected[j] += window_fit_settings.fixed_expected[j];
            }
        }
        double expected_sum = std::accumulate(full_expected.begin(), full_expected.end(), 0.0);
        double observed_count = std::accumulate(histogram.counts.begin(), histogram.counts.end(), 0.0);

        double fitted_pe_sum = 0.0;
        for (const auto& p : fit.pulses) {
            fitted_pe_sum += p.amplitude_pe;
        }
        const double fit_expected_sum = sum_vector(fit.expected_total);
        const double fixed_expected_sum = sum_vector(window_fit_settings.fixed_expected);
        const double background_expected_sum = window_fit_settings.background_per_bin * static_cast<double>(histogram.counts.size());

        const double template_mass_in_window = fitted_pe_sum > 0.0 ? fit_expected_sum / fitted_pe_sum : 0.0;
        const double pe_per_observed_count = observed_count > 0 ? fitted_pe_sum / static_cast<double>(observed_count) : 0.0;
        const double background_fraction = observed_count > 0 ? background_expected_sum / static_cast<double>(observed_count) : 0.0;
        const double fit_fraction = observed_count > 0 ? fit_expected_sum / static_cast<double>(observed_count) : 0.0;

        WindowSummary summary;
        summary.window_index = window_index;
        summary.start_time_us = start_time_us;
        summary.end_time_us = end_time_us;
        summary.width_us = end_time_us - start_time_us;
        summary.bin_width_us = bin_width_us;
        summary.pulse_count = static_cast<int>(fit.pulses.size());
        summary.seed_count = static_cast<int>(seeds.size());
        summary.observed_count = static_cast<int>(std::llround(observed_count));
        summary.expected_count = expected_sum;
        summary.final_nll = -compute_log_likelihood(histogram.counts, full_expected);
        summary.fitted_pe_sum = fitted_pe_sum;
        summary.fit_expected_sum = fit_expected_sum;
        summary.fixed_expected_sum = fixed_expected_sum;
        summary.background_expected_sum = background_expected_sum;
        summary.template_mass_in_window = template_mass_in_window;
        summary.pe_per_observed_count = pe_per_observed_count;
        summary.background_fraction = background_fraction;
        summary.fit_fraction = fit_fraction;
        summary.local_background_rate_hz = window_background_rate_hz;
        summary.local_background_gap_us = gap_bg.free_pe_interval_us;
        summary.local_background_gap_hits = gap_bg.n_free;
        summary.local_background_updated = gap_bg.valid;
        output_summaries.push_back(summary);
        last_model_end_time_us = std::max(last_model_end_time_us, model_end_time_us);

        double template_support_us = static_cast<double>(pulse_template_.pmf().size()) * pulse_template_.native_bin_width_us();
        std::vector<PulseCandidate> updated_carry;
        for (const PulseCandidate& pulse : carry_pulses) {
            double template_start_us = pulse.time_us;
            if (template_start_us + template_support_us > start_time_us) {
                updated_carry.push_back(pulse);
            }
        }
        carry_pulses.swap(updated_carry);

        const int seed_count_in_window = static_cast<int>(seeds.size());
        const int fit_pulse_count_in_window = static_cast<int>(fit.pulses.size());

        const std::string fit_topology = classify_seed_fit_topology(
            seed_count_in_window,
            fit_pulse_count_in_window
        );

        for (std::size_t ip = 0; ip < fit.pulses.size(); ++ip) {
            const PulseCandidate& pulse = fit.pulses[ip];

            TaggedPulse tagged;
            tagged.time_us = pulse.time_us;
            tagged.amplitude_pe = pulse.amplitude_pe;
            tagged.window_index = window_index;
            tagged.width_us = end_time_us - start_time_us;

            // Deprecated compatibility flag.
            // Do not use this as physical pileup.
            tagged.is_pileup = fit_pulse_count_in_window > 1;
            tagged.uses_fine_bins = uses_fine_bins;

            tagged.seed_count_in_window = seed_count_in_window;
            tagged.fit_pulse_count_in_window = fit_pulse_count_in_window;
            tagged.pulse_rank_in_window = pulse_rank_by_time(fit.pulses, ip);
            tagged.nearest_fit_dt_us = nearest_fit_neighbor_dt_us(fit.pulses, ip);
            tagged.fit_topology = fit_topology;

            output_pulses.push_back(tagged);
            carry_pulses.push_back(pulse);
        }

        i = next_index;
        ++window_index;
    }
}

RegionResult WindowedPulseProcessor::analyze(
    const std::vector<Hit>& hits,
    double background_start_us,
    double background_end_us,
    double signal_start_us,
    double signal_end_us,
    double end_start_us,
    double end_end_us,
    const RegionSettings& region_settings,
    const FitSettings& fit_settings,
    const std::vector<debug::TruthPulse>* truth_pulses,
    int chunk_index,
    int global_window_offset,
    double hold_time_s
) const {
    RegionResult result;

    const double background_duration_us =
        std::max(0.0, background_end_us - background_start_us);

    std::vector<Hit> background_hits;
    if (region_settings.enable_background_fit && background_duration_us > 0.0) {
        background_hits = select_hits(hits, background_start_us, background_end_us);
    }

    std::vector<Hit> signal_hits = select_hits(hits, signal_start_us, signal_end_us);
    std::vector<Hit> end_hits = select_hits(hits, end_start_us, end_end_us);

    double background_rate_hz = 0.0;
    if (!background_hits.empty() && background_duration_us > 0.0) {
        background_rate_hz = static_cast<double>(background_hits.size()) / (background_duration_us * 1.0e-6);
    }

    if (region_settings.enable_background_fit && background_duration_us > 0.0) {
        for (int iter = 0; iter < region_settings.background_iterations; ++iter) {
            std::vector<TaggedPulse> background_pulses;
            std::vector<WindowSummary> background_summaries;
            fit_stream(background_hits,
                       region_settings,
                       fit_settings,
                       background_rate_hz,
                       background_start_us,
                       background_pulses,
                       background_summaries,
                       nullptr, -1, 0, "background", hold_time_s);

            double fitted_pe_sum = 0.0;
            for (const TaggedPulse& pulse : background_pulses) {
                fitted_pe_sum += pulse.amplitude_pe;
            }

            const double duration_s = background_duration_us * 1.0e-6;
            
            double new_background_rate_hz = 0.0;
            if (duration_s > 0.0) {
                new_background_rate_hz = std::max(0.0, (static_cast<double>(background_hits.size()) - fitted_pe_sum) / duration_s);
            }

            double delta = std::abs(new_background_rate_hz - background_rate_hz);
            double tolerance = region_settings.background_tolerance_fraction * std::max(1.0, background_rate_hz);
            
            background_rate_hz = new_background_rate_hz;
            
            result.background_pulses = std::move(background_pulses);
            result.background_window_summaries = std::move(background_summaries);

            if (delta < tolerance) {
                break;
            }
        }
    }

    fit_stream(signal_hits,
               region_settings,
               fit_settings,
               background_rate_hz,
               signal_start_us,
               result.signal_pulses,
               result.signal_window_summaries,
               truth_pulses,
               chunk_index,
               global_window_offset,
               "signal",
               hold_time_s);

    fit_stream(end_hits,
               region_settings,
               fit_settings,
               background_rate_hz,
               end_start_us,
               result.end_pulses,
               result.end_window_summaries,
               nullptr, -1, 0, "end", hold_time_s);

    result.background_rate_hz = background_rate_hz;
    return result;
}

} // namespace ucn
