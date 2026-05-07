#include "ucn/app/WindowedPulseProcessor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace ucn {
namespace {

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

bool WindowedPulseProcessor::build_window_histogram(const std::vector<Hit>& hits,
                                                    int start_index,
                                                    double bin_width_us,
                                                    const RegionSettings& region_settings,
                                                    int& next_index,
                                                    double& start_time_us,
                                                    double& end_time_us,
                                                    double& model_end_time_us,
                                                    Histogram& histogram) const {
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

        for (int j = i + 1; j < n_hits; ++j) {
            const double dt = hits[j].time_us - t0;
            if (dt > region_settings.coincidence_window_us) break;
            if (hits[j].channel != ch0) armed = true;
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

    // Telescoping until gap > min_gap_us
    int j = seed_index + 1;
    while (j < n_hits) {
        const double gap = hits[j].time_us - hits[j - 1].time_us;
        if (gap > region_settings.min_gap_us) break;
        ++j;
    }
    const double last_hit_time_us = hits[j - 1].time_us;

    // Pre-pad nad post-pad
    const double prepad_us = std::max(0.0, region_settings.fit_start_padding_us);
    const double postpad_us = std::max(0.0, region_settings.fit_end_padding_us);
    
    start_time_us = seed_time_us - prepad_us;
    end_time_us = last_hit_time_us;
    model_end_time_us = last_hit_time_us + postpad_us;
    next_index = j;

    const double model_width_us = model_end_time_us - start_time_us;
    if (model_width_us < bin_width_us) {
        return false;
    }
    
    const int n_bins = static_cast<int>(std::ceil(model_width_us / bin_width_us));
    if (n_bins <= 0) {
        return false;
    }

    histogram.bin_edges_us.clear();
    histogram.counts.assign(static_cast<std::size_t>(n_bins), 0.0);
    
    for (int b = 0; b <= n_bins; ++b) {
        histogram.bin_edges_us.push_back(start_time_us + b * bin_width_us);
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
    double template_support_us = static_cast<double>(pulse_template_.pmf().size()) * pulse_template_.native_bin_width_us();

    for (const PulseCandidate& pulse : carry_pulses) {
        double template_start_us = pulse.time_us;
        double template_end_us = template_start_us + template_support_us;
        if (template_end_us <= window_start_us - 20.0) {
            continue;
        }
        std::vector<double> component = pulse_template_.shifted_to_histogram(pulse.time_us, histogram.bin_edges_us);
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
    std::vector<TaggedPulse>& output_pulses,
    std::vector<WindowSummary>& output_summaries,
    const std::vector<debug::TruthPulse>* truth_pulses,
    int chunk_index,
    int global_window_offset
) const {
    std::vector<PulseCandidate> carry_pulses;

    int i = 0;
    int window_index = 0;
    
    while (i < static_cast<int>(hits.size())) {

        Histogram coarse_histogram;
        double start_time_us = 0.0;
        double end_time_us = 0.0;
        double model_end_time_us = 0.0;
        int next_index = i;

        bool ok = build_window_histogram(hits,
                                         i,
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

        // std::vector<double> fixed_expected = build_carry_expected(carry_pulses, histogram);
        std::vector<double> fixed_expected(histogram.counts.size(), 0.0);
        std::vector<double> seeds = find_coincidence_seeds(
            hits, start_time_us, end_time_us, bin_width_us, region_settings
        );
        
        if (seeds.empty()) {
            i = next_index;
            ++window_index;
            continue;
        }

        const std::vector<debug::TruthPulse> truth_here = 
            select_truth_in_window(truth_pulses, start_time_us, model_end_time_us);
        
        const debug::DebugCaseType case_type = classify_debug_case(
            static_cast<int>(truth_here.size()), static_cast<int>(seeds.size())
        );

        bool capture_debug = false;
        std::string case_id;

        if (debug_writer_ &&
            debug_writer_->enabled() &&
            case_type != debug::DebugCaseType::Unknown &&
            debug_writer_->can_capture(case_type)) {
            capture_debug = true;
            case_id = debug_writer_->next_case_id(case_type);
        }

        FitSettings window_fit_settings = fit_settings;
        window_fit_settings.background_per_bin = background_rate_hz * bin_width_us * 1.0e-6;
        window_fit_settings.fixed_expected = fixed_expected;

        auto fit_t0 = std::chrono::steady_clock::now();

        FitResult fit = fitter_.fit(
            histogram, seeds, window_fit_settings, capture_debug ? debug_writer_.get() : nullptr, case_id
        );

        auto fit_t1 = std::chrono::steady_clock::now();

        if (fit.pulses.empty()) {
            i = next_index;
            ++window_index;
            continue;
        }

        if (capture_debug) {
            debug::WindowDebugContext ctx;
            ctx.case_id = case_id;
            ctx.case_type = case_type;
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
        double observed_sum = std::accumulate(histogram.counts.begin(), histogram.counts.end(), 0.0);

        WindowSummary summary;
        summary.window_index = window_index;
        summary.start_time_us = start_time_us;
        summary.end_time_us = end_time_us;
        summary.width_us = end_time_us - start_time_us;
        summary.bin_width_us = bin_width_us;
        summary.pulse_count = static_cast<int>(fit.pulses.size());
        summary.observed_count = static_cast<int>(std::llround(observed_sum));
        summary.expected_count = expected_sum;
        summary.final_nll = -compute_log_likelihood(histogram.counts, full_expected);
        summary.seed_count = static_cast<int>(seeds.size());
        output_summaries.push_back(summary);

        double template_support_us = static_cast<double>(pulse_template_.pmf().size()) * pulse_template_.native_bin_width_us();
        std::vector<PulseCandidate> updated_carry;
        for (const PulseCandidate& pulse : carry_pulses) {
            double template_start_us = pulse.time_us;
            if (template_start_us + template_support_us > start_time_us) {
                updated_carry.push_back(pulse);
            }
        }
        carry_pulses.swap(updated_carry);

        for (const PulseCandidate& pulse : fit.pulses) {
            TaggedPulse tagged;
            tagged.time_us = pulse.time_us;
            tagged.amplitude_pe = pulse.amplitude_pe;
            tagged.window_index = window_index;
            tagged.width_us = end_time_us - start_time_us;
            tagged.is_pileup = fit.pulses.size() > 1;
            tagged.uses_fine_bins = uses_fine_bins;
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
    int global_window_offset
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
                       background_pulses,
                       background_summaries,
                       nullptr, -1, 0);

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
               result.signal_pulses,
               result.signal_window_summaries,
               truth_pulses,
               chunk_index,
               global_window_offset);

    fit_stream(end_hits,
               region_settings,
               fit_settings,
               background_rate_hz,
               result.end_pulses,
               result.end_window_summaries,
               nullptr, -1, 0);

    result.background_rate_hz = background_rate_hz;
    return result;
}

} // namespace ucn
