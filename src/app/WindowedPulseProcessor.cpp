#include "ucn/app/WindowedPulseProcessor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <chrono>

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

} // namespace

WindowedPulseProcessor::WindowedPulseProcessor(const PulseTemplate& pulse_template,
                                               const GreedyLRTFitter& fitter)
    : pulse_template_(pulse_template),
      fitter_(fitter) {
}

void WindowedPulseProcessor::set_debug_max_windows(int n) {
    debug_max_windows_ = n;
}

void WindowedPulseProcessor::set_progress_enabled(bool enabled) {
    progress_enabled_ = enabled;
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
                                                    Histogram& histogram) const {
    int n_hits = static_cast<int>(hits.size());
    if (start_index >= n_hits) {
        next_index = start_index;
        return false;
    }

    int j = start_index + 1;
    while (j < n_hits) {
        double gap = hits[j].time_us - hits[j - 1].time_us;
        if (gap > region_settings.min_gap_us) {
            break;
        }
        ++j;
    }

    const double first_hit_time_us = hits[start_index].time_us;
    const double raw_start_time_us = first_hit_time_us - std::max(0.0, region_settings.fit_start_padding_us);

    start_time_us = raw_start_time_us;
    
    double last_hit_time_us = hits[j - 1].time_us;
    double padded_end_us = last_hit_time_us + region_settings.fit_end_padding_us;
    double next_hit_time_us = std::numeric_limits<double>::infinity();
    if (j < n_hits) {
        next_hit_time_us = hits[j].time_us;
    }

    if (std::isfinite(next_hit_time_us)) {
        end_time_us = std::min(padded_end_us,
                               std::nextafter(next_hit_time_us,
                                              -std::numeric_limits<double>::infinity()));
    } else {
        end_time_us = padded_end_us;
    }

    next_index = j;

    double width_us = end_time_us - start_time_us;
    if (width_us < bin_width_us) {
        return false;
    }

    int n_bins = static_cast<int>(std::ceil(width_us / bin_width_us));
    if (n_bins < 1) {
        return false;
    }

    histogram.bin_edges_us.clear();
    histogram.counts.assign(static_cast<std::size_t>(n_bins), 0.0);
    for (int i = 0; i <= n_bins; ++i) {
        histogram.bin_edges_us.push_back(start_time_us + static_cast<double>(i) * bin_width_us);
    }

    for (int i = start_index; i < next_index; ++i) {
        double dt_us = hits[i].time_us - start_time_us;
        int bin = static_cast<int>(dt_us / bin_width_us);
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

void WindowedPulseProcessor::fit_stream(const std::vector<Hit>& hits,
                                        const RegionSettings& region_settings,
                                        const FitSettings& fit_settings,
                                        double background_rate_hz,
                                        std::vector<TaggedPulse>& output_pulses,
                                        std::vector<WindowSummary>& output_summaries) const {
    std::vector<PulseCandidate> carry_pulses;

    int i = 0;
    int window_index = 0;
    int debug_window_count = 0;

    const double stream_start_us = hits.empty() ? 0.0 : hits.front().time_us;
    const double stream_end_us   = hits.empty() ? 0.0 : hits.back().time_us;
    const double stream_width_s  = std::max(1.0e-12, (stream_end_us - stream_start_us) * 1.0e-6);

    int last_report_second = -1;    
    
    while (i < static_cast<int>(hits.size())) {
        if (debug_max_windows_ >= 0 && debug_window_count >= debug_max_windows_) {
            if (region_settings.debug) {
                std::cerr << std::fixed << std::setprecision(3);
                std::cerr << "[STOP] Reached debug_max_windows_ = "
                        << debug_max_windows_ << "\n";
            }
            break;
        }

        Histogram coarse_histogram;
        double start_time_us = 0.0;
        double end_time_us = 0.0;
        int next_index = i;

        bool ok = build_window_histogram(hits,
                                         i,
                                         region_settings.coarse_bin_width_us,
                                         region_settings,
                                         next_index,
                                         start_time_us,
                                         end_time_us,
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
        std::vector<double> seeds = find_coincidence_seeds(hits,
                                                           start_time_us,
                                                           end_time_us,
                                                           bin_width_us,
                                                           region_settings);

        if (progress_enabled_) {
            const double reached_s =
                (end_time_us - stream_start_us) * 1.0e-6;

            const double percent =
                100.0 * reached_s / stream_width_s;

            const int report_second =
                static_cast<int>(std::floor(reached_s));

            if (report_second > last_report_second || end_time_us >= stream_end_us) {
                last_report_second = report_second;

                std::cerr << std::fixed << std::setprecision(2)
                        << "\r[fit_stream] reached "
                        << reached_s << " / " << stream_width_s << " s"
                        << " (" << percent << "%)"
                        << " | window=" << window_index
                        << " | width_us=" << (end_time_us - start_time_us)
                        << " | hits=" << next_index << "/" << hits.size()
                        << " | seeds=" << seeds.size()
                        << std::flush;
            }
        }

        if (seeds.empty()) {
            i = next_index;
            ++window_index;
            continue;
        }

        if (region_settings.debug) {
            debug_window_count++;
            std::cerr << std::fixed << std::setprecision(3);
            std::cerr << "\n========== [WINDOW " << window_index << "] ==========\n";
            std::cerr << "[WINDOW] start_time_us=" << start_time_us
                    << " end_time_us=" << end_time_us
                    << " width_us=" << (end_time_us - start_time_us)
                    << " next_index=" << next_index
                    << " uses_fine_bins=" << uses_fine_bins
                    << " bin_width_us=" << bin_width_us
                    << "\n";

            std::cerr << "[WINDOW-EDGES] bin_edges_us =";
            for (double x : histogram.bin_edges_us) std::cerr << " " << x;
            std::cerr << "\n";

            std::cerr << "[WINDOW-HIST] counts =";
            for (double c : histogram.counts) std::cerr << " " << c;
            std::cerr << "\n";

            double carry_sum = std::accumulate(fixed_expected.begin(),
                                            fixed_expected.end(), 0.0);
            std::cerr << "[WINDOW-PRE-FIT] carry_sum=" << carry_sum
                    << " n_seeds=" << seeds.size()
                    << "\n";

            std::cerr << "[WINDOW-SEEDS] seeds =";
            for (double s : seeds) std::cerr << " " << s;
            std::cerr << "\n";
        }

        FitSettings window_fit_settings = fit_settings;
        window_fit_settings.background_per_bin = background_rate_hz * bin_width_us * 1.0e-6;
        window_fit_settings.fixed_expected = fixed_expected;

        if (progress_enabled_) {
            if (end_time_us - start_time_us > 4000) {
                std::cerr << std::fixed << std::setprecision(3)
                        << "\n[fit_stream] fitting large window=" << window_index
                        << " start_s=" << (start_time_us - stream_start_us) * 1.0e-6
                        << " end_s=" << (end_time_us - stream_start_us) * 1.0e-6
                        << " width_us=" << (end_time_us - start_time_us)
                        << " n_bins=" << histogram.counts.size()
                        << " n_hits=" << (next_index - i)
                        << " n_seeds=" << seeds.size()
                        << "\n";
            }
        }

        auto fit_t0 = std::chrono::steady_clock::now();

        FitResult fit = fitter_.fit(histogram, seeds, window_fit_settings);

        auto fit_t1 = std::chrono::steady_clock::now();

        if (progress_enabled_) {
            if (end_time_us - start_time_us > 4000) {
                const double fit_seconds =
                    std::chrono::duration<double>(fit_t1 - fit_t0).count();

                std::cerr << "[fit_stream] done window=" << window_index
                        << " fit_time_s=" << fit_seconds
                        << " n_pulses=" << fit.pulses.size()
                        << "\n";
            }
        }

        if (fit.pulses.empty()) {
            i = next_index;
            ++window_index;
            continue;
        }

        if (region_settings.debug) {
            std::cerr << "[WINDOW-FIT-DONE] window_index=" << window_index
                    << " n_pulses=" << fit.pulses.size()
                    << " final_nll=" << fit.final_nll
                    << "\n";

            for (std::size_t k = 0; k < fit.pulses.size(); ++k) {
                std::cerr << "  pulse[" << k << "]"
                        << " t=" << fit.pulses[k].time_us
                        << " a=" << fit.pulses[k].amplitude_pe
                        << "\n";
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
        double expected_sum = std::accumulate(fit.expected_total.begin(), fit.expected_total.end(), 0.0);
        double observed_sum = std::accumulate(histogram.counts.begin(), histogram.counts.end(), 0.0);

        WindowSummary summary;
        summary.window_index = window_index;
        summary.start_time_us = start_time_us;
        summary.end_time_us = end_time_us;
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
            tagged.window_width_us = end_time_us - start_time_us;
            tagged.is_pileup = fit.pulses.size() > 1;
            tagged.uses_fine_bins = uses_fine_bins;
            output_pulses.push_back(tagged);
            carry_pulses.push_back(pulse);
        }

        i = next_index;
        ++window_index;
    }

    if (progress_enabled_) {
        std::cerr << "\n[fit_stream] finished. windows=" << window_index
                << " pulses_out=" << output_pulses.size()
                << "\n";
    }
}

RegionResult WindowedPulseProcessor::analyze(const std::vector<Hit>& hits,
                                             double background_start_us,
                                             double background_end_us,
                                             double signal_start_us,
                                             double signal_end_us,
                                             double end_start_us,
                                             double end_end_us,
                                             const RegionSettings& region_settings,
                                             const FitSettings& fit_settings) const {
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
                       background_summaries);

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
               result.signal_window_summaries);

    fit_stream(end_hits,
               region_settings,
               fit_settings,
               background_rate_hz,
               result.end_pulses,
               result.end_window_summaries);

    result.background_rate_hz = background_rate_hz;
    return result;
}

} // namespace ucn
