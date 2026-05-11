#include "ucn/inference/GreedyLRTFitter.hpp"
#include "ucn/inference/Likelihood.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <numeric>

// This branch implements the local-sequential Greedy LRT fitter only.
// It does not support the legacy global-greedy scan.
// Discovery is local and left-to-right.
// Final PE amplitudes are produced by a full-window simultaneous refit.

namespace ucn
{

namespace {

double poisson_nll_bin(
    double observed, double expected,
    double fixed_expected, double background_per_bin
) {
    const double mu = std::max(
        expected + std::max(0.0, fixed_expected) + std::max(0.0, background_per_bin), 1.0e-12
    );
    return mu - observed * std::log(mu);
}

struct BinRange {
    std::size_t first = 0;
    std::size_t last = 0; // one past last
};

BinRange bins_overlapping_interval(
    const Histogram& histogram,
    double left_us,
    double right_us
) {
    BinRange r;
    r.first = histogram.counts.size();
    r.last = histogram.counts.size();

    if (right_us <= left_us || histogram.bin_edges_us.size() != histogram.counts.size() + 1) {
        return r;
    }

    for (std::size_t i = 0; i < histogram.counts.size(); ++i) {
        const double bin_left = histogram.bin_edges_us[i];
        const double bin_right = histogram.bin_edges_us[i + 1];

        if (bin_right <= left_us) continue;
        if (bin_left >= right_us) break;

        if (r.first == histogram.counts.size()) {
            r.first = i;
        }
        r.last = i + 1;
    }

    if (r.first == histogram.counts.size()) {
        r.first = 0;
        r.last = 0;
    }

    return r;
}

BinRange local_partition_for_cluster(
    const Histogram& histogram,
    const std::vector<ClusterBound>& bounds,
    std::size_t cluster_index
) {
    if (bounds.empty() || cluster_index >= bounds.size()) {
        return {};
    }

    const double hist_left = histogram.bin_edges_us.front();
    const double hist_right = histogram.bin_edges_us.back();
    const double center = bounds[cluster_index].cluster_time_us;

    double left = hist_left;
    double right = hist_right;

    if (cluster_index > 0) {
        left = 0.5 * (bounds[cluster_index - 1].cluster_time_us + center);
    }
    if (cluster_index + 1 < bounds.size()) {
        right = 0.5 * (center + bounds[cluster_index + 1].cluster_time_us);
    }

    left = std::clamp(left, hist_left, hist_right);
    right = std::clamp(right, hist_left, hist_right);

    return bins_overlapping_interval(histogram, left, right);
}

double poisson_nll_range(
    const Histogram& histogram,
    const std::vector<double>& expected,
    const std::vector<double>& fixed_expected,
    double background_per_bin,
    BinRange range
) {
    range.last = std::min(range.last, histogram.counts.size());

    double nll = 0.0;
    for (std::size_t i = range.first; i < range.last; ++i) {
        const double fixed = fixed_expected.empty() ? 0.0 : fixed_expected[i];
        nll += poisson_nll_bin(
            histogram.counts[i],
            expected[i],
            fixed,
            background_per_bin
        );
    }
    return nll;
}

std::vector<double> add_scaled_component(
    const std::vector<double>& base,
    const std::vector<double>& component,
    double amplitude
) {
    std::vector<double> out = base;
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] += amplitude * component[i];
    }
    return out;
}

double optimizer_cluster_local_full_amplitude(
    const Histogram& histogram,
    const std::vector<double>& base_expected,
    const std::vector<double>& component,
    const std::vector<double>& fixed_expected,
    double background_per_bin,
    double max_amplitude_pe,
    double local_template_mass_floor,
    BinRange range,
    double& local_mass_out
) {
    range.last = std::min(range.last, histogram.counts.size());

    local_mass_out = 0.0;
    for (std::size_t i = range.first; i < range.last; ++i) {
        local_mass_out += component[i];
    }
    if (!(local_mass_out > local_template_mass_floor) ||
        !std::isfinite(local_mass_out)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double local_upper = std::max(0.0, max_amplitude_pe * local_mass_out);

    double local_amp = 0.0;
    for (std::size_t i = range.first; i < range.last; ++i) {
        const double fixed = fixed_expected.empty() ? 0.0 : fixed_expected[i];
        const double residual = 
            histogram.counts[i]
            - base_expected[i]
            - fixed
            - background_per_bin;

        if (residual > 0.0) {
            local_amp += residual;
        }
    }

    local_amp = std::clamp(local_amp, 0.0, local_upper);
    for (int iter = 0; iter < 40; ++iter) {
        double grad = 0.0, hess = 0.0;
        for (std::size_t i = range.first; i < range.last; ++i) {
            const double s_local = component[i] / local_mass_out;
            double mu = base_expected[i] + local_amp * s_local;
            if (!fixed_expected.empty()) {
                mu += std::max(0.0, fixed_expected[i]);
            }
            mu += std::max(0.0, background_per_bin);
            mu = std::max(mu, 1.0e-12);

            grad += s_local * (1.0 - histogram.counts[i] / mu);
            hess += s_local * s_local * histogram.counts[i] / (mu * mu);
        }

        if (std::abs(grad) < 1.0e-10) break;
        if (hess <= 1.0e-12) break;

        double candidate = local_amp - grad / hess;
        candidate = std::clamp(candidate, 0.0, local_upper);

        if (std::abs(candidate - local_amp) < 1.0e-10) {
            local_amp = candidate;
            break;
        }
        local_amp = candidate;
    }
    return local_amp / local_mass_out;
}

} // namespace

GreedyLRTFitter::GreedyLRTFitter(const PulseTemplate& pulse_template)
    : pulse_template_(pulse_template) {
}
    
std::vector<std::vector<double>> GreedyLRTFitter::cluster_seed_times(
    std::span<const double> coincidence_times_us,
    double cluster_gap_us
) const {
    std::vector<double> sorted(coincidence_times_us.begin(), coincidence_times_us.end());
    std::sort(sorted.begin(), sorted.end());

    std::vector<std::vector<double>> clusters;
    if (sorted.empty()) {
        return clusters;
    }

    std::vector<double> current;
    current.push_back(sorted[0]);
    for (std::size_t i = 1; i < sorted.size(); ++i) {
        if (std::abs(sorted[i] - sorted[i - 1]) <= cluster_gap_us) {
            current.push_back(sorted[i]);
        } else {
            clusters.push_back(current);
            current.clear();
            current.push_back(sorted[i]);
        }
    }
    clusters.push_back(current);
    return clusters;
}

std::vector<ClusterBound> GreedyLRTFitter::build_cluster_bounds(
    const std::vector<std::vector<double>>& clusters,
    const Histogram& histogram,
    const FitSettings& settings
) const {
    std::vector<ClusterBound> bounds;
    if (clusters.empty() || histogram.bin_edges_us.empty()) {
        return bounds;
    }

    const double bin_left_us  = histogram.bin_edges_us.front();
    const double bin_right_us = histogram.bin_edges_us.back();

    // cluster centers = first seed in each cluster, same spirit as legacy
    std::vector<double> centers;
    centers.reserve(clusters.size());
    for (const auto& c : clusters) {
        if (!c.empty()) centers.push_back(c.front());
    }

    if (centers.empty()) {
        return bounds;
    }

    for (std::size_t i = 0; i < centers.size(); ++i) {
        const double t_center = centers[i];

        double left_span = settings.max_offset_us;
        double right_span = settings.max_offset_us;

        if (i > 0) {
            const double left_mid = 0.5 * (t_center - centers[i - 1]);
            left_span = std::min(settings.max_offset_us, left_mid);
        }

        if (i + 1 < centers.size()) {
            const double right_mid = 0.5 * (centers[i + 1] - t_center);
            right_span = std::min(settings.max_offset_us, right_mid);
        }

        ClusterBound b;
        b.cluster_time_us = t_center;
        b.left_us  = std::max(bin_left_us,  t_center - left_span);
        b.right_us = std::min(bin_right_us, t_center + right_span);

        if (b.right_us < b.left_us) {
            b.right_us = b.left_us;
        }

        bounds.push_back(b);
    }

    return bounds;
}

FitResult GreedyLRTFitter::discover_local_sequential(
    const Histogram& histogram,
    const std::vector<ClusterBound>& bounds,
    const FitSettings& settings,
    ucn::debug::DebugSink* debug_sink,
    const std::string& debug_case_id
) const {
    FitResult result;
    result.expected_total.assign(histogram.counts.size(), 0.0);
    result.final_nll = likelihood::poisson_nll(
        histogram.counts,
        result.expected_total,
        settings.fixed_expected,
        settings.background_per_bin
    );

    std::vector<double> running_fixed_expected = settings.fixed_expected;
    if (running_fixed_expected.empty()) {
        running_fixed_expected.assign(histogram.counts.size(), 0.0);
    }

    const std::vector<double> zero_base(histogram.counts.size(), 0.0);

    auto candidate_is_too_close = [&](double candidate_time_us) {
        for (const PulseCandidate& pulse : result.pulses) {
            if (std::abs(pulse.time_us - candidate_time_us) < settings.min_spacing_us) {
                return true;
            }
        }
        return false;
    };

    auto emit_debug_row = [&](
        const std::string& status, int cluster_index, double trial_time_us,
        double trial_amp, double nll_before, double nll_after, double delta_nll,
        double margin, int accepted
    ) {
        if (!debug_sink) return;

         ucn::debug::LRTTrialDebug row;
        row.case_id = debug_case_id;
        row.status = status;
        row.fit_iter = cluster_index + 1;
        row.cluster_index = cluster_index;
        row.trial_time_us = trial_time_us;
        row.trial_amp = trial_amp;
        row.nll_before = nll_before;
        row.nll_after = nll_after;
        row.delta_nll = delta_nll;
        row.margin = margin;
        row.accepted = accepted;

        if (accepted) {
            debug_sink->on_lrt_accept(row);
        } else {
            debug_sink->on_lrt_trial(row);
        }
    };

    for (std::size_t cluster_index = 0; cluster_index < bounds.size(); ++cluster_index) {
        const ClusterBound& bound = bounds[cluster_index];
        const BinRange local_range = local_partition_for_cluster(histogram, bounds, cluster_index);

        if (local_range.last <= local_range.first) {
            emit_debug_row(
                "skip_empty_partition", static_cast<int>(cluster_index), bound.cluster_time_us, 0.0,
                result.final_nll, result.final_nll, 0.0, -std::numeric_limits<double>::infinity(), 0
            );
            continue;
        }

        const double old_local_nll = poisson_nll_range(
            histogram, zero_base, running_fixed_expected, settings.background_per_bin, local_range
        );

        double best_margin = -std::numeric_limits<double>::infinity();
        double best_delta = 0.0;
        double best_time = 0.0;
        double best_amplitude = 0.0;
        std::vector<double> best_component;

        for (double time_us = bound.left_us;
             time_us <= bound.right_us + 0.5 * settings.scan_step_us;
             time_us += settings.scan_step_us) {
                
            if (candidate_is_too_close(time_us)) {
                emit_debug_row(
                    "skip_too_close", static_cast<int>(cluster_index), time_us, 0.0,
                    old_local_nll, old_local_nll, 0.0, -std::numeric_limits<double>::infinity(), 0
                );
                continue;
            }

            std::vector<double> component = pulse_template_.shifted_to_histogram(time_us, histogram.bin_edges_us);
            double local_mass = 0.0;
            const double amplitude = optimizer_cluster_local_full_amplitude(
                histogram, zero_base, component, running_fixed_expected,
                settings.background_per_bin, settings.max_amplitude_pe, settings.local_template_mass_floor,
                local_range, local_mass
            );

            if (!std::isfinite(amplitude)) {
                emit_debug_row(
                    "skip_low_mass", static_cast<int>(cluster_index), time_us, 0.0,
                    old_local_nll, old_local_nll, 0.0, -std::numeric_limits<double>::infinity(), 0
                );
                continue;
            }
            if (amplitude < settings.min_amplitude_pe) {
                emit_debug_row(
                    "skip_low_amp", static_cast<int>(cluster_index), time_us, amplitude,
                    old_local_nll, old_local_nll, 0.0, -std::numeric_limits<double>::infinity(), 0
                );
                continue;
            }

            std::vector<double> trial_expected = add_scaled_component(zero_base, component, amplitude);
            const double trial_local_nll = poisson_nll_range(
                histogram, trial_expected, running_fixed_expected, settings.background_per_bin, local_range
            );

            const double local_delta = old_local_nll - trial_local_nll;
            const double margin = local_delta - settings.discovery_delta_nll_cut;

            emit_debug_row(
                "candidate", static_cast<int>(cluster_index), time_us, amplitude,
                old_local_nll, trial_local_nll, local_delta, margin, 0
            );

            if (margin > best_margin) {
                best_margin = margin;
                best_delta = local_delta;
                best_time = time_us;
                best_amplitude = amplitude;
                best_component = std::move(component);
            }
        }

        if (best_margin < 0.0 || best_component.empty()) {
            emit_debug_row(
                "reject_cluster", static_cast<int>(cluster_index), bound.cluster_time_us, best_amplitude,
                old_local_nll, old_local_nll, best_delta, best_margin, 0
            );
            continue;
        }

        emit_debug_row(
            "accept", static_cast<int>(cluster_index), best_time, best_amplitude,
            old_local_nll, old_local_nll - best_delta, best_delta, best_margin, 1
        );

        result.pulses.push_back(PulseCandidate{best_time, best_amplitude});

        for (std::size_t i = 0; i < result.expected_total.size(); ++i) {
            const double add = best_amplitude * best_component[i];
            result.expected_total[i] += add;
            running_fixed_expected[i] += add;
        }

        result.final_nll = likelihood::poisson_nll(
            histogram.counts, result.expected_total, settings.fixed_expected, settings.background_per_bin
        );
    }

    return result;
}

FitResult GreedyLRTFitter::refit_amplitudes(
    const Histogram& histogram,
    const FitResult& input,
    const FitSettings& settings
) const {
    FitResult out = input;

    if (out.pulses.empty()) {
        out.expected_total.assign(histogram.counts.size(), 0.0);
        out.final_nll = likelihood::poisson_nll(
            histogram.counts,
            out.expected_total,
            settings.fixed_expected,
            settings.background_per_bin
        );
        return out;
    }

    std::vector<std::vector<double>> components;
    std::vector<double> amplitudes;

    components.reserve(out.pulses.size());
    amplitudes.reserve(out.pulses.size());

    for (const PulseCandidate& pulse : out.pulses) {
        components.push_back(
            pulse_template_.shifted_to_histogram(pulse.time_us, histogram.bin_edges_us)
        );
        amplitudes.push_back(
            std::clamp(pulse.amplitude_pe, 0.0, settings.max_amplitude_pe)
        );
    }

    amplitudes = likelihood::refit_all_amplitudes(
        histogram.counts, components, amplitudes,
        settings.fixed_expected, settings.background_per_bin, 0.0, // allow weak pulses to shrink before purning
        settings.max_amplitude_pe, settings.max_refit_steps, settings.refit_tolerance
    );

    for (std::size_t j = 0; j < out.pulses.size(); ++j) {
        out.pulses[j].amplitude_pe = amplitudes[j];
    }

    out.expected_total = likelihood::sum_expected(components, amplitudes);
    if (out.expected_total.empty()) {
        out.expected_total.assign(histogram.counts.size(), 0.0);
    }

    out.final_nll = likelihood::poisson_nll(
        histogram.counts,
        out.expected_total,
        settings.fixed_expected,
        settings.background_per_bin
    );

    return out;
}

FitResult GreedyLRTFitter::prune_by_deletion_lrt(
    const Histogram& histogram,
    const FitResult& input,
    const FitSettings& settings
) const {
    FitResult current = input;

    for (int pass = 0; pass < settings.max_prune_passes; ++pass) {
        current = refit_amplitudes(histogram, current, settings);

        if (current.pulses.empty()) break;

        double best_remove_delta = std::numeric_limits<double>::infinity();
        std::size_t best_remove_index = current.pulses.size();
        FitResult best_removed_result;

        for (std::size_t remove_index = 0; remove_index < current.pulses.size(); ++remove_index) {
            FitResult trial;
            trial.pulses.reserve(current.pulses.size() - 1);

            for (std::size_t j = 0; j < current.pulses.size(); ++j) {
                if (j != remove_index) {
                    trial.pulses.push_back(current.pulses[j]);
                }
            }

            trial = refit_amplitudes(histogram, trial, settings);

            const double delta_remove = trial.final_nll - current.final_nll;
            const bool weak_amplitude = current.pulses[remove_index].amplitude_pe < settings.min_amplitude_pe;
            const bool weak_lrt = delta_remove < settings.prune_delta_nll_cut;

            if ((weak_amplitude || weak_lrt) &&
                delta_remove < best_remove_delta) {
                best_remove_delta = delta_remove;
                best_remove_index = remove_index;
                best_removed_result = std::move(trial);
            }
        }

        if (best_remove_index == current.pulses.size()) break;

        current = std::move(best_removed_result);
    }
    return current;
}

FitResult GreedyLRTFitter::finalize_result(
    const Histogram& histogram,
    const FitResult& discovered,
    const FitSettings& settings
) const {
    FitResult refit = refit_amplitudes(histogram, discovered, settings);
    FitResult pruned = prune_by_deletion_lrt(histogram, refit, settings);
    return refit_amplitudes(histogram, pruned, settings);
}

FitResult GreedyLRTFitter::fit(
    const Histogram& histogram,
    std::span<const double> coincidence_times_us,
    const FitSettings& settings,
    ucn::debug::DebugSink* debug_sink,
    const std::string& debug_case_id
) const {
    if (histogram.bin_edges_us.size() != histogram.counts.size() + 1) {
        throw std::invalid_argument("Histogram bin_edges_us must have size counts.size() + 1.");
    }
    if (!settings.fixed_expected.empty() && settings.fixed_expected.size() != histogram.counts.size()) {
        throw std::invalid_argument("fixed_expected must match histogram size.");
    }

    const std::vector<std::vector<double>> clusters = cluster_seed_times(coincidence_times_us, settings.cluster_gap_us);
    const std::vector<ClusterBound> bounds = build_cluster_bounds(clusters, histogram, settings);

    FitResult discovered = discover_local_sequential(
        histogram, bounds, settings, debug_sink, debug_case_id
    );

    return finalize_result(histogram, discovered, settings);
}

} // namespace ucn
