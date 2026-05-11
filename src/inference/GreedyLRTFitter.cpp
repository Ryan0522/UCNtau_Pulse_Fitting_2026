#include "ucn/inference/GreedyLRTFitter.hpp"
#include "ucn/inference/Likelihood.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <numeric>

namespace ucn
{

namespace {

double close_pulse_regularization_penalty(
    const PulseCandidate& candidate,
    const std::vector<PulseCandidate>& existing_pulses,
    const FitSettings& settings
) {
    if (!settings.enable_close_pulse_regularization) return 0.0;
    if (settings.close_reg_lambda_nll <= 0.0) return 0.0;
    if (settings.close_reg_close_tau_us <= 0.0) return 0.0;
    if (settings.close_reg_tail_tau_us <= 0.0) return 0.0;

    double best_dt_us = std::numeric_limits<double>::infinity();
    double previous_amp_pe = 0.0;

    for (const PulseCandidate& p : existing_pulses) {
        const double dt_us = candidate.time_us - p.time_us;
        if (dt_us <= 0.0) continue;
        if (dt_us < best_dt_us) {
            best_dt_us = dt_us;
            previous_amp_pe = p.amplitude_pe;
        }
    }

    if (std::isinf(best_dt_us)) return 0.0;
    if (best_dt_us > settings.close_reg_window_us) return 0.0;

    const double residual_scale_pe = 
        settings.close_reg_eta * previous_amp_pe *
            std::exp(-best_dt_us / settings.close_reg_tail_tau_us)
        + settings.close_reg_floor_pe;

    const double safe_residual_scale_pe = std::max(residual_scale_pe, 1.0e-12);
    const double z = candidate.amplitude_pe / safe_residual_scale_pe;

    return settings.close_reg_lambda_nll *
           std::exp(-best_dt_us / settings.close_reg_close_tau_us) *
           std::exp(-z);
}

double poisson_nll_bin(
    double observed, double expected,
    double fixed_expected, double background_per_bin
) {
    const double mu = std::max(
        expected + std::max(0.0, fixed_expected) + std::max(0.0, background_per_bin), 1.0e-12
    );
    return mu - observed * std::log(mu);
}

double local_nll_improvement(
    const Histogram& histogram,
    const std::vector<double>& old_expected,
    const std::vector<double>& new_expected,
    const FitSettings& settings,
    double candidate_time_us
) {
    if (!settings.enable_local_evidence) {
        return std::numeric_limits<double>::infinity();
    }
    if (histogram.bin_edges_us.size() != histogram.counts.size() + 1) {
        throw std::invalid_argument("Histogram bin_edges_us must have size counts.size() + 1.");
    }
    if (old_expected.size() != histogram.counts.size() ||
        new_expected.size() != histogram.counts.size()) {
        throw std::invalid_argument("Expected histograms must match observed histogram size.");
    }
    if (!settings.fixed_expected.empty() &&
        settings.fixed_expected.size() != histogram.counts.size()) {
        throw std::invalid_argument("fixed_expected must match histogram size.");
    }

    const double left_us  = candidate_time_us - settings.local_evidence_pre_us;
    const double right_us = candidate_time_us + settings.local_evidence_post_us;

    double old_nll = 0.0;
    double new_nll = 0.0;
    int n_bins_used = 0;

    for (std::size_t i = 0; i < histogram.counts.size(); ++i) {
        const double bin_left_us = histogram.bin_edges_us[i];
        const double bin_right_us = histogram.bin_edges_us[i + 1];

        if (bin_right_us <= left_us || bin_left_us >= right_us) continue;

        const double fixed = settings.fixed_expected.empty() ? 0.0 : settings.fixed_expected[i];
        old_nll += poisson_nll_bin(
            histogram.counts[i], old_expected[i], fixed, settings.background_per_bin
        );
        new_nll += poisson_nll_bin(
            histogram.counts[i], new_expected[i], fixed, settings.background_per_bin
        );
        ++n_bins_used;
    }

    if (n_bins_used == 0) {
        return -std::numeric_limits<double>::infinity();
    }

    return old_nll - new_nll;
}

bool passes_local_evidence(
    const Histogram& histogram,
    const std::vector<double>& old_expected,
    const std::vector<double>& new_expected,
    const FitSettings& settings,
    double candidate_time_us,
    double& local_delta_out
) {
    if (!settings.enable_local_evidence) {
        local_delta_out = std::numeric_limits<double>::infinity();
        return true;
    }

    local_delta_out = local_nll_improvement(
        histogram, old_expected, new_expected, settings, candidate_time_us
    );
    return local_delta_out >= settings.local_delta_nll_cut;
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
    const FitSettings& settings,
    BinRange range
) {
    range.last = std::min(range.last, histogram.counts.size());

    double nll = 0.0;
    for (std::size_t i = range.first; i < range.last; ++i) {
        const double fixed = settings.fixed_expected.empty() ? 0.0 : settings.fixed_expected[i];
        nll += poisson_nll_bin(
            histogram.counts[i],
            expected[i],
            fixed,
            settings.background_per_bin
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
    const FitSettings& settings,
    BinRange range,
    double& local_mass_out
) {
    range.last = std::min(range.last, histogram.counts.size());

    local_mass_out = 0.0;
    for (std::size_t i = range.first; i < range.last; ++i) {
        local_mass_out += component[i];
    }
    if (!(local_mass_out > settings.local_template_mass_floor) ||
        !std::isfinite(local_mass_out)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double local_upper = std::max(0.0, settings.max_amplitude_pe * local_mass_out);

    double local_amp = 0.0;
    for (std::size_t i = range.first; i < range.last; ++i) {
        const double fixed = settings.fixed_expected.empty() ? 0.0 : settings.fixed_expected[i];
        const double residual = 
            histogram.counts[i]
            - base_expected[i]
            - fixed
            - settings.background_per_bin;

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
            if (!settings.fixed_expected.empty()) {
                mu += std::max(0.0, settings.fixed_expected[i]);
            }
            mu += std::max(0.0, settings.background_per_bin);
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

FitResult GreedyLRTFitter::remove_weak_pulses(
    const Histogram& histogram,
    const FitResult& current,
    const std::vector<std::vector<double>>& components,
    const FitSettings& settings
) const {
    FitResult best = current;
    std::vector<std::vector<double>> best_components = components;

    if (!settings.enable_back_pruning || current.pulses.size() < 2) {
        return best;
    }

    bool changed = true;
    while (changed && best.pulses.size() > 1) {
        changed = false;

        for (std::size_t remove_index = 0; remove_index < best.pulses.size(); ++remove_index) {
            const PulseCandidate removed_pulse = best.pulses[remove_index];
            
            std::vector<PulseCandidate> trial_pulses;
            std::vector<std::vector<double>> trial_components;
            std::vector<double> trial_amplitudes;

            for (std::size_t j = 0; j < best.pulses.size(); ++j) {
                if (j == remove_index) {
                    continue;
                }
                trial_pulses.push_back(best.pulses[j]);
                trial_components.push_back(best_components[j]);
                trial_amplitudes.push_back(best.pulses[j].amplitude_pe);
            }

            trial_amplitudes = likelihood::refit_all_amplitudes(
                histogram.counts,
                trial_components,
                trial_amplitudes,
                settings.fixed_expected,
                settings.background_per_bin,
                settings.min_amplitude_pe,
                settings.max_amplitude_pe,
                settings.max_coordinate_descent_steps
            );

            for (std::size_t j = 0; j < trial_pulses.size(); ++j) {
                trial_pulses[j].amplitude_pe = trial_amplitudes[j];
            }

            std::vector<double> trial_expected = likelihood::sum_expected(trial_components, trial_amplitudes);
            if (trial_expected.empty()) {
                trial_expected.assign(histogram.counts.size(), 0.0);
            }

            double trial_nll = likelihood::poisson_nll(
                histogram.counts,
                trial_expected,
                settings.fixed_expected,
                settings.background_per_bin
            );

            const double global_delta_keep = trial_nll - best.final_nll;
            const double penalty = close_pulse_regularization_penalty(
                removed_pulse, trial_pulses, settings
            );
            const double required_delta_keep = settings.delta_nll_cut + penalty;

            double local_delta_keep = std::numeric_limits<double>::infinity();
            const bool local_pass = passes_local_evidence(
                histogram,
                trial_expected,
                best.expected_total,
                settings,
                removed_pulse.time_us,
                local_delta_keep
            );

            if (global_delta_keep < required_delta_keep || !local_pass) {
                best.pulses = trial_pulses;
                best_components = trial_components;
                best.expected_total = trial_expected;
                best.final_nll = trial_nll;
                changed = true;
                break;
            }
        }
    }

    return best;
}

FitResult GreedyLRTFitter::fit_cluster_local_sequantial(
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

    std::vector<std::vector<double>> components;
    
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
        double penalty_nll, double required_delta_nll, double margin, int accepted
    ) {
        if (!debug_sink) return;

         ucn::debug::LRTTrialDebug row;
        row.case_id = debug_case_id;
        row.status = status;
        row.cluster_index = cluster_index;
        row.trial_time_us = trial_time_us;
        row.trial_amp = trial_amp;
        row.nll_before = nll_before;
        row.nll_after = nll_after;
        row.delta_nll = delta_nll;
        row.penalty_nll = penalty_nll;
        row.required_delta_nll = required_delta_nll;
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
                "local_skip_empty_partition", static_cast<int>(cluster_index), bound.cluster_time_us, 0.0,
                result.final_nll, result.final_nll, 0.0, 0.0,
                settings.delta_nll_cut, -std::numeric_limits<double>::infinity(), 0
            );
            continue;
        }

        const double old_local_nll = poisson_nll_range(
            histogram, result.expected_total, settings, local_range
        );

        double best_margin = -std::numeric_limits<double>::infinity();
        double best_delta = 0.0;
        double best_penalty = 0.0;
        double best_required_delta = settings.delta_nll_cut;
        double best_time = 0.0;
        double best_amplitude = 0.0;
        std::vector<double> best_component;
        std::vector<double> best_expected;

        for (double time_us = bound.left_us;
             time_us <= bound.right_us + 0.5 * settings.scan_step_us;
             time_us += settings.scan_step_us) {
            if (candidate_is_too_close(time_us)) {
                emit_debug_row(
                    "local_skip_too_close", static_cast<int>(cluster_index), time_us, 0.0,
                    old_local_nll, old_local_nll, 0.0, 0.0,
                    settings.delta_nll_cut, -std::numeric_limits<double>::infinity(), 0
                );
                continue;
            }

            std::vector<double> component = pulse_template_.shifted_to_histogram(time_us, histogram.bin_edges_us);
            double local_mass = 0.0;
            const double amplitude = optimizer_cluster_local_full_amplitude(
                histogram, result.expected_total, component,
                settings, local_range, local_mass
            );

            if (!std::isfinite(amplitude)) {
                emit_debug_row(
                    "local_skip_low_mass", static_cast<int>(cluster_index), time_us, 0.0,
                    old_local_nll, old_local_nll, 0.0, 0.0,
                    settings.delta_nll_cut, -std::numeric_limits<double>::infinity(), 0
                );
                continue;
            }
            if (amplitude < settings.min_amplitude_pe) {
                emit_debug_row(
                    "local_skip_low_amp", static_cast<int>(cluster_index), time_us, amplitude,
                    old_local_nll, old_local_nll, 0.0, 0.0,
                    settings.delta_nll_cut, -std::numeric_limits<double>::infinity(), 0
                );
                continue;
            }

            std::vector<double> trial_expected = add_scaled_component(result.expected_total, component, amplitude);
            const double trial_local_nll = poisson_nll_range(
                histogram, trial_expected, settings, local_range
            );

            const double local_delta = old_local_nll - trial_local_nll;

            PulseCandidate candidate{time_us, amplitude};
            const double penalty = close_pulse_regularization_penalty(
                candidate,
                result.pulses,
                settings
            );

            const double required_delta = settings.delta_nll_cut + penalty;
            const double margin = local_delta - required_delta;

            emit_debug_row(
                "local_candidate", static_cast<int>(cluster_index), time_us, amplitude,
                old_local_nll, trial_local_nll, local_delta, penalty,
                required_delta, margin, 0
            );

            if (margin > best_margin) {
                best_margin = margin;
                best_delta = local_delta;
                best_penalty = penalty;
                best_required_delta = required_delta;
                best_time = time_us;
                best_amplitude = amplitude;
                best_component = std::move(component);
                best_expected = std::move(trial_expected);
            }
        }

        if (best_margin < 0.0 || best_component.empty()) {
            emit_debug_row(
                "local_reject_cluster", static_cast<int>(cluster_index), bound.cluster_time_us, best_amplitude,
                old_local_nll, old_local_nll, best_delta, best_penalty,
                best_required_delta, best_margin, 0
            );
            continue;
        }

        emit_debug_row(
            "local_accept", static_cast<int>(cluster_index), best_time, best_amplitude,
            old_local_nll, old_local_nll - best_delta, best_delta, best_penalty,
            best_required_delta, best_margin, 1
        );

        result.pulses.push_back(PulseCandidate{best_time, best_amplitude});
        components.push_back(std::move(best_component));
        result.expected_total = std::move(best_expected);

        result.final_nll = likelihood::poisson_nll(
            histogram.counts, result.expected_total, settings.fixed_expected, settings.background_per_bin
        );
    }

    return result;
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

    FitResult result;
    result.expected_total.assign(histogram.counts.size(), 0.0);
    result.final_nll = likelihood::poisson_nll(
        histogram.counts,
        result.expected_total,
        settings.fixed_expected,
        settings.background_per_bin
    );

    std::vector<std::vector<double>> components;

    std::vector<std::vector<double>> clusters = cluster_seed_times(
        coincidence_times_us,
        settings.cluster_gap_us
    );

    std::vector<ClusterBound> bounds = build_cluster_bounds(clusters, histogram, settings);
    
    if (settings.use_cluster_local_amplitude_fit) {
        return fit_cluster_local_sequantial(
            histogram, bounds, settings, debug_sink, debug_case_id
        );
    }
    
    std::vector<bool> cluster_used(bounds.size(), false);

    auto candidate_is_too_close = [&](double candidate_time_us) {
        for (const PulseCandidate& pulse : result.pulses) {
            if (std::abs(pulse.time_us - candidate_time_us) < settings.min_spacing_us) {
                return true;
            }
        }
        return false;
    };

    int fit_iter = 0;
    auto emit_debug_row = [&](
        const std::string& status, int iter, int cluster_index, double trial_time_us,
        double trial_amp, double nll_before, double nll_after, double delta_nll,
        double penalty_nll, double required_delta_nll, double local_delta_nll, 
        int local_pass, double margin, int accepted
    ) {
        if (!debug_sink) return;

         ucn::debug::LRTTrialDebug row;
        row.case_id = debug_case_id;
        row.status = status;
        row.fit_iter = iter;
        row.cluster_index = cluster_index;
        row.trial_time_us = trial_time_us;
        row.trial_amp = trial_amp;
        row.nll_before = nll_before;
        row.nll_after = nll_after;
        row.delta_nll = delta_nll;
        row.penalty_nll = penalty_nll;
        row.required_delta_nll = required_delta_nll;
        row.local_delta_nll = local_delta_nll;
        row.local_pass = local_pass;
        row.margin = margin;
        row.accepted = accepted;

        if (accepted) {
            debug_sink->on_lrt_accept(row);
        } else {
            debug_sink->on_lrt_trial(row);
        }
    };

    while (true) {
        ++fit_iter;

        double best_margin = -std::numeric_limits<double>::infinity();
        double best_delta = 0.0;
        double best_required_delta = settings.delta_nll_cut;
        double best_penalty = 0.0;
        double best_local_delta = std::numeric_limits<double>::infinity();
        int best_cluster_index = -1;
        double best_time = 0.0;
        double best_amplitude = 0.0;
        std::vector<double> best_component;
        std::vector<double> best_expected;
        std::vector<double> best_refit_amplitudes;

        for (std::size_t cluster_index = 0; cluster_index < bounds.size(); ++cluster_index) {          
            if (!settings.allow_multiple_fits_per_cluster && cluster_used[cluster_index]) {
                emit_debug_row(
                    "skip_cluster_used", fit_iter, static_cast<int>(cluster_index), 0.0, 0.0,
                    result.final_nll, result.final_nll, 0.0, 0.0, settings.delta_nll_cut,
                    0.0, 1, -std::numeric_limits<double>::infinity(), 0
                );
                continue;
            }

            const ClusterBound& bound = bounds[cluster_index];

            for (double time_us = bound.left_us;
                time_us <= bound.right_us + 0.5 * settings.scan_step_us;
                time_us += settings.scan_step_us) {
                if (candidate_is_too_close(time_us)) {
                    emit_debug_row(
                        "skip_too_close", fit_iter, static_cast<int>(cluster_index), time_us, 0.0,
                        result.final_nll, result.final_nll, 0.0, 0.0, settings.delta_nll_cut,
                        0.0, 1, -std::numeric_limits<double>::infinity(), 0
                    );
                    continue;
                }

                std::vector<double> component = pulse_template_.shifted_to_histogram(time_us, histogram.bin_edges_us);
                double init_amplitude = settings.min_amplitude_pe;
                for (std::size_t i = 0; i < histogram.counts.size(); ++i) {
                    const double fixed = settings.fixed_expected.empty() ? 0.0 : settings.fixed_expected[i];
                    const double residual = histogram.counts[i] - result.expected_total[i] - fixed - settings.background_per_bin;
                    if (residual > 0.0) {
                        init_amplitude += residual * component[i];
                    }
                }
                init_amplitude = std::clamp(init_amplitude, settings.min_amplitude_pe, settings.max_amplitude_pe);

                double amplitude = likelihood::optimize_single_amplitude(
                    histogram.counts,
                    result.expected_total,
                    component,
                    settings.fixed_expected,
                    settings.background_per_bin,
                    0.0, // allow zero amplitude for initial scan
                    settings.max_amplitude_pe,
                    init_amplitude
                );
                if (amplitude < settings.min_amplitude_pe) continue;

                std::vector<std::vector<double>> trial_components = components;
                std::vector<double> trial_amplitudes;
                for (const PulseCandidate& pulse : result.pulses) {
                    trial_amplitudes.push_back(pulse.amplitude_pe);
                }
                trial_components.push_back(component);
                trial_amplitudes.push_back(amplitude);

                trial_amplitudes = likelihood::refit_all_amplitudes(
                    histogram.counts,
                    trial_components,
                    trial_amplitudes,
                    settings.fixed_expected,
                    settings.background_per_bin,
                    settings.min_amplitude_pe,
                    settings.max_amplitude_pe,
                    settings.max_coordinate_descent_steps
                );
                
                const double candidate_refit_amp = trial_amplitudes.back();
                if (candidate_refit_amp < settings.min_amplitude_pe) {
                    emit_debug_row(
                        "skip_low_refit_amp", fit_iter, static_cast<int>(cluster_index), time_us, candidate_refit_amp,
                        result.final_nll, result.final_nll, 0.0, 0.0, settings.delta_nll_cut,
                        0.0, 1, -std::numeric_limits<double>::infinity(), 0
                    );
                    continue;
                }

                std::vector<double> trial_expected = likelihood::sum_expected(trial_components, trial_amplitudes);
                double trial_nll = likelihood::poisson_nll(
                    histogram.counts,
                    trial_expected,
                    settings.fixed_expected,
                    settings.background_per_bin
                );
                const double delta = result.final_nll - trial_nll;

                std::vector<PulseCandidate> refit_existing = result.pulses;
                for (std::size_t j = 0; j < refit_existing.size() && j < trial_amplitudes.size(); ++j) {
                    refit_existing[j].amplitude_pe = trial_amplitudes[j];
                }
                PulseCandidate candidate{time_us, trial_amplitudes.back()};

                const double penalty = close_pulse_regularization_penalty(
                    candidate, refit_existing, settings
                );
                const double required_delta = settings.delta_nll_cut + penalty;

                double local_delta = std::numeric_limits<double>::infinity();
                const bool local_pass = passes_local_evidence(
                    histogram,
                    result.expected_total,
                    trial_expected,
                    settings,
                    time_us,
                    local_delta
                );

                const double margin = local_pass ? (delta - required_delta)
                                                 : -std::numeric_limits<double>::infinity();

                if (debug_sink) {
                    emit_debug_row(
                        "candidate", fit_iter, static_cast<int>(cluster_index), time_us, candidate_refit_amp,
                        result.final_nll, trial_nll, delta, penalty, required_delta, 
                        local_delta, local_pass ? 1 : 0, margin, 0
                    );
                }

                if (margin > best_margin) {
                    best_margin = margin;
                    best_delta = delta;
                    best_required_delta = required_delta;
                    best_penalty = penalty;
                    best_local_delta = local_delta;
                    best_cluster_index = static_cast<int>(cluster_index);
                    best_time = time_us;
                    best_amplitude = trial_amplitudes.back();
                    best_component = component;
                    best_expected = trial_expected;
                    best_refit_amplitudes = trial_amplitudes;
                }
            }
        }

        if (best_cluster_index < 0 || best_margin < 0.0) {
            emit_debug_row(
                best_cluster_index < 0 ? "stop_no_candidate" : "stop_negative_margin",
                fit_iter, best_cluster_index, best_time, best_amplitude,
                result.final_nll, result.final_nll, best_delta, best_penalty, best_required_delta,
                best_local_delta, 1, best_margin, 0
            );
            break;
        }

        if (!settings.allow_multiple_fits_per_cluster &&
            best_cluster_index >= 0) {
            cluster_used[static_cast<std::size_t>(best_cluster_index)] = true;
        }

        if (debug_sink) {
            emit_debug_row(
                "accept", fit_iter, best_cluster_index, best_time, best_amplitude,
                result.final_nll, 
                best_expected.empty() 
                    ? result.final_nll
                    : likelihood::poisson_nll(
                        histogram.counts, best_expected,
                        settings.fixed_expected, settings.background_per_bin
                    ),
                best_delta, best_penalty, best_required_delta,
                best_local_delta, 1, best_margin, 1
            );
        }

        result.pulses.push_back(PulseCandidate{best_time, best_amplitude});
        components.push_back(best_component);

        for (std::size_t j = 0; j < result.pulses.size() && j < best_refit_amplitudes.size(); ++j) {
            result.pulses[j].amplitude_pe = best_refit_amplitudes[j];
        }
        
        result.expected_total = best_expected;
        result.final_nll = likelihood::poisson_nll(
            histogram.counts,
            result.expected_total,
            settings.fixed_expected,
            settings.background_per_bin
        );

        FitResult pruned = remove_weak_pulses(histogram, result, components, settings);
        if (pruned.pulses.size() != result.pulses.size()) {
            std::vector<std::vector<double>> pruned_components;
            for (const PulseCandidate& pulse : pruned.pulses) {
                pruned_components.push_back(
                    pulse_template_.shifted_to_histogram(pulse.time_us, histogram.bin_edges_us)
                );
            }
            components = pruned_components;
            result = pruned;
        }
    }

    return result;
}

} // namespace ucn
