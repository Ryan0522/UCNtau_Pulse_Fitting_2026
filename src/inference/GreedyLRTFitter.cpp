#include "ucn/inference/GreedyLRTFitter.hpp"
#include "ucn/inference/Likelihood.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace ucn
{

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
    if (histogram.bin_edges_us.size() < 2) {
        return bounds;
    }

    double hist_left = histogram.bin_edges_us.front();
    double hist_right = histogram.bin_edges_us.back();

    for (const std::vector<double>& cluster : clusters) {
        if (cluster.empty()) {
            continue;
        }
        double cluster_time = cluster.front();
        ClusterBound bound;
        bound.cluster_time_us = cluster_time;
        bound.left_us = std::max(hist_left, cluster_time - settings.max_offset_us);
        bound.right_us = std::min(hist_right, cluster_time + settings.max_offset_us);
        bounds.push_back(bound);
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
    if (!settings.enable_back_pruning || current.pulses.size() < 2) {
        return best;
    }

    bool changed = true;
    while (changed && best.pulses.size() > 1) {
        changed = false;

        for (std::size_t remove_index = 0; remove_index < best.pulses.size(); ++remove_index) {
            std::vector<PulseCandidate> trial_pulses;
            std::vector<std::vector<double>> trial_components;
            std::vector<double> trial_amplitudes;

            for (std::size_t j = 0; j < best.pulses.size(); ++j) {
                if (j == remove_index) {
                    continue;
                }
                trial_pulses.push_back(best.pulses[j]);
                trial_components.push_back(components[j]);
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
            double delta = trial_nll - best.final_nll;

            if (delta < settings.delta_nll_cut) {
                best.pulses = trial_pulses;
                best.expected_total = trial_expected;
                best.final_nll = trial_nll;
                changed = true;
                break;
            }
        }
    }

    return best;
}

FitResult GreedyLRTFitter::fit(
    const Histogram& histogram,
    std::span<const double> coincidence_times_us,
    const FitSettings& settings
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

    if (settings.debug) {
        std::cerr << "clusters=" << clusters.size() << "\n";
    }

    std::vector<bool> cluster_used(bounds.size(), false);

    while (true) {
        double best_delta = 0.0;
        int best_cluster_index = -1;
        double best_time = 0.0;
        double best_amplitude = 0.0;
        std::vector<double> best_component;
        std::vector<double> best_expected;
        std::vector<double> best_refit_amplitudes;

        for (std::size_t cluster_index = 0; cluster_index < bounds.size(); ++cluster_index) {
            if (cluster_used[cluster_index]) {
                continue;
            }

            const ClusterBound& bound = bounds[cluster_index];
            for (double time_us = bound.left_us; time_us <= bound.right_us + 0.5 * settings.scan_step_us; time_us += settings.scan_step_us) {
                bool too_close = false;
                for (const PulseCandidate& pulse : result.pulses) {
                    if (std::abs(pulse.time_us - time_us) < settings.min_spacing_us) {
                        too_close = true;
                        break;
                    }
                }
                if (too_close) {
                    continue;
                }

                std::vector<double> component = pulse_template_.shifted_to_histogram(time_us, histogram.bin_edges_us);
                double init_amplitude = settings.min_amplitude_pe;
                for (std::size_t i = 0; i < histogram.counts.size(); ++i) {
                    double residual = histogram.counts[i] - result.expected_total[i];
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
                    settings.min_amplitude_pe,
                    settings.max_amplitude_pe,
                    init_amplitude
                );

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

                std::vector<double> trial_expected = likelihood::sum_expected(trial_components, trial_amplitudes);
                double trial_nll = likelihood::poisson_nll(
                    histogram.counts,
                    trial_expected,
                    settings.fixed_expected,
                    settings.background_per_bin
                );
                double delta = result.final_nll - trial_nll;

                if (delta > best_delta) {
                    best_delta = delta;
                    best_cluster_index = static_cast<int>(cluster_index);
                    best_time = time_us;
                    best_amplitude = trial_amplitudes.back();
                    best_component = component;
                    best_expected = trial_expected;
                    best_refit_amplitudes = trial_amplitudes;
                }
            }
        }

        if (best_cluster_index < 0 || best_delta < settings.delta_nll_cut) {
            break;
        }

        cluster_used[static_cast<std::size_t>(best_cluster_index)] = true;
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

        if (settings.debug) {
            std::cerr << "accepted cluster=" << best_cluster_index
                      << " time_us=" << best_time
                      << " delta_nll=" << best_delta
                      << " pulses=" << result.pulses.size() << "\n";
        }

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
