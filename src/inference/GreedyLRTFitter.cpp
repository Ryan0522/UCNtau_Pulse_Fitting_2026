#include "ucn/inference/GreedyLRTFitter.hpp"
#include "ucn/inference/Likelihood.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <iomanip>
#include <string>

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
    if (settings.debug) {
        std::cerr << "\n[FIT-ENTER]"
                << " nbins=" << histogram.counts.size()
                << " scan_step_us=" << settings.scan_step_us
                << " max_offset_us=" << settings.max_offset_us
                << " delta_nll_cut=" << settings.delta_nll_cut
                << " min_spacing_us=" << settings.min_spacing_us
                << " min_amplitude_pe=" << settings.min_amplitude_pe
                << " max_amplitude_pe=" << settings.max_amplitude_pe
                << "\n";

        std::cerr << "[FIT-HIST] counts =";
        for (double c : histogram.counts) std::cerr << " " << c;
        std::cerr << "\n";

        std::cerr << "[FIT-SEEDS-IN] seeds =";
        for (double s : coincidence_times_us) std::cerr << " " << s;
        std::cerr << "\n";
    }

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
    
    if (settings.debug) {
        std::cerr << "[FIT-CLUSTERS] n_clusters=" << clusters.size() << "\n";
        for (std::size_t k = 0; k < clusters.size(); ++k) {
            std::cerr << "  cluster[" << k << "] =";
            for (double t : clusters[k]) std::cerr << " " << t;
            std::cerr << "\n";
        }
    }

    std::vector<ClusterBound> bounds = build_cluster_bounds(clusters, histogram, settings);

    if (settings.debug) {
        std::cerr << "[FIT-BOUNDS] n_bounds=" << bounds.size() << "\n";
        for (std::size_t k = 0; k < bounds.size(); ++k) {
            std::cerr << "  bound[" << k << "]"
                    << " center=" << bounds[k].cluster_time_us
                    << " left=" << bounds[k].left_us
                    << " right=" << bounds[k].right_us
                    << " width=" << (bounds[k].right_us - bounds[k].left_us)
                    << "\n";
        }
    }

    std::vector<bool> cluster_used(bounds.size(), false);

    int fit_iter = 0;
    while (true) {
        if (settings.debug) {
            fit_iter++;
            std::cerr << "\n[FIT-ITER " << fit_iter << "]"
                    << " current_n_pulses=" << result.pulses.size()
                    << " current_final_nll=" << result.final_nll
                    << "\n";

            std::cerr << "[FIT-ITER-PULSES]";
            for (const auto& p : result.pulses) {
                std::cerr << " (t=" << p.time_us
                        << ", a=" << p.amplitude_pe << ")";
            }
            std::cerr << "\n";
        }

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

            if (settings.debug) {
                std::cerr << "[FIT-ITER-CLUSTER] cluster_index=" << cluster_index
                        << " left=" << bound.left_us
                        << " right=" << bound.right_us
                        << "\n";
            }

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
                    0.0, // allow zero amplitude for initial scan
                    settings.max_amplitude_pe,
                    init_amplitude
                );

                if (amplitude < settings.min_amplitude_pe) {
                    // reject candidate
                    continue;
                }

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
                
                if (settings.debug) {
                    std::cerr << "[TRY]"
                            << " iter=" << fit_iter
                            << " cluster=" << cluster_index
                            << " time_us=" << time_us
                            << " trial_amp=" << trial_amplitudes.back()
                            << " trial_nll=" << trial_nll
                            << " delta=" << delta
                            << "\n";
                }

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

        if (settings.debug) {
            std::cerr << "[FIT-ACCEPT]"
                    << " iter=" << fit_iter
                    << " accepted_time=" << best_time
                    << " accepted_amp=" << best_amplitude
                    << " accepted_delta=" << best_delta
                    << "\n";

            std::cerr << "[FIT-AFTER-REFIT]";
            for (const auto& p : result.pulses) {
                std::cerr << " (t=" << p.time_us
                        << ", a=" << p.amplitude_pe << ")";
            }
            std::cerr << "\n";
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
