#include "ucn/inference/Likelihood.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ucn::likelihood 
{

std::vector<double> add_background_and_fixed(
     const std::vector<double>& expected,
     const std::vector<double>& fixed_expected,
     double background_per_bin
) {
    std::vector<double> total = expected;
    if (!fixed_expected.empty()) {
        if (fixed_expected.size() != expected.size()) {
            throw std::invalid_argument("Expected and fixed expected histograms must have the same size.");
        }
        for (std::size_t i = 0; i < total.size(); ++i) {
            total[i] += std::max(0.0, fixed_expected[i]);
        }
    }
    if (background_per_bin > 0.0) {
        for (double& value : total) {
            value += background_per_bin;
        }
    }
    return total;
}

double poisson_nll(
    const std::vector<double>& observed,
    const std::vector<double>& expected,
    const std::vector<double>& fixed_expected,
    double background_per_bin,
    double eps
) {
    if (observed.size() != expected.size()) {
        throw std::invalid_argument("Observed and expected histograms must have the same size.");
    }

    std::vector<double> total = add_background_and_fixed(expected, fixed_expected, background_per_bin);

    double nll = 0.0;
    for (std::size_t i = 0; i < observed.size(); ++i) {
        double mu = std::max(total[i], eps);
        nll += mu - observed[i] * std::log(mu);
    }
    return nll;
}

std::vector<double> sum_expected(
    const std::vector<std::vector<double>>& components,
    const std::vector<double>& amplitudes
) {
    if (components.empty()) {
        return std::vector<double>();
    }
    if (components.size() != amplitudes.size()) {
        throw std::invalid_argument("components and amplitudes must have the same size.");
    }

    std::vector<double> total(components[0].size(), 0.0);
    for (std::size_t j = 0; j < components.size(); ++j) {
        if (components[j].size() != total.size()) {
            throw std::invalid_argument("All components must have the same size.");
        }
        for (std::size_t i = 0; i < total.size(); ++i) {
            total[i] += amplitudes[j] * components[j][i];
        }
    }
    return total;
}


double optimize_single_amplitude(
    const std::vector<double>& observed,
    const std::vector<double>& base_expected,
    const std::vector<double>& component,
    const std::vector<double>& fixed_expected,
    double background_per_bin,
    double min_amplitude_pe,
    double max_amplitude_pe,
    double initial_amplitude_pe
) {
    if (observed.size() != base_expected.size() || observed.size() != component.size()) {
        throw std::invalid_argument("Histogram vectors must all have the same size.");
    }

    double lower = std::max(0.0, min_amplitude_pe);
    double upper = std::max(lower, max_amplitude_pe);
    double a = std::clamp(initial_amplitude_pe, lower, upper);

    for (int iter = 0; iter < 30; ++iter) {
        double grad = 0.0;
        double hess = 0.0;

        for (std::size_t i = 0; i < observed.size(); ++i) {
            double mu = base_expected[i] + a * component[i];
            if (!fixed_expected.empty()) {
                mu += std::max(0.0, fixed_expected[i]);
            }
            mu += std::max(0.0, background_per_bin);
            mu = std::max(mu, 1e-12);

            grad += component[i] * (1.0 - observed[i] / mu);
            hess += component[i] * component[i] * observed[i] / (mu * mu);
        }

        if (std::abs(grad) < 1e-10) {
            break;
        }
        if (hess <= 1e-12) {
            break;
        }

        double candidate = a - grad / hess;
        candidate = std::clamp(candidate, lower, upper);
        if (std::abs(candidate - a) < 1e-10) {
            a = candidate;
            break;
        }
        a = candidate;
    }

    return a;
}

std::vector<double> refit_all_amplitudes(
    const std::vector<double>& observed,
    const std::vector<std::vector<double>>& components,
    const std::vector<double>& initial_amplitudes,
    const std::vector<double>& fixed_expected,
    double background_per_bin,
    double min_amplitude_pe,
    double max_amplitude_pe,
    int max_steps,
    double tolerance
) {
    if (components.size() != initial_amplitudes.size()) {
        throw std::invalid_argument("components and initial_amplitudes must have the same size.");
    }
    if (components.empty()) {
        return std::vector<double>();
    }

    std::vector<double> amplitudes = initial_amplitudes;
    for (std::size_t j = 0; j < amplitudes.size(); ++j) {
        amplitudes[j] = std::clamp(amplitudes[j], min_amplitude_pe, max_amplitude_pe);
    }

    for (int step = 0; step < max_steps; ++step) {
        bool changed = false;
        double max_change = 0.0;
        for (std::size_t j = 0; j < amplitudes.size(); ++j) {
            std::vector<double> base(observed.size(), 0.0);
            for (std::size_t k = 0; k < amplitudes.size(); ++k) {
                if (k == j) {
                    continue;
                }
                for (std::size_t i = 0; i < base.size(); ++i) {
                    base[i] += amplitudes[k] * components[k][i];
                }
            }
            double updated = optimize_single_amplitude(
                observed,
                base,
                components[j],
                fixed_expected,
                background_per_bin,
                min_amplitude_pe,
                max_amplitude_pe,
                amplitudes[j]
            );
            max_change = std::max(max_change, std::abs(updated - amplitudes[j]));
            if (max_change < tolerance) {
                break;
            }
            if (std::abs(updated - amplitudes[j]) > 1e-9) {
                amplitudes[j] = updated;
                changed = true;
            }
        }
        if (!changed) {
            break;
        }
    }

    return amplitudes;
}

}