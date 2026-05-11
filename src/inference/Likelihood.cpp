#include "ucn/inference/Likelihood.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <limits>

namespace ucn::likelihood 
{

namespace 
{
    
double nll_for_amplitudes(
    const std::vector<double>& observed,
    const std::vector<std::vector<double>>& components,
    const std::vector<double>& amplitudes,
    const std::vector<double>& fixed_expected,
    double background_per_bin
) {
    const std::size_t n_bins = observed.size();
    std::vector<double> mu(n_bins, std::max(0.0, background_per_bin));

    if (!fixed_expected.empty()) {
        for (std::size_t i = 0; i < n_bins; ++i) {
            mu[i] += std::max(0.0, fixed_expected[i]);
        }
    }

    for (std::size_t j = 0; j < components.size(); ++j) {
        for (std::size_t i = 0; i < n_bins; ++i) {
            mu[i] += amplitudes[j] * components[j][i];
        }
    }
    
    double nll = 0.0;
    for (std::size_t i = 0; i < n_bins; ++i) {
        const double m = std::max(mu[i], 1.0e-12);
        nll += m - observed[i] * std::log(m);
    }
    return nll;
}

bool solve_dense_linear_system(
    std::vector<std::vector<double>> A,
    std::vector<double> b,
    std::vector<double>& x
) {
    const std::size_t n = b.size();
    x.assign(n, 0.0);
    if (A.size() != n) return false;
    for (const auto& row : A) {
        if (row.size() != n) return false;
    }

    for (std::size_t col = 0; col < n; ++col) {
        std::size_t pivot = col;
        double best = std::abs(A[col][col]);
        for (std::size_t row = col + 1; row < n; ++row) {
            const double v = std::abs(A[row][col]);
            if (v > best) {
                best = v;
                pivot = row;
            }
        }

        if (best < 1.0e-14 || !std::isfinite(best)) return false;

        if (pivot != col) {
            std::swap(A[pivot], A[col]);
            std::swap(b[pivot], b[col]);
        }

        const double diag = A[col][col];
        for (std::size_t row = col + 1; row < n; ++row) {
            const double factor = A[row][col] / diag;
            if (factor == 0.0) continue;
            A[row][col] = 0.0;
            for (std::size_t k = col + 1; k < n; ++k) {
                A[row][k] -= factor * A[col][k];
            }
            b[row] -= factor * b[col];
        }
    }

    for (std::size_t rr = 0; rr < n; ++rr) {
        const std::size_t row = n - 1 - rr;
        double rhs = b[row];
        for (std::size_t k = row + 1; k < n; ++k) {
            rhs -= A[row][k] * x[k];
        }
        const double diag = A[row][row];
        if (std::abs(diag) < 1.0e-14 || !std::isfinite(diag)) {
            return false;
        }
        x[row] = rhs / diag;
    }
    return true;
}

} // namespace 


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
    if (!fixed_expected.empty() && fixed_expected.size() != observed.size()) {
        throw std::invalid_argument("fixed_expected must match observed size.");
    }

    const std::size_t n_pulses = components.size();
    const std::size_t n_bins = observed.size();
    
    const double lower = std::max(0.0, min_amplitude_pe);
    const double upper = std::max(lower, max_amplitude_pe);
    const double tol = std::max(tolerance, 1.0e-12);

    std::vector<double> amplitudes = initial_amplitudes;
    for (double& a : amplitudes) {
        a = std::clamp(a, lower, upper);
    }

    for (int iter = 0; iter < max_steps; ++iter) {
        std::vector<double> mu(n_bins, std::max(0.0, background_per_bin));
        if (!fixed_expected.empty()) {
            for (std::size_t i = 0; i < n_bins; ++i) {
                mu[i] += std::max(0.0, fixed_expected[i]);
            }
        }
        for (std::size_t j = 0; j < n_pulses; ++j) {
            for (std::size_t i = 0; i < n_bins; ++i) {
                mu[i] += amplitudes[j] * components[j][i];
            }
        }
        for (double& m : mu) {
            m = std::max(m, 1.0e-12);
        }

        std::vector<double> grad(n_pulses, 0.0);
        std::vector<std::vector<double>> hess(n_pulses, std::vector<double>(n_pulses, 0.0));

        for (std::size_t i = 0; i < n_bins; ++i) {
            const double inv_mu = 1.0 / mu[i];
            const double obs_over_mu = observed[i] * inv_mu;
            const double h_weight = observed[i] * inv_mu * inv_mu;

            for (std::size_t j = 0; j < n_pulses; ++j) {
                const double sj = components[j][i];
                grad[j] += sj * (1.0 - obs_over_mu);

                for (std::size_t k = 0; k <= j; ++k) {
                    hess[j][k] += h_weight * sj * components[k][i];
                }
            }
        }
        
        for (std::size_t j = 0; j < n_pulses; ++j) {
            for (std::size_t k = 0; k < j; ++k) {
                hess[k][j] = hess[j][k];
            }
        }

        std::vector<std::size_t> active;
        active.reserve(n_pulses);
        double max_kkt = 0.0;

        for (std::size_t j = 0; j < n_pulses; ++j) {
            const bool at_lower = amplitudes[j] <= lower + tol;
            const bool at_upper = amplitudes[j] >= upper - tol;

            double violation = 0.0;
            if (at_lower) {
                violation = std::max(0.0, -grad[j]);
            } else if (at_upper) {
                violation = std::max(0.0, grad[j]);
            } else {
                violation = std::abs(grad[j]);
            }
            max_kkt = std::max(max_kkt, violation);

            const bool blocked_at_lower = at_lower && grad[j] >= 0.0;
            const bool blocked_at_upper = at_upper && grad[j] <= 0.0;
            if (!blocked_at_lower && !blocked_at_upper) {
                active.push_back(j);
            }
        }

        if (max_kkt < tol || active.empty()) break;

        const std::size_t n_active = active.size();
        std::vector<std::vector<double>> H(
            n_active, std::vector<double>(n_active, 0.0)
        );
        std::vector<double> g(n_active, 0.0);

        for (std::size_t r = 0; r < n_active; ++r) {
            const std::size_t jr = active[r];
            g[r] = grad[jr];
            for (std::size_t c = 0; c < n_active; ++c) {
                const std::size_t jc = active[c];
                H[r][c] = hess[jr][jc];
            }
            H[r][r] += 1.0e-10 * (std::abs(H[r][r]) + 1.0);
        }

        std::vector<double> step;
        if (!solve_dense_linear_system(H, g, step)) {
            step.assign(n_active, 0.0);
            for (std::size_t r = 0; r < n_active; ++r) {
                const double diag = std::max(std::abs(H[r][r]), 1.0e-12);
                step[r] = g[r] / diag;
            }
        }

        const double nll_current = nll_for_amplitudes(
            observed, components, amplitudes, fixed_expected, background_per_bin
        );

        bool accepted = false;
        double max_change = 0.0;
        std::vector<double> best = amplitudes;

        double alpha = 1.0;
        for (int ls = 0; ls < 30; ++ls) {
            std::vector<double> trial = amplitudes;
            max_change = 0.0;
            for (std::size_t r = 0; r < n_active; ++r) {
                const std::size_t j = active[r];
                const double updated = std::clamp(
                    amplitudes[j] - alpha * step[r], lower, upper
                );
                max_change = std::max(max_change, std::abs(updated - amplitudes[j]));
                trial[j] = updated;
            }

            const double nll_trial = nll_for_amplitudes(
                observed, components, trial, fixed_expected, background_per_bin
            );

            if (std::isfinite(nll_trial) && nll_trial <= nll_current) {
                best = std::move(trial);
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }

        if (!accepted) {
            break;
        }

        amplitudes = std::move(best);
        if (max_change < tol) {
            break;
        }
    }

    return amplitudes;
}

}