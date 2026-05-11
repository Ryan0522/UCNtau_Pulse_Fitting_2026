#pragma once
#include <vector>

namespace ucn::likelihood
{
    
double poisson_nll(const std::vector<double>& observed,
                   const std::vector<double>& expected,
                   const std::vector<double>& fixed_expected,
                   double background_per_bin,
                   double eps = 1e-12);

std::vector<double> sum_expected(
    const std::vector<std::vector<double>>& components,
    const std::vector<double>& amplitudes
);

std::vector<double> add_background_and_fixed(
    const std::vector<double>& expected,
    const std::vector<double>& fixed_expected,
    double background_per_bin
);

double optimize_single_amplitude(
    const std::vector<double>& observed,
    const std::vector<double>& base_expected,
    const std::vector<double>& component,
    const std::vector<double>& fixed_expected,
    double background_per_bin,
    double min_amplitude_pe,
    double max_amplitude_pe,
    double initial_amplitude_pe
);

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
);

} // namespace ucn::likelihood
