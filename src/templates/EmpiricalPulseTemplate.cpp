#include "ucn/templates/EmpiricalPulseTemplate.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ucn {

EmpiricalPulseTemplate::EmpiricalPulseTemplate(
    double native_bin_width_us, double support_end_us, const std::string& csv_path
) : native_bin_width_us_(native_bin_width_us), support_end_us_(support_end_us) {
    load_csv(csv_path);
    normalize_and_build_cdf();
}

void EmpiricalPulseTemplate::load_csv(const std::string& csv_path) {
    std::ifstream in(csv_path);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open empirical pulse template CSV: " + csv_path);
    }

    std::string line;
    if (!std::getline(in, line)) {
        throw std::runtime_error("Empirical pulse template CSV is empty: " + csv_path);
    }
    std::vector<double> values;
    while (std::getline(in, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string col0, col1;

        if (!std::getline(ss, col0, ',')) continue;
        if (!std::getline(ss, col1, ',')) continue;

        const double p = std::stod(col1);
        if (std::isfinite(p)) {
            values.push_back(std::max(0.0, p));
        }
    }

    if (values.empty()) {
        throw std::runtime_error("Empirical pulse template CSV had no PMF values: " + csv_path);
    }

    const std::size_t n_support = static_cast<std::size_t>(
        std::ceil(support_end_us_ / native_bin_width_us_)
    );

    pmf_unit_.assign(n_support, 0.0);

    const std::size_t n_copy = std::min(n_support, values.size());
    for (std::size_t i = 0; i < n_copy; ++i) {
        pmf_unit_[i] = values[i];
    }
}

void EmpiricalPulseTemplate::normalize_and_build_cdf() {
    double s = 0.0;
    for (double v : pmf_unit_) {
        s += v;
    }

    if (!(s > 0.0) || !std::isfinite(s)) {
        throw std::runtime_error("Empirical pulse template PMF sum is not positive.");
    }

    for (double& v : pmf_unit_) {
        v /= s;
    }

    cdf_edges_.assign(pmf_unit_.size() + 1, 0.0);
    for (std::size_t i = 0; i < pmf_unit_.size(); ++i) {
        cdf_edges_[i + 1] = cdf_edges_[i] + pmf_unit_[i];
    }
    cdf_edges_.back() = 1.0;
}

double EmpiricalPulseTemplate::cdf_at(double t_us) const {
    if (t_us <= 0.0) return 0.0;
    if (t_us >= support_end_us_) return 1.0;

    const double x = t_us / native_bin_width_us_;
    const std::size_t i = static_cast<std::size_t>(std::floor(x));

    if (i >= pmf_unit_.size()) return 1.0;

    const double frac = x - static_cast<double>(i);

    // Linear interpolation inside the native PMF bin.
    return cdf_edges_[i] + frac * pmf_unit_[i];
}

double EmpiricalPulseTemplate::integral(double t0_us, double t1_us) const {
    if (t1_us <= t0_us) return 0.0;

    const double lo = std::clamp(t0_us, 0.0, support_end_us_);
    const double hi = std::clamp(t1_us, 0.0, support_end_us_);

    if (hi <= lo) return 0.0;

    return cdf_at(hi) - cdf_at(lo);
}

std::vector<double> EmpiricalPulseTemplate::shifted_to_histogram(
    double pulse_time_us,
    const std::vector<double>& bin_edges_us
) const {
    if (bin_edges_us.size() < 2) {
        return {};
    }

    std::vector<double> out(bin_edges_us.size() - 1, 0.0);

    for (std::size_t i = 0; i + 1 < bin_edges_us.size(); ++i) {
        const double rel_lo = bin_edges_us[i]     - pulse_time_us;
        const double rel_hi = bin_edges_us[i + 1] - pulse_time_us;
        out[i] = integral(rel_lo, rel_hi);
    }

    return out;
}


} // namespace ucn