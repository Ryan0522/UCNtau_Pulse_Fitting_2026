#include "ucn/templates/DoubleExponentialPulseTemplate.hpp"

#include <algorithm>
#include <cmath>
#include<stdexcept>

namespace ucn
{

namespace {

std::vector<double> build_double_exp(
    double native_bin_width_us, 
    double start_offset_us,
    double fast_rate, 
    double fast_amplitude, 
    double slow_rate,
    double slow_amplitude, 
    double support_end_us
) {
    if (native_bin_width_us <= 0.0 || support_end_us <= 0.0) {
        throw std::invalid_argument("Double-exponential parameters must be positive.");
    }

    int n_bins = static_cast<int>(std::ceil(support_end_us / native_bin_width_us));
    if (n_bins < 1) {
        n_bins = 1;
    }

    std::vector<double> pmf(n_bins, 0.0);
    double total = 0.0;

    for (int i = 0; i < n_bins; ++i) {
        double t_us = i * native_bin_width_us;
        double value = fast_amplitude * std::exp(-fast_rate * t_us) + slow_amplitude * std::exp(-slow_rate * t_us);
        pmf[i] = value;
        total += value;
    }

    if (total <= 0.0) {
        throw std::runtime_error("Double-exponential template has non-positive mass.");
    }
    for (double& value : pmf) {
        value = value / total;
    }
    return pmf;
} 

} // namespace


DoubleExponentialPulseTemplate::DoubleExponentialPulseTemplate(
    double native_bin_width_us,
    double support_end_us,
    double fast_amplitude,
    double fast_rate,
    double slow_amplitude,
    double slow_rate,
    double start_offset_us
)
    : native_bin_width_us_(native_bin_width_us),
      support_end_us_(support_end_us),
      fast_amplitude_(fast_amplitude),
      fast_rate_(fast_rate),
      slow_amplitude_(slow_amplitude),
      slow_rate_(slow_rate),
      start_offset_us_(start_offset_us),
      pmf_unit_(build_double_exp(native_bin_width_us, start_offset_us,
                                 fast_rate, fast_amplitude, slow_rate,
                                 slow_amplitude, support_end_us)) {}

double DoubleExponentialPulseTemplate::native_bin_width_us() const {
    return native_bin_width_us_;
}

std::vector<double> DoubleExponentialPulseTemplate::pmf() const {
    return pmf_unit_;
}

double DoubleExponentialPulseTemplate::integral(double t0_us, double t1_us) const {
    if (t1_us <= t0_us) {
        return 0.0;
    }
    double integral_fast = fast_amplitude_ * (std::exp(-fast_rate_ * t0_us) - std::exp(-fast_rate_ * t1_us)) / fast_rate_;
    double integral_slow = slow_amplitude_ * (std::exp(-slow_rate_ * t0_us) - std::exp(-slow_rate_ * t1_us)) / slow_rate_;
    return integral_fast + integral_slow;
}

std::vector<double> DoubleExponentialPulseTemplate::shifted_to_histogram(
    double pulse_time_us,
    const std::vector<double>& bin_edges_us
) const {
    double template_start_time_us = pulse_time_us;

    if (bin_edges_us.size() < 2) {
        return std::vector<double>();
    }

    double hist_bin_width_us = bin_edges_us[1] - bin_edges_us[0];
    std::size_t n_bins = bin_edges_us.size() - 1;
    std::vector<double> expected(n_bins, 0.0);

    double start_fraction = (template_start_time_us - bin_edges_us[0]) / hist_bin_width_us;
    int start_bin = static_cast<int>(std::floor(start_fraction));
    double fraction = start_fraction - static_cast<double>(start_bin);

    for (std::size_t j = 0; j < pmf_unit_.size(); ++j) {
        int left = start_bin + static_cast<int>(j);
        int right = left + 1;

        if (left >= 0 && static_cast<std::size_t>(left) < n_bins) {
            expected[left] += pmf_unit_[j] * (1.0 - fraction);
        }
        if (right >= 0 && static_cast<std::size_t>(right) < n_bins) {
            expected[right] += pmf_unit_[j] * fraction;
        }
    }

    return expected;
}

} // namespace ucn
