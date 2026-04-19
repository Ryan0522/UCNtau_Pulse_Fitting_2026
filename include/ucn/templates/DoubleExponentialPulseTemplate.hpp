#pragma once
#include "ucn/templates/PulseTemplates.hpp"

namespace ucn
{
    
class DoubleExponentialPulseTemplate : public PulseTemplate {
public:
    DoubleExponentialPulseTemplate(
        double native_bin_width_us,
        double support_end_us,
        double fast_amplitude,
        double fast_rate,
        double slow_amplitude,
        double slow_rate,
        double start_offset_us
    );

    double native_bin_width_us() const override;

    std::vector<double> pmf() const override;
    double integral(double t0_us, double t1_us) const override;

    std::vector<double> shifted_to_histogram(
        double pulse_time_us,
        const std::vector<double>& bin_edges_us
    ) const override;

private:
    double native_bin_width_us_ = 0.5;
    double support_end_us_ = 100.0;

    double fast_amplitude_ = 0.0;
    double fast_rate_ = 0.0;
    double slow_amplitude_ = 0.0;
    double slow_rate_ = 0.0;
    double start_offset_us_ = 0.0;

    double normalization_sum_ = 1.0;
    std::vector<double> pmf_unit_;

    double shape_unnormalized(double t_us) const;
    double normalization_integral() const;
};

} // namespace ucn
