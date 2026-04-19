#pragma once
#include <vector>

namespace ucn
{

class PulseTemplate {
public:
    virtual ~PulseTemplate() = default;

    virtual double native_bin_width_us() const = 0;
    virtual std::vector<double> pmf() const = 0;
    virtual double integral(double t0_us, double t1_us) const = 0;

    virtual std::vector<double> shifted_to_histogram(
        double pulse_time_us,
        const std::vector<double>& bin_edges_us
    ) const = 0;
};
    
} // namespace ucn
