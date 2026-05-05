#pragma once
#include "ucn/templates/PulseTemplates.hpp"
#include <string>
#include <vector>

namespace ucn
{

class EmpiricalPulseTemplate : public PulseTemplate {
public:
    EmpiricalPulseTemplate(
        double native_bin_width_us, double support_end_us, 
        const std::string& csv_path, double reference_time_us
    );

    double native_bin_width_us() const override { return native_bin_width_us_; }
    std::vector<double> pmf() const override { return pmf_unit_; }

    double integral(double t0_us, double t1_us) const override;

    std::vector<double> shifted_to_histogram(
        double pulse_time_us, const std::vector<double>& bin_edges_us
    ) const override;

private:
    double native_bin_width_us_ = 0.5;
    double support_end_us_ = 1000.0;
    double reference_time_us_ = 10.0;

    std::vector<double> pmf_unit_;
    std::vector<double> cdf_edges_;

    void load_csv(const std::string& csv_path);
    void normalize_and_build_cdf();

    double cdf_at(double t_us) const;
};
    
} // namespace ucn
