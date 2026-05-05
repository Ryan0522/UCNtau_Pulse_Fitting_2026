#pragma once
#include "ucn/templates/PulseTemplates.hpp"
#include <cmath>
#include <vector>

namespace ucn
{

class GaussianQuadPulseTemplate : public PulseTemplate {
public:
    GaussianQuadPulseTemplate(
        double native_bin_width_us,
        double support_end_us,
        bool use_smooth_tail_onset,
        double baseline,
        double gauss_amp, double gauss_mu, double gauss_sigma,
        double tail_start_us, double tail_width_us,
        double a1, double tau1,
        double a2, double tau2,
        double a3, double tau3,
        double a4, double tau4
    );

    double native_bin_width_us() const override { return native_bin_width_us_; }
    std::vector<double> pmf() const override { return pmf_unit_; }
    
    double integral(double t0_us, double t1_us) const override;

    std::vector<double> shifted_to_histogram(
        double pulse_time_us,
        const std::vector<double>& bin_edges_us
    ) const override;

private:
    double native_bin_width_us_;
    double support_end_us_;
    bool use_smooth_tail_onset_ = false;

    double c_, Ag_, mu_g_, sigma_g_;
    double t0_, tail_width_us_ = 1.0e-6;
    double A1_, tau1_, A2_, tau2_, A3_, tau3_, A4_, tau4_;
    
    double normalization_factor_ = 1.0;
    std::vector<double> pmf_unit_;

    double tail_gate(double t) const;
    double shape_unnormalized(double t_us) const;
    double numeric_integral(double t0_us, double t1_us) const;
    double analytic_integral(double t) const;
    double raw_integral(double t0_us, double t1_us) const;
};
    
} // namespace ucn
