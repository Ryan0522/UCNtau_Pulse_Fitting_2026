#include "ucn/templates/GaussianQuadPulseTemplate.hpp"
#include <numeric>
#include <cmath>
#include <algorithm>
#include <vector>

namespace ucn
{
       
GaussianQuadPulseTemplate::GaussianQuadPulseTemplate(
double native_bin_width_us, double support_end_us,
    double baseline, double gauss_amp, double gauss_mu, double gauss_sigma,
    double tail_start_us,
    double a1, double tau1, double a2, double tau2, double a3, double tau3, double a4, double tau4)
    : native_bin_width_us_(native_bin_width_us), support_end_us_(support_end_us),
      c_(baseline), Ag_(gauss_amp), mu_g_(gauss_mu), sigma_g_(gauss_sigma),
      t0_(tail_start_us), A1_(a1), tau1_(tau1), A2_(a2), tau2_(tau2), 
      A3_(a3), tau3_(tau3), A4_(a4), tau4_(tau4)
{
    normalization_factor_ = analytic_integral(support_end_us_) - analytic_integral(0.0);
    
    size_t n_bins = static_cast<size_t>(support_end_us_ / native_bin_width_us_);
    pmf_unit_.resize(n_bins, 0.0);
    for (size_t i = 0; i < n_bins; ++i) {
        const double lo = static_cast<double>(i) * native_bin_width_us_;
        const double hi = std::min(lo + native_bin_width_us_, support_end_us_);
        pmf_unit_[i] = integral(lo, hi);
    }

    const double s = std::accumulate(pmf_unit_.begin(), pmf_unit_.end(), 0.0);
    for (double& v : pmf_unit_) {
        v /= s;
    }
}

double GaussianQuadPulseTemplate::shape_unnormalized(double t) const {
    double val = c_ + Ag_ * std::exp(-0.5 * std::pow((t - mu_g_) / sigma_g_, 2));
    if (t >= t0_) {
        val += A1_ * std::exp(-(t - t0_) / tau1_);
        val += A2_ * std::exp(-(t - t0_) / tau2_);
        val += A3_ * std::exp(-(t - t0_) / tau3_);
        val += A4_ * std::exp(-(t - t0_) / tau4_);
    }
    return val;
}

double GaussianQuadPulseTemplate::analytic_integral(double t) const {
    double term_c = c_ * t;
    double term_g = Ag_ * sigma_g_ * std::sqrt(M_PI / 2.0) * std::erf((t - mu_g_) / (sigma_g_ * std::sqrt(2.0)));
    double term_exps = 0.0;
    if (t > t0_) {
        term_exps += A1_ * tau1_ * (1.0 - std::exp(-(t - t0_) / tau1_));
        term_exps += A2_ * tau2_ * (1.0 - std::exp(-(t - t0_) / tau2_));
        term_exps += A3_ * tau3_ * (1.0 - std::exp(-(t - t0_) / tau3_));
        term_exps += A4_ * tau4_ * (1.0 - std::exp(-(t - t0_) / tau4_));
    }
    return term_c + term_g + term_exps;
}

double GaussianQuadPulseTemplate::integral(double t0_us, double t1_us) const {
    if (t1_us <= t0_us) return 0.0;

    const double lo = std::clamp(t0_us, 0.0, support_end_us_);
    const double hi = std::clamp(t1_us, 0.0, support_end_us_);
    if (hi <= lo) return 0.0;
    
    return (analytic_integral(hi) - analytic_integral(lo)) / normalization_factor_;
}

std::vector<double> GaussianQuadPulseTemplate::shifted_to_histogram(
    double pulse_time_us, const std::vector<double>& bin_edges_us) const
{
    std::vector<double> hist(bin_edges_us.size() - 1, 0.0);
    for (std::size_t i = 0; i < hist.size(); ++i) {
        const double rel_lo = bin_edges_us[i] - pulse_time_us;
        const double rel_hi = bin_edges_us[i + 1] - pulse_time_us;
        hist[i] = integral(rel_lo, rel_hi);
    }
    return hist;
}

} // namespace ucn
