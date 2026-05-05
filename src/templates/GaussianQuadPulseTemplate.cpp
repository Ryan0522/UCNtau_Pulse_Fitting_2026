#include "ucn/templates/GaussianQuadPulseTemplate.hpp"
#include <numeric>
#include <cmath>
#include <algorithm>
#include <vector>

namespace ucn
{
       
GaussianQuadPulseTemplate::GaussianQuadPulseTemplate(
double native_bin_width_us, double support_end_us, bool use_smooth_tail_onset,
    double baseline, double gauss_amp, double gauss_mu, double gauss_sigma,
    double tail_start_us, double tail_width_us,
    double a1, double tau1, double a2, double tau2, double a3, double tau3, double a4, double tau4)
    : native_bin_width_us_(native_bin_width_us), support_end_us_(support_end_us),
      use_smooth_tail_onset_(use_smooth_tail_onset),
      c_(baseline), Ag_(gauss_amp), mu_g_(gauss_mu), sigma_g_(gauss_sigma),
      t0_(tail_start_us), tail_width_us_(tail_width_us),
      A1_(a1), tau1_(tau1), A2_(a2), tau2_(tau2), 
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

double GaussianQuadPulseTemplate::tail_gate(double t) const {
    if (!use_smooth_tail_onset_) return (t >= t0_) ? 1.0 : 0.0;
    
    const double x = (t - t0_) / tail_width_us_;

    if (x > 50.0) return 1.0;
    if (x < -50.0) return 0.0;

    return 1.0 / (1.0 + std::exp(-x));
}

double GaussianQuadPulseTemplate::shape_unnormalized(double t) const {
    double val = c_ + Ag_ * std::exp(-0.5 * std::pow((t - mu_g_) / sigma_g_, 2));
    
    const double dt = std::max(t - t0_, 0.0);
    const double gate = tail_gate(t);
    
    val += gate * (
        A1_ * std::exp(-dt / tau1_) +
        A2_ * std::exp(-dt / tau2_) +
        A3_ * std::exp(-dt / tau3_) +
        A4_ * std::exp(-dt / tau4_)
    );
    return val;
}

double GaussianQuadPulseTemplate::numeric_integral(double t0_us, double t1_us) const {
    if (t1_us <= t0_us) return 0.0;

    const double lo = std::clamp(t0_us, 0.0, support_end_us_);
    const double hi = std::clamp(t1_us, 0.0, support_end_us_);
    if (hi <= lo) return 0.0;

    const double dx = std::min(0.01, 0.02 * native_bin_width_us_);
    const int n = std::max(1, static_cast<int>(std::ceil((hi - lo) / dx)));
    const double h = (hi - lo) / n;

    double sum = 0.0;
    for (int k = 0; k < n; ++k) {
        const double x_mid = lo + (k + 0.5) * h;
        sum += shape_unnormalized(x_mid) * h;
    }
    return sum;
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

double GaussianQuadPulseTemplate::raw_integral(double t0_us, double t1_us) const {
    if (use_smooth_tail_onset_) {
        return numeric_integral(t0_us, t1_us);
    }
    return analytic_integral(t1_us) - analytic_integral(t0_us);
}

double GaussianQuadPulseTemplate::integral(double t0_us, double t1_us) const {
    if (t1_us <= t0_us) return 0.0;

    const double lo = std::clamp(t0_us, 0.0, support_end_us_);
    const double hi = std::clamp(t1_us, 0.0, support_end_us_);
    if (hi <= lo) return 0.0;
    
    return raw_integral(lo, hi) / normalization_factor_;
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
