#include "ucn/debug/DebugCsvWriter.hpp"

#include <iomanip>
#include <stdexcept>
#include <cmath>

namespace fs = std::filesystem;

namespace ucn::debug {

namespace {

int hold_key(double hold_time_s) {
    return static_cast<int>(std::llround(hold_time_s));
}

void write_lrt_row(std::ofstream& out, const LRTTrialDebug& r) {
    out << r.case_id << ','
        << r.status << ','
        << r.fit_iter << ','
        << r.cluster_index << ','
        << std::setprecision(17)
        << r.trial_time_us << ','
        << r.trial_amp << ','
        << r.nll_before << ','
        << r.nll_after << ','
        << r.delta_nll << ','
        << r.penalty_nll << ','
        << r.required_delta_nll << ','
        << r.local_delta_nll << ','
        << r.local_pass << ','
        << r.margin << ','
        << r.accepted << '\n';
}

} // namespace

DebugCsvWriter::DebugCsvWriter(
    const fs::path& out_dir,
    int max_per_case,
    const PulseTemplate& pulse_template
)
    : max_per_case_(max_per_case), pulse_template_(pulse_template)
{
    enabled_ = max_per_case_ > 0;
    if (!enabled_) return;
    
    fs::create_directories(out_dir);

    manifest_.open(out_dir / "manifest.csv");
    bins_.open(out_dir / "bins.csv");
    markers_.open(out_dir / "markers.csv");
    components_.open(out_dir / "components.csv");
    lrt_trials_.open(out_dir / "lrt_trials.csv");
    lrt_accepts_.open(out_dir / "lrt_accepts.csv");

    if (!manifest_ || !bins_ || !markers_ || !components_ || !lrt_trials_ || !lrt_accepts_) {
        throw std::runtime_error("Could not open one or more debug CSV files.");
    }

    write_headers();
}

void DebugCsvWriter::write_headers() {
    manifest_ << "case_id,case_type,region,hold_time_s,chunk_index,local_window_index,global_window_index,"
              << "start_time_us,end_time_us,model_end_time_us,bin_width_us,n_bins,"
              << "observed_count,expected_count,seed_count,truth_count,pulse_count,final_nll\n";

    bins_ << "case_id,bin_index,left_us,center_us,right_us,"
          << "observed,fixed_expected,background_expected,fit_expected,total_expected,residual\n";

    markers_ << "case_id,marker_type,marker_index,time_us,amplitude_pe\n";

    components_ << "case_id,pulse_index,bin_index,center_us,amplitude_pe,"
                << "unit_template,component_expected\n";

    lrt_trials_ << "case_id,status,fit_iter,cluster_index,trial_time_us,trial_amp,"
                << "nll_before,nll_after,delta_nll,penalty_nll,required_delta_nll,"
                << "local_delta_nll,local_pass,margin,accepted\n";

    lrt_accepts_ << "case_id,status,fit_iter,cluster_index,trial_time_us,trial_amp,"
                 << "nll_before,nll_after,delta_nll,penalty_nll,required_delta_nll,"
                 << "local_delta_nll,local_pass,margin,accepted\n";
}

bool DebugCsvWriter::can_capture(DebugCaseType type, double hold_time_s) const {
    if (!enabled_) return false;

    const auto key = std::make_tuple(type, hold_key(hold_time_s));
    auto it = counts_.find(key);
    const int n = (it == counts_.end()) ? 0 : it->second;
    return n < max_per_case_;
}

std::string DebugCsvWriter::next_case_id(DebugCaseType type, double hold_time_s) {
    const int hk = hold_key(hold_time_s);
    const auto key = std::make_tuple(type, hk);
    const int n = counts_[key]++;
    std::ostringstream ss;
    ss << "ht" << hk << "_"
       << case_name(type) << "_"
       << std::setw(3) << std::setfill('0') << n;
    return ss.str();
}

void DebugCsvWriter::write_window(
    const WindowDebugContext& ctx,
    const Histogram& histogram,
    const FitResult& fit,
    const std::vector<double>& fixed_expected,
    double background_per_bin
) {
    if (!enabled_) return;

    const int n_bins = static_cast<int>(histogram.counts.size());
    
    double observed_count = 0.0;
    for (double c : histogram.counts) observed_count += c;

    double expected_count = 0.0;
    for (double e : fit.expected_total) expected_count += e;

    manifest_ << ctx.case_id << ','
              << case_name(ctx.case_type) << ','
              << ctx.region << ','
              << ctx.hold_time_s << ','
              << ctx.chunk_index << ','
              << ctx.local_window_index << ','
              << ctx.global_window_index << ','
              << std::setprecision(17)
              << ctx.start_time_us << ','
              << ctx.end_time_us << ','
              << ctx.model_end_time_us << ','
              << ctx.bin_width_us << ','
              << n_bins << ','
              << observed_count << ','
              << expected_count << ','
              << ctx.seeds.size() << ','
              << ctx.truth_pulses.size() << ','
              << fit.pulses.size() << ','
              << fit.final_nll << '\n';

    for (int i = 0; i < n_bins; ++i) {
        const double left = histogram.bin_edges_us[i];
        const double right = histogram.bin_edges_us[i + 1];
        const double center = 0.5 * (left + right);

        const double obs = histogram.counts[i];
        const double fixed = (i < static_cast<int>(fixed_expected.size())) ? fixed_expected[i] : 0.0;
        const double fit_exp = (i < static_cast<int>(fit.expected_total.size())) ? fit.expected_total[i] : 0.0;
        const double bg = background_per_bin;
        const double total = fixed + bg + fit_exp;
        const double resid = obs - total;

        bins_ << ctx.case_id << ','
              << i << ','
              << std::setprecision(17)
              << left << ','
              << center << ','
              << right << ','
              << obs << ','
              << fixed << ','
              << bg << ','
              << fit_exp << ','
              << total << ','
              << resid << '\n';
    }

    for (std::size_t i = 0; i < ctx.seeds.size(); ++i) {
        markers_ << ctx.case_id << ",seed," << i << ','
                 << std::setprecision(17) << ctx.seeds[i] << ",\n";
    }

    for (std::size_t i = 0; i < ctx.truth_pulses.size(); ++i) {
        markers_ << ctx.case_id << ",truth," << i << ','
                 << std::setprecision(17)
                 << ctx.truth_pulses[i].time_us << ','
                 << ctx.truth_pulses[i].amplitude_pe << '\n';
    }

    for (std::size_t p = 0; p < fit.pulses.size(); ++p) {
        const auto& pulse = fit.pulses[p];

        markers_ << ctx.case_id << ",fit," << p << ','
                 << std::setprecision(17)
                 << pulse.time_us << ','
                 << pulse.amplitude_pe << '\n';

        const auto unit = pulse_template_.shifted_to_histogram(
            pulse.time_us,
            histogram.bin_edges_us
        );

        for (int b = 0; b < n_bins; ++b) {
            const double left = histogram.bin_edges_us[b];
            const double right = histogram.bin_edges_us[b + 1];
            const double center = 0.5 * (left + right);
            const double u = unit[b];
            const double comp = pulse.amplitude_pe * u;

            components_ << ctx.case_id << ','
                        << p << ','
                        << b << ','
                        << std::setprecision(17)
                        << center << ','
                        << pulse.amplitude_pe << ','
                        << u << ','
                        << comp << '\n';
        }
    }
}

void DebugCsvWriter::on_lrt_trial(const LRTTrialDebug& r) {
    if (!enabled_) return;
    write_lrt_row(lrt_trials_, r);
}

void DebugCsvWriter::on_lrt_accept(const LRTTrialDebug& r) {
    if (!enabled_) return;
    write_lrt_row(lrt_accepts_, r);
}

bool DebugCsvWriter::can_capture_any_observed(double hold_time_s) const {
    return can_capture(DebugCaseType::ObservedOneSeedOneFit, hold_time_s) ||
           can_capture(DebugCaseType::ObservedOneSeedMultiFit, hold_time_s) ||
           can_capture(DebugCaseType::ObservedMultiSeed, hold_time_s);
}

void DebugCsvWriter::write_lrt_trials_for_case(
    const std::string& case_id,
    const std::vector<LRTTrialDebug>& rows
) {
    if (!enabled_) return;

    for (LRTTrialDebug row : rows) {
        row.case_id = case_id;
        write_lrt_row(lrt_trials_, row);
    }
}

void DebugCsvWriter::write_lrt_accepts_for_case(
    const std::string& case_id,
    const std::vector<LRTTrialDebug>& rows
) {
    if (!enabled_) return;

    for (LRTTrialDebug row : rows) {
        row.case_id = case_id;
        write_lrt_row(lrt_accepts_, row);
    }
}

} // namespace ucn::debug