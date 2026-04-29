#include "ucn/inference/CoincidenceFitter.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>

namespace {

std::uint64_t mix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

double deterministic_uniform01(std::uint64_t seed,
                               double start_time_us,
                               int start_channel,
                               int n_pe) {
    const auto tkey = static_cast<std::uint64_t>(
        std::llround(start_time_us * 1000.0)
    );

    std::uint64_t x = seed;
    x ^= mix64(tkey);
    x ^= mix64(static_cast<std::uint64_t>(start_channel + 1024));
    x ^= mix64(static_cast<std::uint64_t>(n_pe + 4096));

    x = mix64(x);

    // 53 random mantissa bits -> [0, 1)
    return static_cast<double>(x >> 11) * (1.0 / 9007199254740992.0);
}

void update_free_pe_interval(double free_time_us,
                             double& last_free_time_us,
                             std::size_t& n_free,
                             double& free_pe_interval_us) {
    if (n_free == 0) {
        last_free_time_us = free_time_us;
        n_free = 1;
        return;
    }

    const double dt_us = free_time_us - last_free_time_us;
    last_free_time_us = free_time_us;

    if (dt_us <= 0.0) {
        ++n_free;
        return;
    }

    if (n_free == 1) {
        free_pe_interval_us = dt_us;
    } else if (n_free < 100) {
        const double nf = static_cast<double>(n_free);
        free_pe_interval_us =
            ((nf - 1.0) / nf) * free_pe_interval_us + (1.0 / nf) * dt_us;
    } else {
        free_pe_interval_us = 0.99 * free_pe_interval_us + 0.01 * dt_us;
    }

    ++n_free;
}

} // namespace

namespace ucn {
    
CoincidenceFitter::CoincidenceFitter(CoincidenceSettings settings)
    : settings_(settings) {}

CoincidenceFitResult CoincidenceFitter::find(const std::vector<Hit>& input_hits) const {
    CoincidenceFitResult out;

    if (input_hits.empty()) return out;

    std::vector<Hit> hits = input_hits;
    std::sort(hits.begin(), hits.end(), [](const Hit& a, const Hit& b) {
        return a.time_us < b.time_us;
    });

    const double wc = settings_.coincidence_window_us;
    const double wt = settings_.telescoping_window_us;
    const int threshold = settings_.pe_threshold;

    double last_free_time_us = 0.0;
    double free_pe_interval_us = 0.0;
    std::size_t n_free = 0;

    std::size_t i = 0;
    while (i < hits.size()) {
        int n_pe = 1;
        bool found_other_channel = false;
        bool raw_candidate = false;

        std::size_t j = i + 1;

        while (j < hits.size()) {
            if (!found_other_channel) {
                const double dt_from_start = hits[j].time_us - hits[i].time_us;

                if (dt_from_start > wc) {
                    break;
                }

                ++n_pe;

                if (hits[j].channel != hits[i].channel) {
                    found_other_channel = true;
                }

                ++j;
                continue;
            }

            const double gap_from_previous = hits[j].time_us - hits[j - 1].time_us;

            if (gap_from_previous >= wt) {
                raw_candidate = (n_pe >= threshold);
                break;
            }

            ++n_pe;
            ++j;
        }

        if (found_other_channel && j == hits.size() && n_pe >= threshold) {
            raw_candidate = true;
        }

        if (raw_candidate) {
            const std::size_t last_in_event =
                (j == hits.size()) ? (hits.size() - 1) : (j - 1);

            const double start_time_us = hits[i].time_us;
            const double end_time_us = hits[last_in_event].time_us + wt;
            const double length_us = end_time_us - start_time_us;

            double n_pileup = static_cast<double>(n_pe);

            if (settings_.apply_pileup_correction && free_pe_interval_us > 0.0) {
                const double rn = deterministic_uniform01(
                    settings_.rng_seed,
                    start_time_us,
                    hits[i].channel,
                    n_pe
                );

                n_pileup =
                    static_cast<double>(n_pe) - length_us / free_pe_interval_us + rn;
            } 

            CoincidenceEvent ev;
            ev.start_time_us = start_time_us;
            ev.end_time_us = end_time_us;
            ev.length_us = length_us;
            ev.start_channel = hits[i].channel;
            ev.n_pe = n_pe;
            ev.n_pileup = n_pileup;
            ev.free_pe_interval_us = free_pe_interval_us;
            ev.passes_raw_threshold = (n_pe >= threshold);
            ev.passes_pileup_threshold = (n_pileup >= static_cast<double>(threshold));

            out.events.push_back(ev);

            i = j;
        } else {
            update_free_pe_interval(
                hits[i].time_us,
                last_free_time_us,
                n_free,
                free_pe_interval_us
            );

            ++i;
        }
    }

    if (!out.events.empty()) {
        double sum_length = 0.0;
        double sum_npe = 0.0;

        for (const auto& ev : out.events) {
            sum_length += ev.length_us;
            sum_npe += static_cast<double>(ev.n_pe);
        }

        out.average_length_us = sum_length / static_cast<double>(out.events.size());
        out.average_n_pe = sum_npe / static_cast<double>(out.events.size());
    }

    return out;
}

} // namespace ucn