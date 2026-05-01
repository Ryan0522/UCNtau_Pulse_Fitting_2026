#include "ucn/io/RootRunLoader.hpp"

#include <TFile.h>
#include <TTree.h>

#include <algorithm>
#include <filesystem>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace ucn::io {

namespace {

struct RootEvent {
    Int_t channel = 0;
    Int_t edge = 0;
    Int_t tag = 0;
    Int_t full = 0;
    ULong64_t time = 0;
    Double_t realtime = 0.0; // seconds
};

std::string run_to_string(int run_number) {
    return std::to_string(run_number);
}

fs::path build_root_path(const AnalysisConfig& cfg, int run_number) {
    return fs::path(cfg.data_folder) /
           (cfg.root_filename_prefix + run_to_string(run_number) + cfg.root_filename_suffix);
}

void sort_hits(std::vector<Hit>& hits) {
    std::sort(hits.begin(), hits.end(),
              [](const Hit& a, const Hit& b) { return a.time_us < b.time_us; });
}

std::optional<json> find_run_record(const json& runinfo_json, const std::string& run_id) {
    if (runinfo_json.is_null() || runinfo_json.empty()) {
        return std::nullopt;
    }

    if (runinfo_json.is_object()) {
        if (runinfo_json.contains(run_id)) {
            return runinfo_json.at(run_id);
        }

        // fallback: search values for a matching run field
        for (auto it = runinfo_json.begin(); it != runinfo_json.end(); ++it) {
            if (!it.value().is_object()) {
                continue;
            }
            const json& rec = it.value();
            if ((rec.contains("run") && rec.at("run").dump() == run_id) ||
                (rec.contains("runnum") && rec.at("runnum").dump() == run_id) ||
                (rec.contains("run_number") && rec.at("run_number").dump() == run_id)) {
                return rec;
            }
        }
    }

    if (runinfo_json.is_array()) {
        for (const auto& rec : runinfo_json) {
            if (!rec.is_object()) {
                continue;
            }

            if ((rec.contains("run") && std::to_string(rec.at("run").get<int>()) == run_id) ||
                (rec.contains("runnum") && std::to_string(rec.at("runnum").get<int>()) == run_id) ||
                (rec.contains("run_number") && std::to_string(rec.at("run_number").get<int>()) == run_id)) {
                return rec;
            }
        }
    }

    return std::nullopt;
}

double pick_number_with_fallback(const json& rec,
                                 std::initializer_list<const char*> keys,
                                 double fallback) {
    for (const char* key : keys) {
        if (rec.contains(key) && rec.at(key).is_number()) {
            return rec.at(key).get<double>();
        }
    }
    return fallback;
}

double maybe_convert_to_us(double value, bool input_is_seconds) {
    return input_is_seconds ? value * 1.0e6 : value;
}

void read_tree_into_segments(TTree* tree,
                             const std::map<int, std::string>& channel_to_segment,
                             std::map<std::string, std::vector<Hit>>& segments) {
    if (tree == nullptr) {
        return;
    }

    RootEvent evt;
    tree->SetBranchAddress("channel", &evt.channel);
    tree->SetBranchAddress("edge", &evt.edge);
    tree->SetBranchAddress("tag", &evt.tag);
    tree->SetBranchAddress("full", &evt.full);
    tree->SetBranchAddress("time", &evt.time);
    tree->SetBranchAddress("realtime", &evt.realtime);

    const Long64_t n_entries = tree->GetEntries();
    for (Long64_t i = 0; i < n_entries; ++i) {
        if (tree->GetEntry(i) <= 0) {
            continue;
        }

        const auto it = channel_to_segment.find(evt.channel);
        if (it == channel_to_segment.end()) {
            continue;
        }

        Hit hit;
        hit.time_us = evt.realtime * 1.0e6;
        hit.channel = evt.channel;
        segments[it->second].push_back(hit);
    }
}

} // namespace

RunWindow resolve_run_window(const AnalysisConfig& cfg, int run_number) {
    RunWindow window = cfg.default_window;

    if (!cfg.use_runinfo_windows || cfg.runinfo_json.empty()) {
        return window;
    }

    const std::string run_id = run_to_string(run_number);
    const std::optional<json> rec = find_run_record(cfg.runinfo_json, run_id);
    if (!rec.has_value()) {
        return window;
    }

    const double start_s = pick_number_with_fallback(*rec, {"fill_time"}, 0) + 
               pick_number_with_fallback(*rec, {"clean_time"}, 0) + 
               pick_number_with_fallback(*rec, {"hold_time"}, 0) +
               40;

    window.signal_start_us = start_s * 1.0e6;

    window.background_start_us = std::max(0.0, (start_s - 60.0) * 1.0e6);
    window.background_end_us   = start_s * 1.0e6;

    window.signal_start_us = start_s * 1.0e6;
    window.signal_end_us   = (start_s + 60.0) * 1.0e6;

    window.end_start_us = (start_s + 120.0) * 1.0e6;
    window.end_end_us   = (start_s + 180.0) * 1.0e6;

    return window;
}

LoadedRun load_root_run(const AnalysisConfig& cfg, int run_number) {
    const std::string run_id = run_to_string(run_number);
    const fs::path root_path = build_root_path(cfg, run_number);

    if (!fs::exists(root_path)) {
        throw std::runtime_error("ROOT file not found: " + root_path.string());
    }

    TFile* file = TFile::Open(root_path.string().c_str(), "READ");
    if (file == nullptr || file->IsZombie()) {
        throw std::runtime_error("Could not open ROOT file: " + root_path.string());
    }

    if (cfg.root_channel_maps.empty()) {
        file->Close();
        throw std::runtime_error("No ROOT channel map configured.");
    }

    if (cfg.require_tems_tree) {
        TTree* tems = dynamic_cast<TTree*>(file->Get(cfg.tems_tree_name.c_str()));
        if (tems == nullptr) {
            file->Close();
            throw std::runtime_error(
                "Missing required tree in " + root_path.string() +
                ". Expected: " + cfg.tems_tree_name
            );
        }
    }

    std::map<std::string, std::vector<Hit>> seg_map;
    for (const std::string& segment_name : cfg.segment_order) {
        seg_map[segment_name] = {};
    }

    for (const auto& [tree_name, channel_to_segment] : cfg.root_channel_maps) {
        TTree* tree = dynamic_cast<TTree*>(file->Get(tree_name.c_str()));
        if (tree == nullptr) {
            file->Close();
            throw std::runtime_error(
                "Missing required mapped tree in " + root_path.string() +
                ". Expected: " + tree_name
            );
        }

        for (const auto& [channel, segment_name] : channel_to_segment) {
            if (!seg_map.count(segment_name)) {
                seg_map[segment_name] = {};
            }
        }

        read_tree_into_segments(tree, channel_to_segment, seg_map);
    }

    file->Close();

    LoadedRun out;
    out.run_id = run_id;

    for (auto& [segment_name, hits] : seg_map) {
        sort_hits(hits);
        out.segments.push_back(LoadedSegment{segment_name, std::move(hits)});
    }

    return out;
}

} // namespace ucn::io