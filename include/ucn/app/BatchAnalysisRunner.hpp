#pragma once
#include "ucn/io/AnalysisConfig.hpp"

namespace ucn::app {

class BatchAnalysisRunner {
public:
    explicit BatchAnalysisRunner(const io::AnalysisConfig& cfg);

    void run() const;

private:
    io::AnalysisConfig cfg_;
};

} // namespace ucn::app