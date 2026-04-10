#include "ucn/app/BatchAnalysisRunner.hpp"
#include "ucn/io/AnalysisConfig.hpp"

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: run_batch_analysis <config.json>\n";
        return 1;
    }

    try {
        const std::string config_path = argv[1];
        const ucn::io::AnalysisConfig cfg = ucn::io::load_analysis_config(config_path);

        ucn::app::BatchAnalysisRunner runner(cfg);
        runner.run();
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << '\n';
        return 2;
    }

    return 0;
}