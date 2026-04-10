cmake_minimum_required(VERSION 3.20)
project(ucn_pulse_fitting_translation_complete LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ROOT REQUIRED COMPONENTS Core RIO Tree)
find_package(nlohmann_json CONFIG REQUIRED)

add_library(ucn_core
    src/templates/TabulatedPulseTemplate.cpp
    src/templates/DoubleExponentialPulseTemplate.cpp
    src/inference/Likelihood.cpp
    src/inference/GreedyLRTFitter.cpp
    src/app/AnalysisPipeline.cpp
    src/app/WindowedPulseProcessor.cpp
    src/app/BatchAnalysisRunner.cpp
    src/io/CsvPulseWriter.cpp
    src/io/CsvTemplateLoader.cpp
    src/io/AnalysisConfig.cpp
    src/io/RootRunLoader.cpp
)

target_include_directories(ucn_core PUBLIC include)

target_link_libraries(ucn_core
    PUBLIC
        ROOT::Core
        ROOT::RIO
        ROOT::Tree
        nlohmann_json::nlohmann_json
)

target_compile_options(ucn_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(pulse_analysis_main apps/pulse_analysis_main.cpp)
target_link_libraries(pulse_analysis_main PRIVATE ucn_core)

add_executable(run_batch_analysis apps/run_batch_analysis.cpp)
target_link_libraries(run_batch_analysis PRIVATE ucn_core)

include(CTest)
if(BUILD_TESTING)
    add_executable(test_ucn_core tests/test_ucn_core.cpp)
    target_link_libraries(test_ucn_core PRIVATE ucn_core)
    add_test(NAME test_ucn_core COMMAND test_ucn_core)
endif()