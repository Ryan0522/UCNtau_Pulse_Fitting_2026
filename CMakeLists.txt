cmake_minimum_required(VERSION 3.20)
project(ucn_pulse_fitting LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ROOT REQUIRED COMPONENTS Core RIO Tree)
find_package(nlohmann_json CONFIG REQUIRED)

add_library(ucn_core
    src/templates/DoubleExpPulseTemplate.cpp
    src/inference/GreedyLRTFitter.cpp
    src/app/WindowedPulseProcessor.cpp
    src/app/BatchAnalysisRunner.cpp
    src/io/AnalysisConfig.cpp
    src/io/RootRunLoader.cpp
    src/inference/Likelihood.cpp
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

add_executable(run_batch_analysis apps/run_batch_analysis.cpp)
target_link_libraries(run_batch_analysis PRIVATE ucn_core)