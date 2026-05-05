cmake_minimum_required(VERSION 3.20)
project(ucn_pulse_fitting LANGUAGES CXX)


set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)
if(POLICY CMP0126)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
endif()
include(FetchContent)

FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE 
)
FetchContent_MakeAvailable(json)

# --- Project Settings ---
# Enforce C++20 for std::span support
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(UCN_ENABLE_WARNINGS "Enable extra compiler warnings" ON)
option(UCN_BUILD_TESTS "Build tests" OFF)

# --- Find ROOT Framework ---
find_package(ROOT REQUIRED COMPONENTS Core RIO Tree)

# --- Core Library Target ---
add_library(ucn_core
    src/templates/DoubleExponentialPulseTemplate.cpp
    src/templates/GaussianTripPulseTemplate.cpp
    src/templates/GaussianQuadPulseTemplate.cpp
    src/templates/EmpiricalPulseTemplate.cpp
    src/inference/Likelihood.cpp
    src/inference/GreedyLRTFitter.cpp
    src/inference/CoincidenceFitter.cpp
    src/app/WindowedPulseProcessor.cpp
    src/app/BatchAnalysisRunner.cpp
    src/io/AnalysisConfig.cpp
    src/io/RootRunLoader.cpp
)

target_include_directories(ucn_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(ucn_core
    PUBLIC
        ROOT::Core
        ROOT::RIO
        ROOT::Tree
        nlohmann_json::nlohmann_json
)

# --- Compiler Warnings ---
if(UCN_ENABLE_WARNINGS)
    if(MSVC)
        target_compile_options(ucn_core PRIVATE /W4)
    else()
        target_compile_options(ucn_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
    endif()
endif()

# --- Executable Target ---
add_executable(run_batch_analysis
    apps/run_batch_analysis.cpp
)
add_executable(test_mc_csv apps/test_mc_csv.cpp)
add_executable(extract_coincidence_pulse_shape
    apps/extract_coincidence_pulse_shape.cpp
)

target_link_libraries(run_batch_analysis PRIVATE ucn_core)
target_link_libraries(test_mc_csv PRIVATE ucn_core)
target_link_libraries(extract_coincidence_pulse_shape PRIVATE ucn_core)

# --- Testing Target ---
if(UCN_BUILD_TESTS)
    enable_testing()
    add_executable(test_ucn_core tests/test_ucn_core.cpp)
    target_link_libraries(test_ucn_core PRIVATE ucn_core)
    add_test(NAME test_ucn_core COMMAND test_ucn_core)
endif()