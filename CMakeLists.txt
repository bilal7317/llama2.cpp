cmake_minimum_required(VERSION 3.10)
project(llama2_cpp)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Use O3 optimization for all builds
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")

# Specify absolute path for model.bin
set(MODEL_BIN_PATH "${CMAKE_SOURCE_DIR}/src/model/model.bin")

# Specify absolute path for analysis.py
set(ANALYSIS_SCRIPT "${CMAKE_SOURCE_DIR}/src/python/analysis/analysis.py")
include(config.cmake)

configure_file(
    ${CMAKE_SOURCE_DIR}/cmake_config.h.in
    ${CMAKE_BINARY_DIR}/cmake_config.h
)

set(RESULTS_FILE "${CMAKE_SOURCE_DIR}/${RESULTS_DIR}/output_${DATATYPE}_${BITS}_bit.txt")
set(CONFIG_FILE "${CMAKE_SOURCE_DIR}/configs/config.txt")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(run_exec src/cpp/run.cpp)
target_link_libraries(run_exec PRIVATE stdc++fs)

add_executable(custom_exec src/cpp/custom.cpp)
add_executable(run_c_exec src/cpp/run.c)

target_link_libraries(run_c_exec m)

# Custom target to run model with correct path
add_custom_target(run_model
    COMMAND run_exec ${MODEL_BIN_PATH}
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    DEPENDS run_exec
)

# Target to update CMake config from Python
add_custom_target(update_cmake_config
    COMMAND python3 ${CMAKE_SOURCE_DIR}/update_cmake_config.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# Target to run analysis manually
add_custom_target(run_analysis
    COMMAND python3 ${ANALYSIS_SCRIPT} --input ${RESULTS_FILE} --config ${CONFIG_FILE}
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)

# Target for full flow: update config, compile, run, then analyze
add_custom_target(full_run
    COMMAND python3 ${CMAKE_SOURCE_DIR}/update_cmake_config.py
    COMMAND ${CMAKE_COMMAND} ..
    COMMAND run_exec ${MODEL_BIN_PATH}
    COMMAND python3 ${ANALYSIS_SCRIPT} --input ${RESULTS_FILE} --config ${CONFIG_FILE}
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    DEPENDS run_exec
)
