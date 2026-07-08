# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.15)

project(zlib-accel-test VERSION 1.0.0)

link_directories(../build)

include(../common.cmake)

find_package(GTest REQUIRED)

# engine_perf links against system libz only — not libzlib-accel — so that
# LD_PRELOAD can inject a different shim for side-by-side comparison:
#
#   ./engine_perf --label=no-map                              # system libz baseline
#   LD_PRELOAD=../build/libzlib-accel.so ./engine_perf        # with zlib-accel shim
add_executable(engine_perf engine_perf.cpp)
target_link_libraries(engine_perf PRIVATE GTest::gtest pthread z)

# All remaining targets link against libzlib-accel
link_libraries(zlib-accel)
link_libraries(gtest pthread)

# TODO Remove once crc16 references fixed in QPL
add_link_options("-Wl,--unresolved-symbols=ignore-all")

add_executable(zlib_accel_test
    zlib_accel_test.cpp
    logging_test.cpp
    test_utils.cpp
    ../utils.cpp
)

add_custom_target(run
    COMMAND ./zlib_accel_test
    DEPENDS zlib_accel_test
)

add_executable(compress_perf compress_perf.cpp)
add_executable(sharded_map_perf sharded_map_perf.cpp)

add_custom_target(perf
    COMMAND ./compress_perf
    COMMAND ./sharded_map_perf
    COMMAND ./engine_perf
    DEPENDS compress_perf sharded_map_perf engine_perf
)

if(COVERAGE)
    add_custom_target(coverage
        COMMAND lcov --directory . --capture --output-file zlib_accel_test.info
        COMMAND lcov --directory ../../build --capture --output-file zlib_accel.info
        COMMAND genhtml -o html zlib_accel_test.info zlib_accel.info
    )
else()
    add_custom_target(coverage
        COMMAND echo "Cannot generate coverage report. Build with COVERAGE=ON."
    )
endif()
