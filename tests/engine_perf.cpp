// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// engine_perf.cpp — Engine-agnostic compress/decompress throughput benchmark.
//
// Links against system libz. Inject a different zlib implementation via
// LD_PRELOAD to compare implementations side-by-side:
//
//   ./engine_perf --label=no-map                                      # system libz
//   LD_PRELOAD=../build/libzlib-accel.so ./engine_perf --label=modulo # current shim
//   LD_PRELOAD=../build-pr59/libzlib-accel.so ./engine_perf --label=fibonacci
//
// Flags (consumed before GTest sees argv):
//   --label=<name>    label written to RecordProperty  [default: "zlib"]
//   --warmup=N        warmup seconds per test case     [default: 1]
//   --measure=N       measure seconds per test case    [default: 3]
//   --min-size=N      skip sizes < N in SizeSweep      [default: 1024]
//   --max-size=N      skip sizes > N in SizeSweep      [default: no limit]
//
// Environment variable (set before process start — affects parameter generation
// at static init time, which runs before main()):
//   ZLIB_ACCEL_PERF_THREADS=N   thread count for SizeSweep; upper bound for
//                                ThreadSweep  [default: min(nproc, 32)]
//
// Two parameterised suites:
//   SizeSweep/{Compress,Decompress}Test.Throughput/<size>_t<N>
//     All sizes 512 B – 1 MB at the configured thread count.
//     512 B is included so --min-size=512 exposes ShardedMap overhead directly.
//     Filtered at runtime by --min-size / --max-size.
//
//   ThreadSweep/{Compress,Decompress}Test.Throughput/64KB_t<N>
//     Thread scaling at a fixed 64 KB input.
//
// RecordProperty keys: label, direction, input_bytes, threads, ops_per_sec, mb_per_sec.

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>
#include <zlib.h>

#include <gtest/gtest.h>

// ---------------------------------------------------------------------------
// Runtime config — populated in main() from flags
// ---------------------------------------------------------------------------
static int         g_warmup_secs  = 1;
static int         g_measure_secs = 3;
static std::string g_label        = "zlib";
static size_t      g_min_size     = 1024;
static size_t      g_max_size     = ~static_cast<size_t>(0);

// Pre-compressed buffers for Decompress tests, keyed by original size.
// Whatever library is loaded (libz or LD_PRELOADed shim) produces standard
// raw deflate at setup time, which all implementations can decompress.
static std::map<size_t, std::vector<uint8_t>> g_compressed;

// ---------------------------------------------------------------------------
// Size/thread configuration
// ---------------------------------------------------------------------------
// 512 B is included so --min-size=512 targets the ShardedMap stress case.
static const std::vector<size_t> kSizes = {
    512, 1024, 4096, 16384, 65536, 262144, 1048576
};

// Fixed input size for the ThreadSweep suite.
static constexpr size_t kScalingSize = 65536;  // 64 KB

static int HwThreadCount() {
    return std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
}

// Read at static-init time (SizeSweepParams / ThreadSweepParams are called
// before main()), so this must use getenv() rather than a flag-set global.
static int ConfiguredThreads() {
    const char* v = getenv("ZLIB_ACCEL_PERF_THREADS");
    if (v && std::atoi(v) > 0)
        return std::min(std::atoi(v), HwThreadCount());
    return std::min(32, HwThreadCount());
}

// ---------------------------------------------------------------------------
// Parameter type and generators
// ---------------------------------------------------------------------------
struct PerfParam { size_t input_bytes; int threads; };

static std::string SizeLabel(size_t bytes) {
    if      (bytes >= 1024 * 1024) return std::to_string(bytes / (1024 * 1024)) + "MB";
    else if (bytes >= 1024)        return std::to_string(bytes / 1024) + "KB";
    else                           return std::to_string(bytes) + "B";
}

static std::string ParamName(const ::testing::TestParamInfo<PerfParam>& info) {
    return SizeLabel(info.param.input_bytes) + "_t" + std::to_string(info.param.threads);
}

// SizeSweep: all sizes at the configured thread count.
// Runtime filtering by g_min_size/g_max_size is done via GTEST_SKIP().
static std::vector<PerfParam> SizeSweepParams() {
    const int t = ConfiguredThreads();
    std::vector<PerfParam> v;
    for (size_t s : kSizes) v.push_back({s, t});
    return v;
}

// ThreadSweep: powers of 2 up to ConfiguredThreads() at the fixed 64 KB size.
static std::vector<PerfParam> ThreadSweepParams() {
    const int max = ConfiguredThreads();
    std::vector<PerfParam> v;
    for (int n = 1; n <= max; n *= 2) v.push_back({kScalingSize, n});
    if ((max & (max - 1)) != 0) v.push_back({kScalingSize, max});
    return v;
}

// ---------------------------------------------------------------------------
// Compress-only test
// ---------------------------------------------------------------------------
class CompressTest : public ::testing::TestWithParam<PerfParam> {};

TEST_P(CompressTest, Throughput) {
    const size_t input_bytes = GetParam().input_bytes;
    const int    num_threads = GetParam().threads;

    if (input_bytes < g_min_size || input_bytes > g_max_size) GTEST_SKIP();

    std::vector<uint8_t> src(input_bytes);
    for (size_t i = 0; i < input_bytes; i++) src[i] = static_cast<uint8_t>(i % 251);

    z_stream probe{};
    deflateInit2(&probe, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY);
    const uLong comp_bound = deflateBound(&probe, static_cast<uLong>(input_bytes));
    deflateEnd(&probe);

    std::atomic<uint64_t> total_ops{0};
    std::atomic<bool>     stop{false};

    auto worker = [&]() {
        std::vector<uint8_t> comp(comp_bound);
        uint64_t ops = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            z_stream s{};
            deflateInit2(&s, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY);
            s.next_in   = const_cast<Bytef*>(src.data());
            s.avail_in  = static_cast<uInt>(input_bytes);
            s.next_out  = comp.data();
            s.avail_out = static_cast<uInt>(comp_bound);
            deflate(&s, Z_FINISH);
            deflateEnd(&s);
            ops++;
        }
        total_ops.fetch_add(ops, std::memory_order_relaxed);
    };

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int i = 0; i < num_threads; i++) threads.emplace_back(worker);

    std::this_thread::sleep_for(std::chrono::seconds(g_warmup_secs));
    const uint64_t ops_start = total_ops.load(std::memory_order_relaxed);
    const auto     t0        = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::seconds(g_measure_secs));
    stop.store(true, std::memory_order_relaxed);
    for (auto& t : threads) t.join();

    const uint64_t ops     = total_ops.load() - ops_start;
    const double   elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    const double   ops_p_s = static_cast<double>(ops) / elapsed;
    const double   mb_p_s  = ops_p_s * static_cast<double>(input_bytes) / (1024.0 * 1024.0);

    RecordProperty("label",       g_label);
    RecordProperty("direction",   "compress");
    RecordProperty("input_bytes", static_cast<int64_t>(input_bytes));
    RecordProperty("threads",     num_threads);
    RecordProperty("ops_per_sec", static_cast<int64_t>(ops_p_s));
    RecordProperty("mb_per_sec",  static_cast<int64_t>(mb_p_s));

    std::cout << "[COMPRESS]   label=" << std::setw(12) << g_label
              << "  size=" << std::setw(7) << SizeLabel(input_bytes)
              << "  threads=" << std::setw(3) << num_threads
              << "  MB/s=" << std::setw(8) << static_cast<int64_t>(mb_p_s)
              << "\n";
}

// ---------------------------------------------------------------------------
// Decompress-only test
// ---------------------------------------------------------------------------
class DecompressTest : public ::testing::TestWithParam<PerfParam> {};

TEST_P(DecompressTest, Throughput) {
    const size_t input_bytes = GetParam().input_bytes;
    const int    num_threads = GetParam().threads;

    if (input_bytes < g_min_size || input_bytes > g_max_size) GTEST_SKIP();

    ASSERT_TRUE(g_compressed.count(input_bytes))
        << "No pre-compressed data for size " << input_bytes;
    const std::vector<uint8_t>& comp = g_compressed.at(input_bytes);
    const uInt comp_size = static_cast<uInt>(comp.size());

    std::atomic<uint64_t> total_ops{0};
    std::atomic<bool>     stop{false};

    auto worker = [&]() {
        std::vector<uint8_t> out(input_bytes);
        uint64_t ops = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            z_stream s{};
            inflateInit2(&s, -15);
            s.next_in   = const_cast<Bytef*>(comp.data());
            s.avail_in  = comp_size;
            s.next_out  = out.data();
            s.avail_out = static_cast<uInt>(input_bytes);
            inflate(&s, Z_FINISH);
            inflateEnd(&s);
            ops++;
        }
        total_ops.fetch_add(ops, std::memory_order_relaxed);
    };

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int i = 0; i < num_threads; i++) threads.emplace_back(worker);

    std::this_thread::sleep_for(std::chrono::seconds(g_warmup_secs));
    const uint64_t ops_start = total_ops.load(std::memory_order_relaxed);
    const auto     t0        = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::seconds(g_measure_secs));
    stop.store(true, std::memory_order_relaxed);
    for (auto& t : threads) t.join();

    const uint64_t ops     = total_ops.load() - ops_start;
    const double   elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    const double   ops_p_s = static_cast<double>(ops) / elapsed;
    const double   mb_p_s  = ops_p_s * static_cast<double>(input_bytes) / (1024.0 * 1024.0);

    RecordProperty("label",       g_label);
    RecordProperty("direction",   "decompress");
    RecordProperty("input_bytes", static_cast<int64_t>(input_bytes));
    RecordProperty("threads",     num_threads);
    RecordProperty("ops_per_sec", static_cast<int64_t>(ops_p_s));
    RecordProperty("mb_per_sec",  static_cast<int64_t>(mb_p_s));

    std::cout << "[DECOMPRESS] label=" << std::setw(12) << g_label
              << "  size=" << std::setw(7) << SizeLabel(input_bytes)
              << "  threads=" << std::setw(3) << num_threads
              << "  MB/s=" << std::setw(8) << static_cast<int64_t>(mb_p_s)
              << "\n";
}

// ---------------------------------------------------------------------------
// Instantiate test suites
// ---------------------------------------------------------------------------
INSTANTIATE_TEST_SUITE_P(SizeSweep,   CompressTest,   ::testing::ValuesIn(SizeSweepParams()),   ParamName);
INSTANTIATE_TEST_SUITE_P(SizeSweep,   DecompressTest, ::testing::ValuesIn(SizeSweepParams()),   ParamName);
INSTANTIATE_TEST_SUITE_P(ThreadSweep, CompressTest,   ::testing::ValuesIn(ThreadSweepParams()), ParamName);
INSTANTIATE_TEST_SUITE_P(ThreadSweep, DecompressTest, ::testing::ValuesIn(ThreadSweepParams()), ParamName);

// ---------------------------------------------------------------------------
// main()
// ---------------------------------------------------------------------------
static void ParseFlags(int& argc, char** argv) {
    int out = 1;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if      (arg.rfind("--label=",    0) == 0) g_label        = arg.substr(8);
        else if (arg.rfind("--warmup=",   0) == 0) g_warmup_secs  = std::atoi(arg.substr(9).c_str());
        else if (arg.rfind("--measure=",  0) == 0) g_measure_secs = std::atoi(arg.substr(10).c_str());
        else if (arg.rfind("--min-size=", 0) == 0) g_min_size     = static_cast<size_t>(std::atoi(arg.substr(11).c_str()));
        else if (arg.rfind("--max-size=", 0) == 0) g_max_size     = static_cast<size_t>(std::atoi(arg.substr(11).c_str()));
        else argv[out++] = argv[i];
    }
    argc = out;
}

static std::vector<uint8_t> DeflateBuffer(const std::vector<uint8_t>& src) {
    z_stream s{};
    deflateInit2(&s, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY);
    const uLong bound = deflateBound(&s, static_cast<uLong>(src.size()));
    std::vector<uint8_t> out(bound);
    s.next_in   = const_cast<Bytef*>(src.data());
    s.avail_in  = static_cast<uInt>(src.size());
    s.next_out  = out.data();
    s.avail_out = static_cast<uInt>(bound);
    deflate(&s, Z_FINISH);
    out.resize(s.total_out);
    deflateEnd(&s);
    return out;
}

int main(int argc, char** argv) {
    ParseFlags(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);

    // Pre-compress all sizes. The loaded library (system libz or LD_PRELOADed
    // shim) produces standard raw deflate that all implementations can decompress.
    for (size_t sz : kSizes) {
        std::vector<uint8_t> src(sz);
        for (size_t i = 0; i < sz; i++) src[i] = static_cast<uint8_t>(i % 251);
        g_compressed[sz] = DeflateBuffer(src);
    }

    const int t = ConfiguredThreads();
    const std::string max_size_str =
        (g_max_size == ~static_cast<size_t>(0)) ? "no limit" : SizeLabel(g_max_size);

    std::cout << "\nzlib-accel engine throughput benchmark\n"
              << "  Label      : " << g_label << "\n"
              << "  Warmup     : " << g_warmup_secs << "s per test (discarded)\n"
              << "  Measure    : " << g_measure_secs << "s per test\n"
              << "  Size filter: " << SizeLabel(g_min_size) << " – " << max_size_str << "\n"
              << "  Threads    : " << t << " (SizeSweep);  1–" << t << " (ThreadSweep)\n"
              << "  Tip: set ZLIB_ACCEL_PERF_THREADS=N before running to change thread count\n"
              << std::string(60, '-') << "\n";

    return RUN_ALL_TESTS();
}

//
// Measures compress-only and decompress-only throughput across a range of
// input sizes and thread counts. Invoke once per engine using a config file:
//
//   ./engine_perf                                          # library default config
//   ./engine_perf --config=/path/to/qat.conf              # load engine config
//   ./engine_perf --config=/path/to/iaa.conf --label=iaa  # with explicit label
//   ./engine_perf --gtest_output=json:results.json         # JSON for HTML visualiser
//
// The --config and --label flags are consumed before GTest sees argv, so all
// standard GTest flags (--gtest_output, --gtest_filter, etc.) work normally.
//
// Two parameterised test suites are run:
//
//   SizeSweep/{Compress,Decompress}Test.Throughput/<size>_t<N>
//     — Sweeps input sizes from 1 KB to 1 MB at max thread count.
//       Useful for latency/throughput curves: engines like QAT have a
//       minimum input threshold (1 KB) below which they fall back to zlib.
//
//   ThreadSweep/{Compress,Decompress}Test.Throughput/64KB_t<N>
//     — Sweeps thread counts from 1 to min(nproc, 32) at a fixed 64 KB input.
//       Useful for tracking scalability regressions.
//
// Decompressed inputs are pre-compressed with zlib (engine-independent) so
// that the Decompress suite exercises only the decompression path.
//
// RecordProperty keys (for HTML visualiser): engine, direction, input_bytes,
//   threads, ops_per_sec, mb_per_sec.

#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>
#include <zlib.h>

#include <gtest/gtest.h>

#include "../config/config.h"

