// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// compress_perf.cpp — Multithreaded compress/decompress throughput benchmark.
//
// Exercises the ShardedMap hot path by running N concurrent threads, each
// continuously calling deflateInit2 → deflate → deflateEnd and
// inflateInit2 → inflate → inflateEnd via the libzlib-accel shim.
//
//   deflateInit2 / inflateInit2  →  ShardedMap::Set
//   deflate      / inflate       →  ShardedMap::Get
//   deflateEnd   / inflateEnd    →  ShardedMap::Unset
//
// A 1 KB input is used so that each round-trip completes quickly, making map
// overhead a proportionally larger share of wall time and amplifying the
// signal from locking differences between the stdlib and TBB implementations.
//
// Usage:
//   ./compress_perf                             # plain stdout table
//   ./compress_perf --gtest_output=json:out.json  # JSON for HTML visualiser
//
// To compare implementations, build libzlib-accel.so twice (once with the
// stdlib ShardedMap, once with USE_TBB) and run this binary against each.

#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>
#include <zlib.h>

#include <gtest/gtest.h>

// ---------------------------------------------------------------------------
// Thread count sweep: powers of 2 from 1 up to hardware_concurrency(),
// with the exact nproc count appended if it is not itself a power of 2.
// ---------------------------------------------------------------------------
static std::vector<int> ThreadCounts() {
  std::vector<int> counts;
  const int max =
      std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
  for (int n = 1; n <= max; n *= 2) counts.push_back(n);
  if (counts.back() != max) counts.push_back(max);
  return counts;
}

// ---------------------------------------------------------------------------
// Benchmark parameters
// ---------------------------------------------------------------------------
static constexpr size_t kInputSize   = 1024;  // 1 KB per op
static constexpr int    kWarmupSecs  = 1;     // discarded before measurement
static constexpr int    kMeasureSecs = 5;     // measurement window

// ---------------------------------------------------------------------------
// Parameterised test: one instance per thread count
// ---------------------------------------------------------------------------
class CompressPerfTest : public ::testing::TestWithParam<int> {};

TEST_P(CompressPerfTest, ThroughputThreadSweep) {
  const int num_threads = GetParam();

  // Compressible source data: repeating byte pattern
  std::vector<uint8_t> src(kInputSize);
  for (size_t i = 0; i < kInputSize; i++) src[i] = static_cast<uint8_t>(i);

  // Compute compressed-output upper bound once (same for all threads)
  z_stream probe{};
  deflateInit2(&probe, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -15, 8,
               Z_DEFAULT_STRATEGY);
  const uLong comp_bound = deflateBound(&probe, static_cast<uLong>(kInputSize));
  deflateEnd(&probe);

  std::atomic<uint64_t> total_ops{0};
  std::atomic<bool>     stop{false};

  auto worker = [&]() {
    std::vector<uint8_t> comp(comp_bound);
    std::vector<uint8_t> decomp(kInputSize);
    uint64_t ops = 0;

    while (!stop.load(std::memory_order_relaxed)) {
      // Compress -------------------------------------------------------
      // deflateInit2 → Set, deflate → Get, deflateEnd → Unset
      z_stream c{};
      deflateInit2(&c, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -15, 8,
                   Z_DEFAULT_STRATEGY);
      c.next_in   = const_cast<Bytef*>(src.data());
      c.avail_in  = static_cast<uInt>(kInputSize);
      c.next_out  = comp.data();
      c.avail_out = static_cast<uInt>(comp_bound);
      deflate(&c, Z_FINISH);
      const uLong comp_size = c.total_out;
      deflateEnd(&c);

      // Decompress -----------------------------------------------------
      // inflateInit2 → Set, inflate → Get, inflateEnd → Unset
      z_stream d{};
      inflateInit2(&d, -15);
      d.next_in   = comp.data();
      d.avail_in  = static_cast<uInt>(comp_size);
      d.next_out  = decomp.data();
      d.avail_out = static_cast<uInt>(kInputSize);
      inflate(&d, Z_FINISH);
      inflateEnd(&d);

      ops++;
    }
    total_ops.fetch_add(ops, std::memory_order_relaxed);
  };

  // Start all worker threads
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (int i = 0; i < num_threads; i++) threads.emplace_back(worker);

  // Warmup: let allocators and CPU caches settle
  std::this_thread::sleep_for(std::chrono::seconds(kWarmupSecs));

  // Measurement window: snapshot ops at start and end to avoid atomic reset races
  const uint64_t ops_start = total_ops.load(std::memory_order_relaxed);
  const auto     t0        = std::chrono::steady_clock::now();
  std::this_thread::sleep_for(std::chrono::seconds(kMeasureSecs));
  stop.store(true, std::memory_order_relaxed);
  for (auto& t : threads) t.join();
  const uint64_t ops_end = total_ops.load(std::memory_order_relaxed);
  const double   elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0)
      .count();

  const uint64_t ops       = ops_end - ops_start;
  const double ops_per_sec = static_cast<double>(ops) / elapsed;
  const double mb_per_sec  = ops_per_sec * kInputSize / (1024.0 * 1024.0);

  // Expose metrics in gtest JSON/XML output (consumed by HTML visualiser)
  RecordProperty("threads",     num_threads);
  RecordProperty("ops_per_sec", static_cast<int64_t>(ops_per_sec));
  RecordProperty("mb_per_sec",  static_cast<int64_t>(mb_per_sec));
  RecordProperty("total_ops",   static_cast<int64_t>(ops));
  RecordProperty("elapsed_sec", static_cast<int64_t>(elapsed));

  // Immediate stdout summary (readable without the HTML tool)
  std::cout << "[PERF]"
            << "  threads=" << std::setw(3) << num_threads
            << "  ops/s="   << std::setw(10) << static_cast<int64_t>(ops_per_sec)
            << "  MB/s="    << std::setw(6)  << static_cast<int64_t>(mb_per_sec)
            << "\n";
}

INSTANTIATE_TEST_SUITE_P(
    ThreadSweep, CompressPerfTest,
    ::testing::ValuesIn(ThreadCounts()),
    [](const ::testing::TestParamInfo<int>& info) {
      return "threads_" + std::to_string(info.param);
    });

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  const int max =
      std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
  std::cout << "\nShardedMap throughput benchmark\n"
            << "  Input size : " << kInputSize << " B/op\n"
            << "  Warmup     : " << kWarmupSecs << "s (discarded)\n"
            << "  Measure    : " << kMeasureSecs << "s per thread count\n"
            << "  Thread sweep: 1";
  for (int n = 2; n <= max; n *= 2) std::cout << ", " << n;
  if ((max & (max - 1)) != 0) std::cout << ", " << max;
  std::cout << "\n" << std::string(52, '-') << "\n";
  return RUN_ALL_TESTS();
}
