// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// sharded_map_perf.cpp — ShardedMap distribution quality and shard throughput.
//
// Test 3: ShardDistributionTest.FibonacciVsModulo
//   Allocates N real z_stream objects, collects their heap addresses, and
//   applies both hash functions. Reports coefficient of variation (CV) and
//   max load factor per function via a per-shard histogram. Asserts that
//   Fibonacci distributes more uniformly.
//
// Test 4: ShardThroughputTest.OpsPerSecond (parameterised)
//   Benchmarks a minimal read-heavy ShardedMap under two key distributions:
//     Clustered — keys are multiples of kNumShards, all map to shard 0
//                 under modulo, but spread under Fibonacci.
//     Uniform   — sequential keys, spread under both hash functions.
//   Runs each combination at a range of thread counts. The throughput gap
//   between Modulo+Clustered and Fibonacci+Clustered directly measures the
//   performance cost of shard hot-spotting under real z_stream pointer
//   alignment patterns.
//
// Usage: ./sharded_map_perf [--gtest_output=json:results.json]

#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include <zlib.h>

#include <gtest/gtest.h>

// ---------------------------------------------------------------------------
// Shared constants
// ---------------------------------------------------------------------------
static constexpr int      kNumShards      = 64;
static constexpr int      kMapPopulation  = 10000;
static constexpr int      kWarmupSecs     = 1;
static constexpr int      kMeasureSecs    = 3;
static constexpr int      kMaxThreads     = 32;  // cap sweep for manageable runtime
static constexpr uint64_t kFibMultiplier  = 11400714819323198485ull;  // 2^64 / phi

// ---------------------------------------------------------------------------
// Hash functions — mirror the ShardedMap implementations being compared
// ---------------------------------------------------------------------------
static int ModuloShard(uintptr_t key, int num_shards) {
  return static_cast<int>(std::hash<uintptr_t>{}(key) %
                           static_cast<uintptr_t>(num_shards));
}

static int FibonacciShard(uintptr_t key, int num_shards) {
  const int      shard_bits = __builtin_ctz(num_shards);
  const uint64_t h          = static_cast<uint64_t>(std::hash<uintptr_t>{}(key));
  return static_cast<int>((h * kFibMultiplier) >> (64 - shard_bits));
}

// ---------------------------------------------------------------------------
// Distribution metrics helpers
// ---------------------------------------------------------------------------
struct DistMetrics {
  double           cv;               // coefficient of variation: stddev / mean
  double           max_load_factor;  // busiest shard / ideal load
  std::vector<int> counts;
};

static DistMetrics ComputeMetrics(const std::vector<int>& counts, int n) {
  const double mean = static_cast<double>(kMapPopulation) / n;
  double       sq   = 0;
  int          mx   = 0;
  for (int c : counts) {
    sq += (c - mean) * (c - mean);
    mx = std::max(mx, c);
  }
  return {std::sqrt(sq / n) / mean, static_cast<double>(mx) / mean, counts};
}

static void PrintHistogram(const std::string& label, const DistMetrics& m,
                            int num_shards) {
  const int mx  = *std::max_element(m.counts.begin(), m.counts.end());
  const int bar = 36;
  const int show = std::min(num_shards, 16);
  std::cout << label << "  CV=" << std::fixed << std::setprecision(3) << m.cv
            << "  max_load=" << std::setprecision(2) << m.max_load_factor << "x\n";
  for (int s = 0; s < show; s++) {
    int b = mx > 0 ? m.counts[s] * bar / mx : 0;
    std::cout << "  [" << std::setw(2) << s << "] " << std::string(b, '#')
              << " " << m.counts[s] << "\n";
  }
  if (num_shards > show)
    std::cout << "  ... (" << num_shards << " shards total)\n";
  std::cout << "\n";
}

// ===========================================================================
// Test 3: Distribution quality
// ===========================================================================

TEST(ShardDistributionTest, FibonacciVsModulo) {
  // Allocate real z_stream objects so addresses reflect actual heap behaviour
  // (allocator alignment, arena boundaries, etc.).
  std::vector<std::unique_ptr<z_stream>> streams(kMapPopulation);
  for (auto& s : streams) s = std::make_unique<z_stream>();

  std::vector<int> mod_counts(kNumShards, 0);
  std::vector<int> fib_counts(kNumShards, 0);
  for (const auto& s : streams) {
    const uintptr_t k = reinterpret_cast<uintptr_t>(s.get());
    mod_counts[ModuloShard(k, kNumShards)]++;
    fib_counts[FibonacciShard(k, kNumShards)]++;
  }

  const DistMetrics mod_m = ComputeMetrics(mod_counts, kNumShards);
  const DistMetrics fib_m = ComputeMetrics(fib_counts, kNumShards);

  std::cout << "\nShard distribution — " << kMapPopulation
            << " real z_stream heap pointers, " << kNumShards << " shards\n"
            << std::string(60, '-') << "\n";
  PrintHistogram("Modulo   ", mod_m, kNumShards);
  PrintHistogram("Fibonacci", fib_m, kNumShards);

  // Encode floats as integer (×1000) for gtest JSON/RecordProperty
  RecordProperty("modulo_cv_m",              static_cast<int64_t>(mod_m.cv * 1000));
  RecordProperty("fibonacci_cv_m",           static_cast<int64_t>(fib_m.cv * 1000));
  RecordProperty("modulo_max_load_pct",      static_cast<int64_t>(mod_m.max_load_factor * 100));
  RecordProperty("fibonacci_max_load_pct",   static_cast<int64_t>(fib_m.max_load_factor * 100));

  EXPECT_LT(fib_m.cv, mod_m.cv)
      << "Fibonacci should distribute z_stream pointers more uniformly than modulo";
  EXPECT_LT(fib_m.max_load_factor, mod_m.max_load_factor)
      << "Fibonacci should produce a lower maximum shard load";
}

// ===========================================================================
// Test 4: Throughput under hot-shard vs uniform key distributions
// ===========================================================================

// Minimal ShardedMap for benchmarking — stdlib shared_mutex + unordered_map,
// hash function supplied at construction time. Kept simple so that only the
// hash function differs between Modulo and Fibonacci runs.
class BenchMap {
 public:
  using ShardFn = int (*)(uintptr_t, int);

  BenchMap(ShardFn fn, int num_shards)
      : fn_(fn), n_(num_shards), maps_(num_shards), mtx_(num_shards) {}

  void Set(uintptr_t key, int value) {
    std::unique_lock lock(mtx_[fn_(key, n_)]);
    maps_[fn_(key, n_)][key] = value;
  }

  int Get(uintptr_t key) {
    const int s = fn_(key, n_);
    std::shared_lock lock(mtx_[s]);
    auto it = maps_[s].find(key);
    return it == maps_[s].end() ? -1 : it->second;
  }

 private:
  ShardFn                                           fn_;
  int                                               n_;
  std::vector<std::unordered_map<uintptr_t, int>>  maps_;
  std::vector<std::shared_mutex>                    mtx_;
};

// ---------------------------------------------------------------------------

struct ThroughputParam {
  const char*     hash_name;
  const char*     key_name;
  BenchMap::ShardFn hash_fn;
  bool            clustered;  // true → keys are multiples of kNumShards
  int             num_threads;
};

static std::vector<ThroughputParam> ThroughputParams() {
  const int max =
      std::min(kMaxThreads,
               std::max(1, static_cast<int>(std::thread::hardware_concurrency())));
  std::vector<int> counts;
  for (int n = 1; n <= max; n *= 2) counts.push_back(n);
  if (counts.back() != max) counts.push_back(max);

  std::vector<ThroughputParam> params;
  for (int t : counts) {
    params.push_back({"Modulo",    "Clustered", ModuloShard,    true,  t});
    params.push_back({"Modulo",    "Uniform",   ModuloShard,    false, t});
    params.push_back({"Fibonacci", "Clustered", FibonacciShard, true,  t});
    params.push_back({"Fibonacci", "Uniform",   FibonacciShard, false, t});
  }
  return params;
}

class ShardThroughputTest : public ::testing::TestWithParam<ThroughputParam> {};

TEST_P(ShardThroughputTest, OpsPerSecond) {
  const auto& p = GetParam();

  // Key pool:
  //   Clustered: multiples of kNumShards → all hash to shard 0 under modulo
  //   Uniform:   sequential integers     → spread evenly across all shards
  std::vector<uintptr_t> keys(kMapPopulation);
  for (int i = 0; i < kMapPopulation; i++)
    keys[i] = p.clustered ? static_cast<uintptr_t>(i) * kNumShards
                           : static_cast<uintptr_t>(i);

  BenchMap bmap(p.hash_fn, kNumShards);
  for (int i = 0; i < kMapPopulation; i++) bmap.Set(keys[i], i);

  std::atomic<uint64_t> total_ops{0};
  std::atomic<bool>     stop{false};
  const int             pool = kMapPopulation;

  auto worker = [&]() {
    uint64_t ops = 0;
    // xorshift64 — cheap PRNG for key selection
    uint64_t rng = 6364136223846793005ull ^
                   reinterpret_cast<uintptr_t>(&ops);
    while (!stop.load(std::memory_order_relaxed)) {
      rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
      bmap.Get(keys[rng % pool]);
      ops++;
    }
    total_ops.fetch_add(ops, std::memory_order_relaxed);
  };

  std::vector<std::thread> threads;
  threads.reserve(p.num_threads);
  for (int i = 0; i < p.num_threads; i++) threads.emplace_back(worker);

  std::this_thread::sleep_for(std::chrono::seconds(kWarmupSecs));
  const uint64_t ops_start = total_ops.load(std::memory_order_relaxed);
  const auto     t0        = std::chrono::steady_clock::now();

  std::this_thread::sleep_for(std::chrono::seconds(kMeasureSecs));
  stop.store(true, std::memory_order_relaxed);
  for (auto& t : threads) t.join();

  const uint64_t ops_end = total_ops.load(std::memory_order_relaxed);
  const double   elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();

  const uint64_t ops       = ops_end - ops_start;
  const double   ops_per_s = static_cast<double>(ops) / elapsed;
  const double   mops      = ops_per_s / 1e6;

  RecordProperty("hash",        p.hash_name);
  RecordProperty("keys",        p.key_name);
  RecordProperty("threads",     p.num_threads);
  RecordProperty("ops_per_sec", static_cast<int64_t>(ops_per_s));
  RecordProperty("total_ops",   static_cast<int64_t>(ops));

  std::cout << "[SHARD]"
            << "  " << std::setw(9) << p.hash_name
            << "  " << std::setw(9) << p.key_name
            << "  threads=" << std::setw(3) << p.num_threads
            << "  Mops/s=" << std::fixed << std::setprecision(1) << std::setw(7)
            << mops << "\n";
}

INSTANTIATE_TEST_SUITE_P(
    HashComparison, ShardThroughputTest,
    ::testing::ValuesIn(ThroughputParams()),
    [](const ::testing::TestParamInfo<ThroughputParam>& info) {
      const auto& p = info.param;
      return std::string(p.hash_name) + "_" + p.key_name + "_threads_" +
             std::to_string(p.num_threads);
    });

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  const int max =
      std::min(kMaxThreads,
               std::max(1, static_cast<int>(std::thread::hardware_concurrency())));
  std::cout << "\nShardedMap distribution and throughput tests\n"
            << "  Shards: " << kNumShards
            << "  Population: " << kMapPopulation
            << "  Warmup: " << kWarmupSecs << "s"
            << "  Measure: " << kMeasureSecs << "s"
            << "  Max threads: " << max << "\n"
            << std::string(60, '-') << "\n";
  return RUN_ALL_TESTS();
}
