#pragma once

#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <vector>

template <class KeyType = uint64_t>
struct EqualityLookup {
  KeyType key;
  uint64_t result;
};

template<typename KeyType>
struct KeyValue {
  KeyType key;
  uint64_t value;
} __attribute__((packed));

struct SearchRange {
  SearchRange(uint64_t low, uint64_t high, uint64_t predict) 
    : lo(low), hi(high), pred(predict) {}
  SearchRange(uint64_t low, uint64_t high) 
    : lo(low), hi(high), pred(low + (high - low) / 2) {}
  uint64_t lo;
  uint64_t hi;
  uint64_t pred;
};

static uint64_t timing(std::function<void()> fn) {
  const auto start = std::chrono::high_resolution_clock::now();
  fn();
  const auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

// Loads values from binary file into vector.
template <typename T>
static std::vector<T> load_binary_data(const std::string& filename, bool print = true) {
  std::vector<T> data;
  const uint64_t ns = timing([&] {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
      std::cerr << "unable to open " << filename << std::endl;
      exit(EXIT_FAILURE);
    }
    // Read size.
    uint64_t size;
    in.read(reinterpret_cast<char*>(&size), sizeof(uint64_t));
    data.resize(size);
    // Read values.
    in.read(reinterpret_cast<char*>(data.data()), size * sizeof(T));
    in.close();
  });
  const double ms = (double)ns / 1e6;

  if (print) {
    std::cout << "read " << data.size() << " values from " << filename << " in "
              << ms << " ms (" << static_cast<double>(data.size()) / 1000 / ms
              << " M values/s)" << std::endl;
  }

  return data;
}

// Writes values from vector into binary file.
template <typename T>
static void write_data(const std::vector<T>& data, const std::string& filename, const bool print = true) {
  const uint64_t ns = timing([&] {
    std::ofstream out(filename, std::ios_base::trunc | std::ios::binary);
    if (!out.is_open()) {
      std::cerr << "unable to open " << filename << std::endl;
      exit(EXIT_FAILURE);
    }
    // Write size.
    const uint64_t size = data.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(uint64_t));
    // Write values.
    out.write(reinterpret_cast<const char*>(data.data()), size * sizeof(T));
    out.close();
  });
  const uint64_t ms = ns / 1e6;
  if (print) {
    std::cout << "wrote " << data.size() << " values to " << filename << " in "
              << ms << " ms (" << static_cast<double>(data.size()) / 1000 / ms
              << " M values/s)" << std::endl;
  }
}

template<typename RandomIt, typename KeyType>
static RandomIt lower_bound(RandomIt begin, size_t lo, size_t hi, KeyType key) {
  size_t left = lo;
  size_t right = hi;
  while (left < right) {
    size_t mid = left + (right - left) / 2;
    if (begin[mid].key < key) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return begin + left;
}

template<typename RandomIt, typename KeyType>
static RandomIt exponential_search_lower_bound(RandomIt first, RandomIt last, KeyType key, size_t predict) {
    // Continue doubling the bound until it contains the lower bound. Then use
    // binary search.
    size_t bound = 1;
    size_t m = predict;
    size_t l, r;  // will do binary search in range [l, r)
    if (first[m].key >= key) {
        size_t size = m;
        while (bound < size && first[m - bound].key >= key) {
            bound *= 2;
        }
        l = m - std::min<size_t>(bound, size);
        r = m - bound / 2;
    } else {
        size_t size = std::distance(first, last) - m;
        while (bound < size && first[m + bound].key < key) {
            bound *= 2;
        }
        l = m + bound / 2;
        r = m + std::min<size_t>(bound, size);
    }
    return lower_bound<RandomIt, KeyType>(first, l, r, key);
}

class ResultManager {
private:
    struct Result {
        uint64_t space_;
        long double build_time_;
        long double query_latency_;
        long double throughput_;
    };

    std::vector<Result> results_;
    std::string latency_unit_;

public:
    ResultManager(std::string latency_unit = "ns") : latency_unit_(latency_unit) {}
    ~ResultManager() {
        results_.clear();
    }

    void AddResult(uint64_t space, long double build_time, long double query_latency, long double throughput) {
        results_.push_back({space, build_time, query_latency, throughput});
    }

    void Clear() {
      results_.clear();
    }

    void PrintResults() {
        std::sort(results_.begin(), results_.end(), [](const Result& L, const Result& R) {
            return L.space_ < R.space_;
        });

        std::cout << "space (byte): ";
        for (size_t i = 0; i < results_.size(); ++i) {
            std::cout << results_[i].space_ << ", ";
        }
        std::cout << std::endl;

        std::cout << "query_latency (" << latency_unit_ << "): ";
        for (size_t i = 0; i < results_.size(); ++i) {
            std::cout << results_[i].query_latency_ << ", ";
        }
        std::cout << std::endl;

        std::cout << "throughput (Mops/s): ";
        for (size_t i = 0; i < results_.size(); ++i) {
            std::cout << results_[i].throughput_ << ", ";
        }
        std::cout << std::endl;

        std::cout << "build_time (s): ";
        for (size_t i = 0; i < results_.size(); ++i) {
            std::cout << results_[i].build_time_ << ", ";
        }
        std::cout << std::endl << std::endl;
    }
};
