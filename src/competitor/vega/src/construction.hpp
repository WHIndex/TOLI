#pragma once

#include <cmath>
#include <iostream>

#include "pla.hpp"

namespace vega {

enum PartType {
  FMTD,  // compact
  FDTM   // sparse
};

struct PartitionInfo {
  PartitionInfo() = default;
  PartitionInfo(PartType type, float slope, uint32_t intercept,
                uint32_t begin_idx, uint32_t end_idx, size_t first_key,
                size_t last_key)
      : type_(type),
        slope_(slope),
        intercept_(intercept),
        begin_idx_(begin_idx),
        end_idx_(end_idx),
        first_key_(first_key),
        last_key_(last_key) {}

  float slope_;
  uint32_t intercept_;

  // [begin_idx_, end_idx_)
  uint32_t begin_idx_;
  uint32_t end_idx_;

  // [first_key_, last_key_)
  size_t first_key_;
  size_t last_key_;

  PartType type_;
};

template <typename Fin>
size_t fast_construct_one_layer(size_t n, size_t epsilon, uint32_t block_size,
                                double buckets_per_key, Fin in,
                                std::vector<PartitionInfo> &partitions) {
  struct Point {
    Point() = default;
    Point(uint64_t a, uint64_t b) : x(a), y(b) {}
    uint64_t x, y;
  };
  struct PartitionBuildState {
    PartitionBuildState(uint64_t epsilon, uint64_t g, double bpk)
        : granularity(g),
          n_add_points(0),
          buckets_per_key(bpk),
          all_partitions(0),
          all_buckets(0),
          pla(epsilon) {}
    uint64_t begin_idx,
        end_idx;  // for origin segments idx, not the rearrange idx
    uint64_t first_x, prev_x, last_x;
    uint64_t min_gap;
    OptimalPiecewiseLinearModel<size_t, size_t> pla;
    bool FDTMOnly;
    const uint64_t granularity;
    uint64_t n_add_points;
    const double buckets_per_key;

    uint64_t all_partitions;
    uint64_t all_buckets;
    PartitionInfo building_info;

    void init_next_state(Point first_p, uint64_t first_idx) {
      pla.reset();
      FDTMOnly = false;
      add(first_p, first_idx);
      begin_idx = end_idx = first_idx;
      first_x = prev_x = last_x = first_p.x;
      min_gap = std::numeric_limits<size_t>::max();
    }

    bool add(Point p, uint64_t idx) {
      bool ret = true;
      prev_x = last_x;
      last_x = p.x;
      min_gap = std::min(min_gap, last_x - prev_x);
      if (!FDTMOnly && pla.add_point(p.x, p.y)) {
        ret = true;
      } else {
        if (min_gap == 0) {
          std::cout << idx << std::endl;
          throw std::runtime_error("min_gap zero");
        }
        size_t cur_n_buckets =
            ceil((double)((last_x - first_x) * granularity) / (double)min_gap);
        double avg_bpk = (all_buckets + cur_n_buckets) / n_add_points;
        if (avg_bpk <= buckets_per_key) {
        // if (false) {
          FDTMOnly = true;
          ret = true;
        } else {
          PartitionInfo &part = building_info;
          if (FDTMOnly) {
            part.slope_ = granularity * 1.0 / (float)min_gap;
            part.intercept_ = 0;
            part.type_ = PartType::FDTM;
            all_buckets +=
                ceil(((prev_x - first_x) * part.slope_ + part.intercept_) /
                     granularity);
          } else {
            auto cs = pla.get_segment();
            auto [slope, intercept] =
                cs.get_floating_point_segment(cs.get_first_x());
            part.slope_ = slope;
            part.intercept_ = intercept;
            part.type_ = PartType::FMTD;
            all_buckets += ((end_idx - begin_idx) / granularity);
          }
          part.begin_idx_ = begin_idx;
          part.end_idx_ = end_idx;
          part.first_key_ = first_x;
          part.last_key_ = prev_x;
          pla.reset();
          all_partitions++;
          FDTMOnly = false;
          ret = false;
        }
      }
      if (ret) {
        end_idx = idx;
        last_x = p.x;
        n_add_points++;
      } else {
        begin_idx = end_idx = idx;
        first_x = prev_x = last_x = p.x;
      }
      return ret;
    }

    void finish() {
      PartitionInfo &part = building_info;
      if (FDTMOnly) {
        part.slope_ = granularity * 1.0 / (float)min_gap;
        part.intercept_ = 0;
        part.type_ = PartType::FDTM;
        all_buckets += ceil(
            ((prev_x - first_x) * part.slope_ + part.intercept_) / granularity);
      } else {
        auto cs = pla.get_segment();
        auto [slope, intercept] =
            cs.get_floating_point_segment(cs.get_first_x());
        part.slope_ = slope;
        part.intercept_ = intercept;
        part.type_ = PartType::FMTD;
        all_buckets += ((end_idx - begin_idx) / granularity);
      }
      part.begin_idx_ = begin_idx;
      part.end_idx_ = end_idx;
      part.first_key_ = first_x;
      part.last_key_ = prev_x;
      pla.reset();
      all_partitions++;
    }

    PartitionInfo get_one_part_info() { return building_info; }
  };

  if (n == 0) return 0;

  Point p, prev_p;
  PartitionBuildState state(epsilon, block_size, buckets_per_key);

  // train
  p = {in(0).first, 0};
  prev_p = p;
  state.init_next_state(p, 0);
  size_t i = block_size;
  for (i = block_size; i < n; i += block_size) {
    // get x and y
    p.x = in(i).first;
    p.y += block_size;

    // skip duplicate
    if (prev_p.x == p.x) continue;

    if (!state.add(p, i)) {
      partitions.push_back(state.get_one_part_info());
      i -= block_size;
      p = {in(i).first, 0};
      state.init_next_state(p, i);
    }

    prev_p = p;
  }

  // try to add last point
  bool add_last_point_success = false;
  i -= block_size;
  if (i < n - 1) {
    p.x = in(n - 1).first;
    p.y += (n - i - block_size);
    add_last_point_success = state.add(p, n - 1);
    if (add_last_point_success) {
      state.finish();
    }
    partitions.push_back(state.get_one_part_info());
  } else {  // i == n - 1
    add_last_point_success = true;
    state.finish();
    partitions.push_back(state.get_one_part_info());
  }
  // add last point to new partition
  if (!add_last_point_success) {
    i = partitions.back().end_idx_;
    p = {in(i).first, 0};
    state.init_next_state(p, i);
    p.x = in(n - 1).first;
    p.y += (n - i);
    state.add(p, n - 1);
    state.finish();
    partitions.push_back(state.get_one_part_info());
  }
  partitions.back().end_idx_ = n;
  return state.all_partitions;
}

// dynamic programming
size_t optimal_construct_one_layer(const std::vector<uint64_t> &keys, uint64_t epsilon, uint64_t space_multiple, bool print = false) {
  uint64_t n = keys.size();
  struct DP {
    // [i, k], [k + 1, j]
    uint64_t i, k, j;
    bool ok = false;
    uint64_t space;
    bool use_compact;
    void init(uint64_t begin, uint64_t mid, uint64_t end, uint64_t s, bool c) {
      i = begin;
      k = mid;
      j = end;
      ok = true;
      space = s;
      use_compact = c;
    }
  };
  auto train = [&](size_t i, size_t j, size_t &space, bool &use_compact) -> bool {
    // try FMTD
    size_t FMTD_space = UINT64_MAX;
    bool FMTD_ok = true;
    size_t FDTM_space = UINT64_MAX;
    size_t min_gap = UINT64_MAX;
    OptimalPiecewiseLinearModel<size_t, size_t> pla(epsilon);
    pla.add_point(keys[i], 0);
    for (size_t idx = i + 1; idx <= j; idx++) {
      size_t x = keys[idx];
      size_t y = idx - i;
      if (FMTD_ok && !pla.add_point(x, y)) {
        FMTD_ok = false;
      }
      min_gap = std::min(min_gap, keys[idx] - keys[idx - 1]);
    }
    if (FMTD_ok) {
      FMTD_space = j - i + 1;
    }
    FDTM_space = std::ceil((double)((keys[j] - keys[i])) / (double)min_gap);
    
    space = FMTD_ok ? FMTD_space : FDTM_space;
    use_compact = FMTD_ok;
    return true;
  };
  std::vector<DP> tmp_layer;
  std::vector<std::vector<DP>> layer;

  // 1. using pla to determine the upper_partition and  upper_space
  auto in_func = [&](auto i) { return std::pair<uint64_t, uint64_t>(keys[i], i); };
  auto out_func = [&](auto cs) {};
  uint64_t upper_partition = vega::make_segmentation(n, epsilon, in_func, out_func);
  uint64_t lower_space = n;
  uint64_t upper_space = n * space_multiple;
  if (print) {
    std::cout << "upper_partition: " << upper_partition << ", upper_space:" << upper_space << std::endl;
  }
  // if (space_multiple < 10) {
  //   std::cout << "space (n_buckets): " << (std::ceil((double)n / epsilon)) << std::endl;
  //   std::cout << "Buckets Per Key: " << ((double)1 / epsilon) << std::endl;
  //   return upper_partition;
  // }

  // 2. initialize the first layer of dp array, 
  //    and the number of corresponding partitions is 1.
  tmp_layer.resize(n);
  OptimalPiecewiseLinearModel<size_t, size_t> tmp_pla(epsilon);
  tmp_pla.add_point(keys[0], 0);
  uint64_t min_gap = UINT64_MAX;
  for (size_t j = 1; j < n; j++) {
    if (tmp_pla.add_point(keys[j], j)) {
      tmp_layer[j].init(0, 0, j, j + 1, true);
    } else {
      for (; j < n; j++) {
        min_gap = std::min(min_gap, keys[j] - keys[j - 1]);
        tmp_layer[j].init(0, 0, j, std::ceil((double)(keys[j] - keys[0]) / min_gap), false);
      }
      break;
    }
    min_gap = std::min(min_gap, keys[j] - keys[j - 1]);
  }

  layer.push_back(std::move(tmp_layer));

  // 3. dynamic programming, the number of partitions starts from 2.
  uint64_t p = 2;
  uint64_t space;
  bool use_compact;
  for (; p <= upper_partition; p++) {
    auto &layer1 = layer.back();
    auto &layer2 = tmp_layer;
    if (print) {
      std::cout << "finish " << p - 1 << ", space: " << layer1[n - 1].space << std::endl;
    }
    // 3.1 meet the space requirements and stop.
    if (layer1[n - 1].ok && layer1[n - 1].space < upper_space) {
      break;
    }

    // 3.2 calculate this layer
    layer2.resize(n);
    for (uint64_t j = p; j < n; j++) {
      int64_t opt_k = -1;
      size_t min_space = UINT64_MAX;
      bool c = false;
      for (size_t k = j - 1; k >= p; k--) {
        if (layer1[k].ok && train(k + 1, j, space, use_compact)) {
          size_t cur_space = layer1[k].space + space;
          if (cur_space <= min_space) {
            min_space = cur_space;
            opt_k = k;
            c = use_compact;
          }
        }
      }
      if (opt_k != -1) {
        layer2[j].init(0, opt_k, j, min_space, c);
      } else {
        layer2[j].ok = false;
      }
    }
    layer.push_back(std::move(layer2));
  }

  std::cout << "models info: " << std::endl;
  
  int64_t c = n - 1;
  for (int64_t i = layer.size() - 1; i >= 0; i--) {
    auto &dp = layer[i][c];
    std::cout << "  " << "part: " << (i + 1) << ", [" << (dp.k + 1) << ", " << dp.j << "] " << (dp.use_compact ? "compact" : "sparse") << std::endl;
    c = dp.k;
  }

  std::cout << "space (n_buckets): " << layer.back()[n - 1].space << std::endl;
  std::cout << "Buckets Per Key: " << ((double)layer.back()[n - 1].space / (double)n / epsilon) << std::endl;
  return layer.size();
}

}  // namespace vega
