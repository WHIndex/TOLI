#include "../indexInterface.h"
#include "./src/include/rs/builder.h"
#include "./src/include/rs/radix_spline.h"
#include "../vega/src/util.h"

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>
#include <cstddef>


template <class KEY_TYPE, class PAYLOAD_TYPE>
class RSInterface final : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
public:
  void init(Param *param = nullptr) {}

  void bulk_load(std::pair<KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num,
                 Param *param = nullptr);

  bool get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param = nullptr);

  bool put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr);

  bool update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr);

  bool remove(KEY_TYPE key, Param *param = nullptr);

  size_t scan(KEY_TYPE key_low_bound, size_t key_num,
              std::pair<KEY_TYPE, PAYLOAD_TYPE> *result,
              Param *param = nullptr);

  long long memory_consumption() { return rs_.GetSize(); }

  void print_stats(std::string s) {
    return ;
  }

private:
    rs::RadixSpline<KEY_TYPE> rs_;
    std::size_t size_;
    uint64_t build_time_;
    const std::pair<KEY_TYPE, PAYLOAD_TYPE>* data_view_;
};

template <class KEY_TYPE, class PAYLOAD_TYPE>
void RSInterface<KEY_TYPE, PAYLOAD_TYPE>::bulk_load(
    std::pair<KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num, Param *param) {
    data_view_ = key_value;
    size_ = num;
    
    if (key_value == nullptr || num == 0) {
        // 空数据：重置计时并返回（rs_ 保持默认构造态）
        build_time_ = 0;
        return;
    }
    
    // 提前说明：要求 key_value 已按 key 升序排列
    const KEY_TYPE min_key = key_value[0].first;
    const KEY_TYPE max_key = key_value[num - 1].first;
    
    build_time_ = timing([&] {
        // 边界保护：如果只有一条记录，min==max 也没问题
        rs::Builder<KEY_TYPE> builder(
            (num > 0 ? min_key : std::numeric_limits<KEY_TYPE>::min()),
            (num > 0 ? max_key : std::numeric_limits<KEY_TYPE>::max()),
            /*num_radix_bits=*/18,    // 可按需暴露到 Param；这里给一个常用默认
            /*max_error=*/32          // 同上：经验参数
        );
    
        // 仅喂入键
        for (size_t i = 0; i < num; ++i) {
        builder.AddKey(key_value[i].first);
        }
        rs_ = builder.Finalize();
    });
}

template <class KEY_TYPE, class PAYLOAD_TYPE>
bool RSInterface<KEY_TYPE, PAYLOAD_TYPE>::get(KEY_TYPE key, PAYLOAD_TYPE &val,
                                                Param *param) {
    (void)param;
    if (data_view_ == nullptr || size_ == 0) return false;
    
    // 使用 RS 缩小搜索范围
    auto bound = rs_.GetSearchBound(key);
    // 将 RS 给出的 [begin, end) 截断到真实数据范围内
    size_t lo = std::min<std::size_t>(bound.begin, size_);
    size_t hi = std::min<std::size_t>(bound.end,   size_);
    
    if (lo > hi) std::swap(lo, hi);      // 极端保护
    if (lo == hi) return false;          // 空区间
    
    const auto* begin = data_view_ + lo;
    const auto* end   = data_view_ + hi;
    
    auto it = std::lower_bound(
        begin, end, key,
        [](const std::pair<KEY_TYPE, PAYLOAD_TYPE>& kv, const KEY_TYPE& k) {
            return kv.first < k;
        });
    
    if (it == end || it->first != key) return false;  // 仅在命中时返回 true
    val = it->second;
    return true;
}

template <class KEY_TYPE, class PAYLOAD_TYPE>
bool RSInterface<KEY_TYPE, PAYLOAD_TYPE>::put(KEY_TYPE key,
                                                PAYLOAD_TYPE value,
                                                Param *param) {
  return false;
}

template <class KEY_TYPE, class PAYLOAD_TYPE>
bool RSInterface<KEY_TYPE, PAYLOAD_TYPE>::update(KEY_TYPE key,
                                                   PAYLOAD_TYPE value,
                                                   Param *param) {
  return false;
}

template <class KEY_TYPE, class PAYLOAD_TYPE>
bool RSInterface<KEY_TYPE, PAYLOAD_TYPE>::remove(KEY_TYPE key, Param *param) {
  return false;
}

template <class KEY_TYPE, class PAYLOAD_TYPE>
size_t RSInterface<KEY_TYPE, PAYLOAD_TYPE>::scan(
    KEY_TYPE key_low_bound, size_t key_num,
    std::pair<KEY_TYPE, PAYLOAD_TYPE> *result, Param *param) {
    (void)param;
    if (result == nullptr || key_num == 0) return 0;
    if (data_view_ == nullptr || size_ == 0) return 0;
    
    // 用 RS 取得候选区间
    auto bound = rs_.GetSearchBound(key_low_bound);
    size_t lo = std::min<std::size_t>(bound.begin, size_);
    size_t hi = std::min<std::size_t>(bound.end,   size_);
    if (lo > hi) std::swap(lo, hi);
    
    const auto* begin = data_view_ + lo;
    const auto* end   = data_view_ + hi;
    
    // 在缩小区间内做 lower_bound
    auto it = std::lower_bound(
        begin, end, key_low_bound,
        [](const std::pair<KEY_TYPE, PAYLOAD_TYPE>& kv, const KEY_TYPE& k) {
            return kv.first < k;
        });
    
    // 若 RS 区间内没有找到 >= 下界的元素，继续在全局尾部收敛（避免极端误差）
    if (it == end) {
        it = std::lower_bound(
            data_view_, data_view_ + size_, key_low_bound,
            [](const std::pair<KEY_TYPE, PAYLOAD_TYPE>& kv, const KEY_TYPE& k) {
            return kv.first < k;
            });
        if (it == data_view_ + size_) return 0;
    }
    
    // 拷贝最多 key_num 条
    size_t copied = 0;
    for (; it != data_view_ + size_ && copied < key_num; ++it, ++copied) {
        result[copied] = *it;
    }
        return copied;
}