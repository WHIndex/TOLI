#include "../indexInterface.h"
#include "./src/vega.hpp"
#include "./src/util.h"

template <class KEY_TYPE, class PAYLOAD_TYPE>
class VEGAInterface final : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
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

  long long memory_consumption() { return vega_.SizeInByte(); }

  void print_stats(std::string s) {
    return ;
  }

private:
    vega::VEGAIndex<KEY_TYPE, 32, 1, 8, 32, false> vega_;
    std::size_t size_;
    uint64_t build_time_;
    const std::pair<KEY_TYPE, PAYLOAD_TYPE>* data_view_;
};

template <class KEY_TYPE, class PAYLOAD_TYPE>
void VEGAInterface<KEY_TYPE, PAYLOAD_TYPE>::bulk_load(
    std::pair<KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num, Param *param) {

    (void)param;  // 暂未使用，避免未使用参数告警

    // 尺寸记录
    size_ = num;

    data_view_ = key_value;
    
    if (key_value == nullptr || num == 0) {
        // 视需要清空/重置 vega_（取决于你的 vega_ 类型是否支持缺省构造）
        // vega_ = decltype(vega_)();
        build_time_ = 0.0;
        return;
    }
    
    // 拆分为 keys / values
    std::vector<KEY_TYPE> keys;
    std::vector<PAYLOAD_TYPE> values;
    keys.reserve(num);
    values.reserve(num);
    
    for (size_t i = 0; i < num; ++i) {
        keys.push_back(key_value[i].first);
        values.push_back(key_value[i].second);
    }
    
    // 计时并构建底层 vega_ 索引
    build_time_ = timing([&] {
        vega_ = decltype(vega_)(keys.begin(), keys.end(),
                                values.begin(), values.end());
    });
}

template <class KEY_TYPE, class PAYLOAD_TYPE>
bool VEGAInterface<KEY_TYPE, PAYLOAD_TYPE>::get(KEY_TYPE key, PAYLOAD_TYPE &val,
                                                Param *param) {
//   bool r = vega_.Lookup(key, &val);
//   return r ? key : 0;
  return vega_.Lookup(key, &val); 
}

template <class KEY_TYPE, class PAYLOAD_TYPE>
bool VEGAInterface<KEY_TYPE, PAYLOAD_TYPE>::put(KEY_TYPE key,
                                                PAYLOAD_TYPE value,
                                                Param *param) {
  return false;
}

template <class KEY_TYPE, class PAYLOAD_TYPE>
bool VEGAInterface<KEY_TYPE, PAYLOAD_TYPE>::update(KEY_TYPE key,
                                                   PAYLOAD_TYPE value,
                                                   Param *param) {
  return false;
}

template <class KEY_TYPE, class PAYLOAD_TYPE>
bool VEGAInterface<KEY_TYPE, PAYLOAD_TYPE>::remove(KEY_TYPE key, Param *param) {
  return false;
}

template <class KEY_TYPE, class PAYLOAD_TYPE>
size_t VEGAInterface<KEY_TYPE, PAYLOAD_TYPE>::scan(
    KEY_TYPE key_low_bound, size_t key_num,
    std::pair<KEY_TYPE, PAYLOAD_TYPE> *result, Param *param) {
    // return vega_.RangeQuery(key_low_bound, key_upper_bound);
    (void)param;
    if (key_num == 0 || result == nullptr) return 0;
    if (data_view_ == nullptr || size_ == 0) return 0;
  
    // 在保存的“有序视图”里找第一个 >= key_low_bound 的位置
    const auto* begin = data_view_;
    const auto* end   = data_view_ + size_;
  
    auto it = std::lower_bound(
        begin, end, key_low_bound,
        [](const std::pair<KEY_TYPE, PAYLOAD_TYPE>& kv, const KEY_TYPE& key) {
          return kv.first < key;
        });
  
    if (it == end) return 0; // 下界在所有键之后
  
    // 计算上界索引（包含上界），即从 it 开始向后 key_num-1
    size_t start_idx = static_cast<size_t>(it - begin);
    size_t upper_idx = start_idx + (key_num == 0 ? 0 : (key_num - 1));
    if (upper_idx >= size_) upper_idx = size_ - 1;
  
    KEY_TYPE key_upper_bound = begin[upper_idx].first; // 包含型上界
  
    return vega_.RangeQuery(key_low_bound, key_upper_bound);
}