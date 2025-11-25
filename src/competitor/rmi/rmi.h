#include "../indexInterface.h"
#include "all_rmis.h"
#include "../vega/src/util.h"

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>
#include <cstddef>

#include "searches/linear_search_avx.h"

template <class KEY_TYPE, class PAYLOAD_TYPE>
class RMIInterface final : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
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

  long long memory_consumption() { return libio_200M_uint64_1::RMI_SIZE + (sizeof(KEY_TYPE) + sizeof(PAYLOAD_TYPE)) * size_; }

  void print_stats(std::string s) {
    return ;
  }

private:
    // rs::RadixSpline<KEY_TYPE> rs_;
    std::size_t rmi_size = 0;
    std::size_t size_;
    uint64_t build_time_;
    const std::pair<KEY_TYPE, PAYLOAD_TYPE>* data_view_;
};

template <class KEY_TYPE, class PAYLOAD_TYPE>
void RMIInterface<KEY_TYPE, PAYLOAD_TYPE>::bulk_load(
    std::pair<KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num, Param *param) {
    // // Define a local load function (dummy implementation assuming success;
    // // replace with actual loading logic if needed, e.g., from generated RMI headers)
    // auto local_rmi_load = [](const char* path) -> bool {
    //     // Here you can implement the actual loading, for example:
    //     // Call the generated load function like wiki::load(path);
    //     // For now, assume it's always successful or handle as needed.
    //     // If you have the namespace from generated headers, use it.
    //     return true;  // Dummy: assume loaded successfully
    // };
    const std::string rmi_path =
        (std::getenv("RMI_PATH") == NULL ? "src/competitor/rmi/rmi_data"
                                              : std::getenv("RMI_PATH"));
    // printf("Loading RMI data from path: %s\n", rmi_path.c_str());
    // libio_200M_uint64_{variant_} 其中variant_这是一个整数成员变量（或模板参数，如 rmi_variant），代表 RMI 模型的 "变体" 或 "Pareto 配置"（Pareto 指性能-大小-准确度的权衡曲线）。RMI 构建过程会生成多个模型变体（通常 0 到 6），每个变体对应不同的 hyperparameters（如层数、模型复杂度、误差界限）。这些变体是从参数网格搜索中选出的 Pareto 最优解，用于基准测试不同 tradeoff：
    // 低值（如 0）：通常是简单模型，构建时间短、模型大小小，但误差界限较大（搜索范围更宽，查找速度稍慢）。
    // 中值（如 1-3）：平衡配置，模型大小中等，误差小，适合大多数场景。
    // 高值（如 4-6）：复杂模型，构建时间长、模型大小大，但误差最小（搜索范围窄，查找最快）。
    if (!libio_200M_uint64_1::load(rmi_path.c_str())) {
      std::cerr <<
          "Could not load RMI data from rmi_data/ -- either an allocation "
          "failed or the file could not be read." << std::endl;
      exit(1);
    }

    build_time_ = libio_200M_uint64_1::BUILD_TIME_NS + timing([&] {
      data_view_ = key_value;
      size_ = num;
    });
}

template <class KEY_TYPE, class PAYLOAD_TYPE>
bool RMIInterface<KEY_TYPE, PAYLOAD_TYPE>::get(KEY_TYPE key, PAYLOAD_TYPE &val,
                                                Param *param) {
    (void)param;
    if (data_view_ == nullptr || size_ == 0) return false;

    // Define a local lookup function (assuming rmi::lookup is the intended function;
    // replace with actual implementation if different, e.g., from generated RMI headers)
    auto local_rmi_lookup = [](uint64_t key, size_t* error) -> uint64_t {
        // Here you can implement or call the actual lookup logic.
        // For example, assuming it's from the rmi namespace:
        return libio_200M_uint64_1::lookup(key, error);  // Use the actual function call
        // If no namespace, or custom: return your_custom_lookup(key, error);
    };

    size_t error;
    uint64_t guess = local_rmi_lookup(static_cast<uint64_t>(key), &error);

    size_t start = (guess < error ? 0 : guess - error);
    size_t stop = (guess + error >= size_ ? size_ : guess + error);

    const auto* begin = data_view_ + start;
    const auto* end = data_view_ + stop;

    // auto it = LinearSearch<0>::lower_bound(
    //     begin, end, key,
    //     [](const std::pair<KEY_TYPE, PAYLOAD_TYPE>& kv, const KEY_TYPE& k) {
    //         return kv.first < k;
    //     });

    // LinearSearch<0> 中的模板参数 <record> 的作用和含义
    // <0>：通常表示默认或标准线性搜索变体
    // <1>：可能表示启用分支化（branching）或优化的线性搜索变体。
    auto it = LinearSearch<0>::lower_bound(
        begin, end, key,
        data_view_ + guess,  // hint：绝对迭代器位置
        std::function<KEY_TYPE(decltype(begin))>([](decltype(begin) it) -> KEY_TYPE {
            return it->first;  // 从 pair 中提取键 (first)
        })
    );

    if (it == end || it->first != key) return false;
    val = it->second;
    return true;
}

template <class KEY_TYPE, class PAYLOAD_TYPE>
bool RMIInterface<KEY_TYPE, PAYLOAD_TYPE>::put(KEY_TYPE key,
                                                PAYLOAD_TYPE value,
                                                Param *param) {
  return false;
}

template <class KEY_TYPE, class PAYLOAD_TYPE>
bool RMIInterface<KEY_TYPE, PAYLOAD_TYPE>::update(KEY_TYPE key,
                                                   PAYLOAD_TYPE value,
                                                   Param *param) {
  return false;
}

template <class KEY_TYPE, class PAYLOAD_TYPE>
bool RMIInterface<KEY_TYPE, PAYLOAD_TYPE>::remove(KEY_TYPE key, Param *param) {
  return false;
}

template <class KEY_TYPE, class PAYLOAD_TYPE>
size_t RMIInterface<KEY_TYPE, PAYLOAD_TYPE>::scan(
    KEY_TYPE key_low_bound, size_t key_num,
    std::pair<KEY_TYPE, PAYLOAD_TYPE> *result, Param *param) {
    (void)param;
    if (data_view_ == nullptr || size_ == 0)
        return 0;
    
    auto local_rmi_lookup = [](uint64_t key, size_t *error) -> uint64_t {
        return libio_200M_uint64_1::lookup(key, error);
    };
    
    size_t error = 0;
    uint64_t guess = local_rmi_lookup(static_cast<uint64_t>(key_low_bound), &error);
    
    uint64_t start = (guess < error ? 0 : guess - error);
    uint64_t stop = (guess + error >= size_ ? size_ : guess + error);
    
    const auto *begin = data_view_ + start;
    const auto *end = data_view_ + stop;
    
    using Iterator = const std::pair<KEY_TYPE, PAYLOAD_TYPE> *;
    auto it = LinearSearch<0>::lower_bound(
        begin, end, key_low_bound, data_view_ + guess,
        std::function<KEY_TYPE(Iterator)>(
            [](Iterator it) -> KEY_TYPE { return it->first; }));
    
    size_t scanned = 0;
    while (it != (data_view_ + size_) && scanned < key_num) {
        result[scanned] = *it;
        ++it;
        ++scanned;
    }
    
    return scanned;
}