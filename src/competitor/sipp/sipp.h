#include "../indexInterface.h"
#include "./src/src/core/sipp.h"


template <class KEY_TYPE, class PAYLOAD_TYPE>
class SIPPInterface final : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
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

  long long memory_consumption() { return sipp.total_size(); }

  void print_stats(std::string s) {
    if (s == "bulkload") {
      sipp.print_depth_stats(s);
      sipp.print_hist_model_stats(s);
      // sipp.print_information_gain(s);
      // sipp.print_node_size(s);
      sipp.print_size_stats(s);
    }
    if (s == "insert") {
      sipp.print_depth_stats(s);
      sipp.print_hist_model_stats(s);
      // sipp.print_information_gain(s);
      sipp.print_smo_stats(s);
      // sipp.print_node_size(s);
      sipp.print_size_stats(s);
    }
    return ;
  }

private:
  SIPP<KEY_TYPE, PAYLOAD_TYPE> sipp;
};

template <class KEY_TYPE, class PAYLOAD_TYPE>
void SIPPInterface<KEY_TYPE, PAYLOAD_TYPE>::bulk_load(
    std::pair<KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num, Param *param) {
  sipp.bulk_load(key_value, static_cast<int>(num));
}

template <class KEY_TYPE, class PAYLOAD_TYPE>
bool SIPPInterface<KEY_TYPE, PAYLOAD_TYPE>::get(KEY_TYPE key, PAYLOAD_TYPE &val,
                                                Param *param) {
  bool exist;
  val = sipp.at(key, false, exist);
  return exist;
}

template <class KEY_TYPE, class PAYLOAD_TYPE>
bool SIPPInterface<KEY_TYPE, PAYLOAD_TYPE>::put(KEY_TYPE key,
                                                PAYLOAD_TYPE value,
                                                Param *param) {
  return sipp.insert(key, value);
}

template <class KEY_TYPE, class PAYLOAD_TYPE>
bool SIPPInterface<KEY_TYPE, PAYLOAD_TYPE>::update(KEY_TYPE key,
                                                   PAYLOAD_TYPE value,
                                                   Param *param) {
  return sipp.update(key, value);
}

template <class KEY_TYPE, class PAYLOAD_TYPE>
bool SIPPInterface<KEY_TYPE, PAYLOAD_TYPE>::remove(KEY_TYPE key, Param *param) {
  return sipp.remove(key);
}

template <class KEY_TYPE, class PAYLOAD_TYPE>
size_t SIPPInterface<KEY_TYPE, PAYLOAD_TYPE>::scan(
    KEY_TYPE key_low_bound, size_t key_num,
    std::pair<KEY_TYPE, PAYLOAD_TYPE> *result, Param *param) {
  if (!result) {
    result = new std::pair<KEY_TYPE, PAYLOAD_TYPE>[key_num];
  }
  return sipp.range_query_len(result, key_low_bound, key_num);
}