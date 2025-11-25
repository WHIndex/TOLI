#ifndef WRAPPER_VEGA_H
#define WRAPPER_VEGA_H

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cstring>

#include "./util.h"
#include "./vega.hpp"

template <typename KeyType, typename ValueType, size_t Epsilon = 32, 
          size_t SparseGranularity = 1, size_t CompactGranularity = 8, size_t LeafGranularity = Epsilon, bool UseLinearSearch = false>
class VEGA {
public:
	VEGA (const std::vector<KeyValue<KeyType>>& data) : data_(data) {
		size_ = data.size();
		std::vector<KeyType> keys;
		std::vector<ValueType> values;
		keys.reserve(size_);
		values.reserve(size_);
		for (auto kv : data) {
			keys.push_back(kv.key);
			values.push_back(kv.value);
		}

		build_time_ = timing([&] { 
			vega_ = decltype(vega_)(keys.begin(), keys.end(), values.begin(), values.end()); }
		);
	}

	KeyType Lookup(const KeyType key, ValueType *value) const {
		vega::payload_t v;
		bool r = vega_.Lookup(key, &v);
		return r ? key : 0;
	}

	SearchRange Predict(const KeyType key) const {
		uint64_t pos = vega_.Predict(key);
		uint64_t lo = pos < Epsilon ? 0 : pos - Epsilon;
		uint64_t hi = pos + Epsilon + 2 < size_ ? size_ : pos + Epsilon + 2;
		return {lo, hi, pos};
	}

	KeyType LookupErrorDistance(const KeyType key, ValueType *value, size_t *distance) const {
		auto p = vega_.Predict(key);
		if (p >= data_.size()) {
			p = data_.size() - 1;
		}
		size_t r;
		if (data_[p].key < key) {
			for (int i = p; i < int(data_.size()); i++) {
				if (data_[i].key >= key) {
					r = i;
					break;
				}
			}
		} else {
			for (int i = p; i >= 0; i--) {
				if (data_[i].key <= key) {
					r = i;
					break;
				}
			}
		}
		*value = data_[r].value;
		*distance = p > r ? p - r : r - p;

		return data_[r].key;
	}

	uint64_t RangeQuery(const KeyType lower_key, const KeyType upper_key) const {
		return vega_.RangeQuery(lower_key, upper_key);
	}

	std::string info() const {
		return "VEGA, epsilon: " + std::to_string(Epsilon) +
			   ", SparseGranularity: " + std::to_string((size_t)SparseGranularity) +
			   ", CompactGranularity: " + std::to_string((size_t)CompactGranularity) +
			   ", LeafGranularity: " + std::to_string((size_t)LeafGranularity) + 
			   ", Search: " + (UseLinearSearch ? "Linear" : "Binary") +
			   ", n_Layer: " + std::to_string(vega_.Height()); 
	}

	// index size
	std::size_t size() const { return vega_.SizeInByte(); }

	uint64_t build_time() const { return build_time_; }

private:
	vega::VEGAIndex<KeyType, Epsilon, SparseGranularity, CompactGranularity, LeafGranularity, UseLinearSearch> vega_;
	const std::vector<KeyValue<KeyType>> &data_;
	std::size_t size_;
	uint64_t build_time_;
};

#endif // WRAPPER_VEGA_H