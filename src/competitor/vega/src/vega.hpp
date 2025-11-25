#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <emmintrin.h>
#include <limits>
#include <immintrin.h>
#include <vector>

#include "pla.hpp"

#define SIMD
#define SIMD512
// #define SIMD256
// #define SIMD128

// #define key32
#define key64

#ifdef SIMD
    #ifdef SIMD128
        #define set1_epi32 _mm_set1_epi32
        #define load_epi32(x) _mm_loadu_si128(reinterpret_cast<const __m128i *>((x)))
        #define cmp_epu32_mask _mm_cmp_epu32_mask
        #define set1_epi64 _mm_set1_epi64x
        #define load_epi64(x) _mm_loadu_si128(reinterpret_cast<const __m128i *>((x)))
        #define cmp_epu64_mask _mm_cmp_epu64_mask
        #define VEGA_SIMD_SIZE 16
    #elif defined SIMD256
        #define set1_epi32 _mm256_set1_epi32
        #define load_epi32(x) _mm256_loadu_si256(reinterpret_cast<const __m256i *>((x)))
        #define cmp_epu32_mask _mm256_cmp_epu32_mask
        #define set1_epi64 _mm256_set1_epi64x
        #define load_epi64(x) _mm256_loadu_si256(reinterpret_cast<const __m256i *>((x)))
        #define cmp_epu64_mask _mm256_cmp_epu64_mask
        #define VEGA_SIMD_SIZE 32
    #else // SIMD512
        #define set1_epi32 _mm512_set1_epi32
        #define load_epi32 _mm512_load_epi32
        #define cmp_epu32_mask _mm512_cmp_epu32_mask
        #define set1_epi64 _mm512_set1_epi64
        #define load_epi64 _mm512_load_epi64
        #define cmp_epu64_mask _mm512_cmp_epu64_mask
        #define VEGA_SIMD_SIZE 64
    #endif

    #ifdef key32
        #define COMPACT_GRANULARITY VEGA_SIMD_SIZE / 4
    #else // key64
        #define COMPACT_GRANULARITY VEGA_SIMD_SIZE / 8
    #endif

    #define define_compact_granularity \
            static constexpr uint64_t compact_granularity = COMPACT_GRANULARITY;

#else
    #define define_compact_granularity \
            static constexpr uint64_t compact_granularity = CompactGranularity;

#endif // SIMD

namespace vega {

using pos_t = uint32_t;
using slope_t = float;
using intercept_t = uint32_t;
using payload_t = uint64_t;

#define VEGA_TEMPLATE_ARGUMENTS                                                                              \
        template <typename KeyType, size_t Epsilon,                                                          \
        size_t SparseGranularity, size_t CompactGranularity, size_t LeafGranularity,                         \
        bool UseLinearSearch>                                                                                \

#define VEGAIndexType                                                                                        \
        VEGAIndex<KeyType, Epsilon, SparseGranularity, CompactGranularity, LeafGranularity, UseLinearSearch> \

struct Model {
    slope_t slope_;
    intercept_t intercept_;
};

/**
 * VEGA: An Active-tuning Learned Index with Group-Wise Learning Granularity
 * 
 * @param KeyType 
 * 
 * @param Epsilon Maximum allowable error
 * 
 * @param xxxGranularity
 *        When @p SIMD optimization is turned on, the @p CompactGranularity template parameter is invalid.
 */
template <typename KeyType, size_t Epsilon = 32, 
          size_t SparseGranularity = 1, size_t CompactGranularity = 8, size_t LeafGranularity = Epsilon, bool UseLinearSearch = false>
class VEGAIndex {
public:
    struct LeafBucket;
    struct LeafLayer;
    
protected:
    struct Segment;
    
    struct SparseBucket;
    struct SparseLayer;
    
    struct CompactBucket;
    struct CompactLayer;

    static constexpr size_t epsilon_recursive = 8;
    static constexpr double buckets_per_key = 0.1;
    static constexpr double bpk_mul = 10.0;
    
    KeyType key0_;
    size_t compact_layer_num_;
    SparseLayer sparse_layer_;
    std::vector<CompactLayer> compact_layers_;
    LeafLayer leaf_layer_;

    define_compact_granularity;

private:
    template<typename RandomIt>
    void BuildNonLeafLayers(RandomIt first, RandomIt last, size_t n) {
        double keys_per_bucket = 1 / buckets_per_key;

        std::vector<Segment> segments;
        segments.reserve(n / (Epsilon * 3));
        
        auto in_fun = [&](auto i) { return std::pair<KeyType, size_t>(first[i], i); };
        auto out_fun = [&](auto cs) {
            auto[cs_slope, cs_intercept] = cs.get_floating_point_segment(cs.get_first_x());
            segments.emplace_back(cs.get_first_x(), cs_slope, cs_intercept);
        };
        size_t segment_num = make_segmentation_at_bucket_level(n, Epsilon, LeafGranularity, in_fun, out_fun);
        if (segment_num > compact_granularity) {
            if (sparse_layer_.FittingDistributionToModel(segments, keys_per_bucket)) {
                return;
            }
        }
        compact_layers_.push_back(CompactLayer());
        compact_layers_[0].FittingModelToDistribution(segments);
        compact_layer_num_ = 1;
        
        size_t bucket_num = compact_layers_[0].BucketNum();
        while (bucket_num > 1) {
            keys_per_bucket *= bpk_mul;
            segments.clear();
            auto in_fun_rec = [&](auto i) {
                return std::pair<KeyType, size_t>(compact_layers_[compact_layer_num_ - 1].BucketAt(i).SmallestKey(), i * compact_granularity);
            };
            segment_num = make_segmentation(bucket_num, epsilon_recursive, in_fun_rec, out_fun);
            if (segment_num > compact_granularity) {
                if (sparse_layer_.FittingDistributionToModel(segments, keys_per_bucket)) {
                    return;
                }
            }
            compact_layers_.push_back(CompactLayer());
            compact_layers_[compact_layer_num_].FittingModelToDistribution(segments);
            bucket_num = compact_layers_[compact_layer_num_].BucketNum();
            ++compact_layer_num_;
        }
    }

public:
    class ConstIterator;

    VEGAIndex() = default;

    template<typename KeyIt, typename PayloadIt>
    explicit VEGAIndex(KeyIt key_first, KeyIt key_last, 
                       PayloadIt payload_first,  PayloadIt payload_last) {
        compact_layer_num_ = 0;

        size_t n = std::distance(key_first, key_last);
        if (n == 0) {
            return;
        }
        key0_ = key_first[0];

        BuildNonLeafLayers(key_first, key_last, n);

        leaf_layer_.Init(key_first, key_last, payload_first, payload_last);
    }

    /**
     * @return Returns the predicted position (rank) of key
    */
    pos_t Predict(const KeyType key) const {
        if (key <= key0_) {
            return 0;
        }
        pos_t predict_pos;
        int L;
        if (sparse_layer_.IsSparse()) {
            predict_pos = sparse_layer_.Predict(key);
            if (compact_layer_num_ == 0) {
                return predict_pos;
            }
            L = compact_layer_num_ - 1;
        } else {
            predict_pos = compact_layers_[compact_layer_num_ - 1].BucketAt(0).Predict(key);
            L = compact_layer_num_ - 2;
        }
        for (; L >= 1; L--) {
            predict_pos = compact_layers_[L].Predict(key, predict_pos / compact_granularity);
        }
        return compact_layers_[0].Predict(key, predict_pos / compact_granularity);
    }

    /**
     * @brief lookup the key, return: 
     *          (1) the payload corresponding to the key that no less than the lookuped key
     *          (2) whether there is the lookuped key
     * @param key lookuped key
     * @param payload [output] the value corresponding to the key that no less than the lookuped key
     * @return whether there exist the lookuped key
     */
    bool Lookup(const KeyType key, payload_t* payload) const {
        auto predict_pos = Predict(key);
        if (UseLinearSearch) {
            return leaf_layer_.Lookup_LS(key, predict_pos, payload); // for epsilon <= 16
        } else {
            return leaf_layer_.Lookup_BS(key, predict_pos, payload);    // for epsilon >= 32
        }
    }

    /**
     * @brief lower bound search, return an iterator to the first key that no less than the lookuped key
     * @param key search key
     * @return the iterator corresponding to the lower bound key
     */
    ConstIterator LowerBound(const KeyType key) const {
        auto predict_pos = Predict(key);
        return leaf_layer_.LowerBound(key, predict_pos);
    }

    /**
     * @brief Return the sum of values in range [lower_key, upper_key).
     * 
     * Interface implemented for benchmarking.
     */
    payload_t RangeQuery(const KeyType lower_key, const KeyType upper_key) const {
        auto lower_predict_pos = Predict(lower_key);
        auto upper_predict_pos = Predict(upper_key);
        return leaf_layer_.RangeScan(lower_predict_pos, upper_predict_pos, lower_key, upper_key);
    }

    ConstIterator Begin() const {
        return ConstIterator(&leaf_layer_, 0);
    }

    ConstIterator End() const {
        return ConstIterator(&leaf_layer_, leaf_layer_.DataNum());
    }

    size_t Height() const {
        return compact_layer_num_ + (sparse_layer_.IsSparse());
    }

    /**
     * @brief Return the index size, not contain data size.
     */
    size_t SizeInByte() const {
        size_t size = 0;
        if (sparse_layer_.IsSparse()) {
            size += sparse_layer_.SizeInByte();
        }
        for (size_t i = 0; i < compact_layer_num_; ++i) {
            size += compact_layers_[i].SizeInByte();
        }
        size += sizeof(key0_) + sizeof(compact_layer_num_) + sizeof(epsilon_recursive)
             + sizeof(buckets_per_key) + sizeof(bpk_mul);
        return size;
    }

    bool HasSparseLayer() const {
        return sparse_layer_.IsSparse();
    }
};

VEGA_TEMPLATE_ARGUMENTS
struct VEGAIndexType::Segment {
    KeyType key_;
    slope_t slope_;
    intercept_t intercept_;

    Segment() = default;
    Segment(KeyType key, slope_t slope, intercept_t intercept) : key_(key), slope_(slope), intercept_(intercept) {}
};

VEGA_TEMPLATE_ARGUMENTS
struct VEGAIndexType::SparseBucket {
    KeyType keys_[SparseGranularity];
    Model payloads_[SparseGranularity];

    SparseBucket() = default;

    void Init(const std::vector<Segment>& segments) {
        size_t n = segments.size();
        size_t i = 0;
        for (; i < n; ++i) {
            keys_[i] = segments[i].key_;
            payloads_[i] = {segments[i].slope_, segments[i].intercept_};
        }
        for (; i < SparseGranularity; ++i) {
            keys_[i] = std::numeric_limits<KeyType>::max();;
        }
    }

    inline KeyType SmallestKey() const {
        return keys_[0];
    }

    inline pos_t Predict(KeyType key) const {
        size_t i = 1;
        for (; i < SparseGranularity; ++i) {
            if (keys_[i] > key) {
                break;
            }
        }
        --i;
        return (key - keys_[i]) * payloads_[i].slope_ + payloads_[i].intercept_;
    }

    static size_t SizeInByte() {
        return (sizeof(KeyType) + sizeof(Model)) * SparseGranularity;
    }
};

VEGA_TEMPLATE_ARGUMENTS
struct VEGAIndexType::SparseLayer {
private:
    double slope_;
    KeyType key0_;
    size_t bucket_num_;
    std::vector<SparseBucket> sparse_buckets_;

    bool SparseOnly(const std::vector<Segment>& segments, 
                    size_t keys_per_bucket, double* slope, KeyType* key0) {
        size_t n = segments.size();
        if (n <= SparseGranularity) {
            return false;
        }

        KeyType min_gap = segments[n - 1].key_ - segments[0].key_;
        KeyType max_gap = segments[n - 1].key_ - segments[0].key_;
        for (size_t i = 0; i < n - SparseGranularity; ++i) {
            min_gap = std::min(min_gap, (segments[i + SparseGranularity].key_ - segments[i].key_));
        }
        if (min_gap == 0) {
            return false;
        }

        double R = 1 + std::ceil((double)max_gap / min_gap);
        double comRatio = R / n;
        if(comRatio > keys_per_bucket) {
            return false;
        }

        *slope = R / max_gap;
        *key0 = segments[0].key_;
        return true;
    }

    void PackSparseBuckets(const std::vector<Segment>& segments) {
        size_t n = segments.size();
        size_t R = std::ceil(slope_ * (segments[n - 1].key_ - key0_));
        sparse_buckets_.resize(R + 1);

        std::vector<Segment> bucket_segs;
        bucket_segs.push_back(segments[0]);
        auto prev_seg = segments[0];
        pos_t prev_pos = 0;
        for (size_t i = 1; i < n; ++i) {
            pos_t pos = std::ceil(slope_ * (segments[i].key_ - key0_));
            if (prev_pos == pos) {
                bucket_segs.push_back(segments[i]);
                prev_seg = segments[i];
            } else {
                sparse_buckets_[prev_pos].Init(bucket_segs);
                bucket_segs.clear();
                bucket_segs.push_back(prev_seg);
                size_t k = prev_pos + 1;
                while (k < pos) {
                    sparse_buckets_[k].Init(bucket_segs);
                    ++k;
                }
                bucket_segs.clear();
                bucket_segs.push_back(segments[i]);
                prev_seg = segments[i];
                prev_pos = pos;
            }
        }
        sparse_buckets_[prev_pos].Init(bucket_segs);
        bucket_num_ = sparse_buckets_.size();
    }

public:
    SparseLayer() : bucket_num_(0) {}

    ~SparseLayer() {
        sparse_buckets_.clear();
        sparse_buckets_.shrink_to_fit();
    }

    bool FittingDistributionToModel(const std::vector<Segment>& segments, size_t keys_per_bucket) {
        if (!SparseOnly(segments, keys_per_bucket, &slope_, &key0_)) {
            return false;
        }
        PackSparseBuckets(segments);
        return true;
    }

    inline pos_t Predict(KeyType key) const {
        pos_t bucket_pos = std::ceil(slope_ * (key - key0_));
        bucket_pos = bucket_pos < bucket_num_ ? bucket_pos : bucket_num_ - 1;
        if (key < sparse_buckets_[bucket_pos].SmallestKey()) {
            --bucket_pos;
        }
        return sparse_buckets_[bucket_pos].Predict(key);
    }

    inline size_t BucketNum() const {
        return bucket_num_;
    }

    inline bool IsSparse() const {
        return bucket_num_ != 0;
    }

    size_t SizeInByte() const {
        size_t size = sizeof(slope_) + sizeof(key0_) + sizeof(bucket_num_);
        size += SparseBucket::SizeInByte() * bucket_num_;
        return size;
    }
};

VEGA_TEMPLATE_ARGUMENTS
struct VEGAIndexType::CompactBucket {
    define_compact_granularity;
    KeyType keys_[compact_granularity] __attribute__((aligned(64)));
    Model payloads_[compact_granularity];

    CompactBucket() = default;

    template<typename RandomIt>
    CompactBucket(RandomIt first, RandomIt last) {
        size_t n = std::distance(first, last) - 1;
        size_t i = 0;
        for (; i <= n; ++i) {
            keys_[i] = first[i].key_;
            payloads_[i] = {first[i].slope_, first[i].intercept_};
        }
        for (; i < compact_granularity; ++i) {
            keys_[i] = first[n].key_;
            payloads_[i] = {first[n].slope_, first[n].intercept_};
        }
    }

    void InitGuard() {
        for (size_t i = 0; i < compact_granularity; ++i) {
            keys_[i] = std::numeric_limits<KeyType>::max();
        }
    }

    inline KeyType SmallestKey() const {
        return keys_[0];
    }

    inline pos_t Predict(KeyType key) const {
        #ifdef SIMD
        
        #ifdef key32
        auto v = set1_epi32(key);
        auto k = load_epi32(keys_);
        auto mask = cmp_epu32_mask(k, v, 2);
        size_t pos = __builtin_popcount(mask) - 1;
        #else
        auto v = set1_epi64(key);
        auto k = load_epi64(keys_);
        auto mask = cmp_epu64_mask(k, v, 2);
        size_t pos = __builtin_popcount(mask) - 1;
        #endif

        #else
        
        size_t pos = 1;
        for (; pos < compact_granularity; ++pos) {
            if (keys_[pos] > key) {
                break;
            }
        }
        --pos;

        #endif
        
        return (key - keys_[pos]) * payloads_[pos].slope_ + payloads_[pos].intercept_;
    }

    static size_t SizeInByte() {
        return (sizeof(KeyType) + sizeof(Model)) * compact_granularity;
    }
};

VEGA_TEMPLATE_ARGUMENTS
struct VEGAIndexType::CompactLayer {
private:
    define_compact_granularity;
    int bucket_num_;
    std::vector<CompactBucket> compact_buckets_;
    
    void PackCompactBuckets(const std::vector<Segment>& segments) {
        size_t n = segments.size();
        bucket_num_ = std::ceil((double)n / compact_granularity);
        compact_buckets_.reserve(bucket_num_ + 1);

        for (int i = 0; i < bucket_num_; ++i) {
            size_t begin = i * compact_granularity;
            size_t end = begin + compact_granularity < n ? begin + compact_granularity : n;
            compact_buckets_.emplace_back(segments.begin() + begin, segments.begin() + end);
        }
    }

public:
    CompactLayer() = default;

    ~CompactLayer() {
        compact_buckets_.clear();
        compact_buckets_.shrink_to_fit();
    }

    bool FittingModelToDistribution(const std::vector<Segment>& segments) {
        PackCompactBuckets(segments);
        return true;
    }

    inline pos_t Predict(KeyType key, pos_t upper_predict_pos) const {
        int bucket_pos = (int)upper_predict_pos < bucket_num_ ? upper_predict_pos : bucket_num_ - 1;
        if (compact_buckets_[bucket_pos].SmallestKey() <= key) {
            ++bucket_pos;
            for (; bucket_pos < bucket_num_; ++bucket_pos) {
                if (compact_buckets_[bucket_pos].SmallestKey() > key) {
                    break;
                }
            }
            --bucket_pos;
        } else {
            --bucket_pos;
            for (; bucket_pos >= 0; --bucket_pos) {
                if (compact_buckets_[bucket_pos].SmallestKey() <= key) {
                    break;
                }
            }
        }
        return compact_buckets_[bucket_pos].Predict(key);
    }

    inline CompactBucket BucketAt(pos_t i) const {
        return compact_buckets_[i];
    }
    
    inline size_t BucketNum() const {
        return bucket_num_;
    }

    inline size_t SizeInByte() const {
        return sizeof(bucket_num_) + bucket_num_ * CompactBucket::SizeInByte();
    }
};

VEGA_TEMPLATE_ARGUMENTS
struct VEGAIndexType::LeafBucket {
    KeyType keys_[LeafGranularity] __attribute__((aligned(64)));
    payload_t payloads_[LeafGranularity];

    #ifdef SIMD
    static constexpr size_t n_simd_ = VEGA_SIMD_SIZE / sizeof(KeyType);
    #endif

    LeafBucket() = default;

    template<typename KeyIt, typename PayloadIt>
    LeafBucket(KeyIt key_first, KeyIt key_last, 
               PayloadIt payload_first,  PayloadIt payload_last) {
        size_t n = std::distance(key_first, key_last);
        size_t i = 0;
        for (; i < n; ++i) {
            keys_[i] = key_first[i];
            payloads_[i] = payload_first[i];
        }
        for (;i < LeafGranularity; ++i) {
            keys_[i] = std::numeric_limits<KeyType>::max();
        }
    }

    inline KeyType SmallestKey() const {
        return keys_[0];
    }

    void InitGuard() {
        for (size_t i = 0; i < LeafGranularity; ++i) {
            keys_[i] = std::numeric_limits<KeyType>::max();
        }
    }

    inline bool Lookup_LS(KeyType key, payload_t* payload) const {
        #ifdef SIMD
        
        #ifdef key32
        const auto keys = set1_epi32(key);
        for (size_t i = 0; i < LeafGranularity; i += n_simd_) {
            auto v = load_epi32(&keys_[i]);
            auto mask = cmp_epu32_mask(keys, v, 0);
            if (mask) {
                *payload = keys_[i + __builtin_ctz(mask)];
                return true;
            }
        }
        #else // key64
        const auto keys = set1_epi64(key);
        for (size_t i = 0; i < LeafGranularity; i += n_simd_) {
            auto v = load_epi64(&keys_[i]);
            auto mask = cmp_epu64_mask(keys, v, 0);
            if (mask) {
                *payload = keys_[i + __builtin_ctz(mask)];
                return true;
            }
        }
        #endif

        #else

        size_t pos = 1;
        for (; pos < LeafGranularity; ++pos) {
            if (keys_[pos] > key) {
                break;
            }
        }
        --pos;
        *payload = payloads_[pos];
        if (keys_[pos] == key) {
            return true;
        }
        
        
        #endif // SIMD

        return false;
    }

    inline bool Lookup_BS(KeyType key, payload_t* payload) const {
        #ifdef SIMD
        
        #ifdef key32
        const auto keys = set1_epi32(key);
        size_t a = 0;
        size_t b = LeafGranularity;
        while (a < b) {
            const int c = (a + b) / 2;
            if (keys_[c] == key) {
                *payload = payloads_[c];
                return true;
            }
            if (key < keys_[c]) {
                b = c;
                if (b >= n_simd_) {
                    auto v = load_epi32(&keys_[c - n_simd_]);
                    auto mask = cmp_epu32_mask(keys, v, 0);
                    if (mask) {
                        *payload = payloads_[b - __builtin_ctz(mask)];
                        return true;
                    }
                }
            } else {
                a = c;
                if (a + n_simd_ <= LeafGranularity) {
                    auto v = load_epi32(&keys_[a]);
                    auto mask = cmp_epu32_mask(keys, v, 0);
                    if (mask) {
                        *payload = payloads_[a + __builtin_ctz(mask)];
                        return true;
                    }
                }
            }
        }
        #else // key64
        const auto k = set1_epi64(key);
        size_t a = 0;
        size_t b = LeafGranularity;
        while (a < b) {
            const size_t c = (a + b) / 2;
            if (keys_[c] == key) {
                *payload = payloads_[c];
                return true;
            }
            if (key < keys_[c]) {
                b = c;
                if (c >= n_simd_) {
                    auto v = load_epi64(&keys_[c - n_simd_]);
                    auto mask = cmp_epu64_mask(k, v, 0);
                    if (mask) {
                        *payload = payloads_[b - __builtin_ctz(mask)];
                        return true;
                    }
                }
            } else {
                a = c;
                if (c + n_simd_ <= LeafGranularity) {
                    auto v = load_epi64(&keys_[c]);
                    auto mask   = cmp_epu64_mask(k, v, 0);
                    if (mask) {
                        *payload = payloads_[a + __builtin_ctz(mask)];
                        return true;
                    }
                }
            }
        }
        #endif

        #else
        
        auto it = std::lower_bound(keys_, keys_ + LeafGranularity, key);
        *payload = *it;
        if (*it == key) {
            return true;
        }

        #endif // SIMD

        return false;
    }

    // return the rank of the first key that no less the lookuped key
    inline pos_t LowerBoundPosition(KeyType key) const {
        #ifdef SIMD
        
        #ifdef key32
        const auto keys = set1_epi32(key);
        for (size_t i = 0; i < LeafGranularity; i += n_simd_) {
            auto v = load_epi32(&keys_[i]);
            auto mask = cmp_epu32_mask(keys, v, 2);
            if (mask != (uint8_t)0) {
                return i + __builtin_ctz(mask);
            }
        }
        #else // key64
        const auto keys = set1_epi64(key);
        for (size_t i = 0; i < LeafGranularity; i += n_simd_) {
            auto v = load_epi64(&keys_[i]);
            auto mask = cmp_epu64_mask(keys, v, 2);
            if (mask != (uint8_t)0) {
                return i + __builtin_ctz(mask);
            }
        }
        #endif

        #else

        size_t pos = 0;
        for (; pos < LeafGranularity; ++pos) {
            if (keys_[pos] >= key) {
                return pos;
            }
        }

        #endif // SIMD

        return LeafGranularity;
    }

    /**
     * @brief Get value sum according to mask.
     * 
     * Interface implemented for benchmarking.
     */
    inline payload_t GetPayloadSum() const {
        payload_t sum = 0;
        // payload only support uint64_t
        for (size_t i = 0; i < LeafGranularity; i += 8) {
            sum += _mm512_reduce_add_epi64(_mm512_load_epi64(&payloads_[i])); 
        }
        return sum;
    }
};

VEGA_TEMPLATE_ARGUMENTS
struct VEGAIndexType::LeafLayer {
    friend class ConstIterator;
private:
    size_t bucket_num_;
    size_t data_num_;
    std::vector<LeafBucket> leaf_buckets_;
    
    inline pos_t LocateBucket(KeyType key, pos_t predict_bucket_pos) const {
        size_t pos = predict_bucket_pos < bucket_num_ ? predict_bucket_pos : bucket_num_ - 1;
        if (leaf_buckets_[pos].SmallestKey() <= key) {
            ++pos;
            for (; pos < bucket_num_; ++pos) {
                if (leaf_buckets_[pos].SmallestKey() > key) {
                    break;
                }
            }
            --pos;
        } else {
            pos = pos == 0 ? 0 : pos - 1;
            for (; pos > 0; --pos) {
                if (leaf_buckets_[pos].SmallestKey() <= key) {
                    break;
                }
            }
        }
        return pos;
    }

    // Returns the bucket position that no keys in next bucket less than the lookuped key.
    inline pos_t LocateLowerBucket(KeyType key, pos_t predict_bucket_pos) const {
        // size_t pos = predict_bucket_pos < bucket_num_ ? predict_bucket_pos : bucket_num_ - 1;
        // if (leaf_buckets_[pos].SmallestKey() <= key) {
        //     ++pos;
        //     for (; pos < bucket_num_; ++pos) {
        //         if (leaf_buckets_[pos].SmallestKey() > key) {
        //             break;
        //         }
        //     }
        //     --pos;
        // } else {
        //     pos = pos == 0 ? 0 : pos - 1;
        //     for (; pos > 0; --pos) {
        //         if (leaf_buckets_[pos].SmallestKey() <= key) {
        //             break;
        //         }
        //     }
        // }
        // return pos;

        size_t pos = predict_bucket_pos < bucket_num_ ? predict_bucket_pos : bucket_num_ - 1;
        if (leaf_buckets_[pos].SmallestKey() < key) {
            ++pos;
            for (; pos < bucket_num_; ++pos) {
                if (leaf_buckets_[pos].SmallestKey() >= key) {
                    break;
                }
            }
            --pos;
        } else {
            pos = pos == 0 ? 0 : pos - 1;
            for (; pos > 0; --pos) {
                if (leaf_buckets_[pos].SmallestKey() < key) {
                    break;
                }
            }
        }
        return pos;
    }

public:
    LeafLayer() = default;

    ~LeafLayer() {
        leaf_buckets_.clear();
        leaf_buckets_.shrink_to_fit();
    }

    template<typename KeyIt, typename PayloadIt>
    void Init(KeyIt key_first, KeyIt key_last, 
              PayloadIt payload_first,  PayloadIt payload_last) {
        size_t n = std::distance(key_first, key_last);
        bucket_num_ = n / LeafGranularity;
        
        bool f = (n % LeafGranularity != 0);
        leaf_buckets_.reserve(bucket_num_ + (size_t)f + 1);
        
        size_t i = 0;
        for (; i < bucket_num_; ++i) {
            leaf_buckets_.emplace_back(
                key_first + i * LeafGranularity, key_first + (i + 1) * LeafGranularity, 
                payload_first + i * LeafGranularity, payload_first + (i + 1) * LeafGranularity         
            );
        }
        
        if (n % LeafGranularity != 0) {
            ++bucket_num_;
            leaf_buckets_.emplace_back(
                key_first + i * LeafGranularity, key_last, 
                payload_first + i * LeafGranularity, payload_last
            );
        }
        data_num_ = n;
    }

    bool Lookup_LS(KeyType key, pos_t predict_pos, payload_t* payload) const {
        pos_t pos = LocateBucket(key, predict_pos / LeafGranularity);
        return leaf_buckets_[pos].Lookup_LS(key, payload);
    }

    bool Lookup_BS(KeyType key, pos_t predict_pos, payload_t* payload) const {
        pos_t pos = LocateBucket(key, predict_pos / LeafGranularity);
        return leaf_buckets_[pos].Lookup_BS(key, payload);
    }

    ConstIterator LowerBound(KeyType key, pos_t predict_pos) const {
        pos_t bucket_idx = LocateLowerBucket(key, predict_pos / LeafGranularity);
        pos_t idx = leaf_buckets_[bucket_idx].LowerBoundPosition(key);
        return ConstIterator(this, bucket_idx * LeafGranularity + idx);
    }

    /**
     * @brief Sum the values between [lower_key, upper_key). 
     * 
     * Interface implemented for benchmarking.
     * Users can implement any range query according to the implementation idea of this function.
     */
    payload_t RangeScan(pos_t predict_lower_key_pos, pos_t predict_upper_key_pos, const KeyType lower_key, const KeyType upper_key) const {
        payload_t sum = 0;
        pos_t lower_bucket_idx = LocateLowerBucket(lower_key, predict_lower_key_pos / LeafGranularity);
        pos_t lower_idx = leaf_buckets_[lower_bucket_idx].LowerBoundPosition(lower_key);
        pos_t upper_bucket_idx = LocateLowerBucket(upper_key, predict_upper_key_pos / LeafGranularity);
        pos_t upper_idx = leaf_buckets_[upper_bucket_idx].LowerBoundPosition(upper_key);

        size_t bucket_idx = lower_bucket_idx;
        size_t idx = lower_idx;
        if (lower_bucket_idx == upper_bucket_idx) {
            while (idx < upper_idx) {
                sum += leaf_buckets_[bucket_idx].payloads_[idx];
                ++idx;
            }
            return sum;
        }
        while (idx < LeafGranularity) {
            sum += leaf_buckets_[bucket_idx].payloads_[idx];
            ++idx;
        }
        ++bucket_idx;
        while (bucket_idx < upper_bucket_idx) {
            // idx = 0;
            // while (idx < LeafGranularity) {
            //     sum += leaf_buckets_[bucket_idx].payloads_[idx];
            //     ++idx;
            // }
            sum += leaf_buckets_[bucket_idx].GetPayloadSum();
            ++bucket_idx;
        }
        idx = 0;
        while (idx < upper_idx) {
            sum += leaf_buckets_[bucket_idx].payloads_[idx];
            ++idx;
        }
        return sum;
    }

    size_t DataNum() const { return data_num_; }
};

VEGA_TEMPLATE_ARGUMENTS
class VEGAIndexType::ConstIterator {
public:
    const LeafLayer *leaf_layer_;
    pos_t cur_;
    
    ConstIterator() {}

    ConstIterator(pos_t cur) : cur_(cur) {}

    ConstIterator(const LeafLayer *leaf_layer, pos_t cur)
        : leaf_layer_(leaf_layer),
          cur_(cur) {}

    ConstIterator(const ConstIterator &other)
        : leaf_layer_(other.leaf_layer_),
          cur_(other.cur_) {}
    
    ConstIterator& operator=(const ConstIterator &other) {
        if (this != &other) {
            leaf_layer_ = other.leaf_layer_;
            cur_ = other.cur_;
        }
        return *this;
    }

    ConstIterator& operator++() {
        cur_++;
        return *this;
    }

    ConstIterator& operator++(int step) {
        if (step < 0) {
            cur_ = cur_ > -step ? cur_ + step : 0;
        } else {
            cur_ += step;
        }
        return *this;
    }

    const payload_t &operator*() const {
        return payload();
    }

    const KeyType &key() const {
        auto bucket_idx = cur_ / LeafGranularity;
        auto idx = cur_ % LeafGranularity;
        return leaf_layer_->leaf_buckets_[bucket_idx].keys_[idx];
    }

    const payload_t &payload() const {
        auto bucket_idx = cur_ / LeafGranularity;
        auto idx = cur_ % LeafGranularity;
        return leaf_layer_->leaf_buckets_[bucket_idx].payloads_[idx];
    }

    bool operator==(ConstIterator &rhs) const {
        return cur_ == rhs.cur_;
    }

    bool operator!=(ConstIterator &rhs) const {
        return cur_ != rhs.cur_;
    }

    bool is_end() const {
        return cur_ >= leaf_layer_->data_num_;
    }
};

} // vega