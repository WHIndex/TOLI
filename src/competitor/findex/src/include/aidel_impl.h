#ifndef __AIDEL_IMPL_H__
#define __AIDEL_IMPL_H__

#include "aidel.h"
#include "util.h"
#include "aidel_model.h"
#include "aidel_model_impl.h"
#include "piecewise_linear_model.h"

namespace aidel {

template<class key_t, class val_t>
inline AIDEL<key_t, val_t>::AIDEL()
    : maxErr(64), learning_step(1000), learning_rate(0.1)
{
    //root = new root_type();
}

template<class key_t, class val_t>
inline AIDEL<key_t, val_t>::AIDEL(int _maxErr, int _learning_step, float _learning_rate)
    : maxErr(_maxErr), learning_step(_learning_step), learning_rate(_learning_rate)
{
    //root = new root_type();
}

template<class key_t, class val_t>
AIDEL<key_t, val_t>::~AIDEL(){
    //root = nullptr;
}

// ====================== train models ========================
template<class key_t, class val_t>
void AIDEL<key_t, val_t>::train(const std::vector<key_t> &keys, 
                                const std::vector<val_t> &vals, size_t _maxErr)
{
    assert(keys.size() == vals.size());
    maxErr = _maxErr;
    // std::cout<<"training begin, length of training_data is:" << keys.size() <<" ,maxErr: "<< maxErr << std::endl;

    size_t start = 0;
    size_t end = learning_step<keys.size()?learning_step:keys.size();
    double last_progress = 0.0;
    while(start<end){
        // COUT_THIS("start:" << start<<" ,end: "<<end);
        lrmodel_type model;
        model.train(keys.begin()+start, end-start);
        size_t err = model.get_maxErr();
        // equal
        if(err == maxErr) {
            // COUT_THIS("err=:" << err <<end);
            append_model(model, keys.begin()+start, vals.begin()+start, end-start, err);
        } else if(err < maxErr) {
            // COUT_THIS("err<:" << err <<end);
            if(end>=keys.size()){
                append_model(model, keys.begin()+start, vals.begin()+start, end-start, err);
                break;
            }
            end += learning_step;
            if(end>keys.size()){
                end = keys.size();
            }
            continue;
        } else {
            // COUT_THIS("err:" << err <<end);
            // std::cout << "Entering backward_train: current err=" << err << ", segment size=" << (end - start) << ", step=" << int(learning_step * learning_rate) << std::endl;
            size_t offset = backward_train(keys.begin()+start, vals.begin()+start, end-start, int(learning_step*learning_rate));
			end = start + offset;
            // std::cout << "Exited backward_train: new offset=" << offset << ", new end=" << end << std::endl;
        }
        // COUT_THIS("learning_step:" << learning_step <<end);
        start = end;
        end += learning_step;
        if(end>=keys.size()){
            end = keys.size();
        }
        // // 循环末尾添加进度打印
        // double current_progress = static_cast<double>(start) / keys.size() * 100.0;
        // if (current_progress - last_progress >= 1.0) {  // 每 1% 打印一次
        //     std::cout << "Training progress: " << current_progress << "% (start=" << start << "/" << keys.size() << ")" << std::endl;
        //     last_progress = current_progress;
        // }
    }

    //root = new root_type(model_keys);
    COUT_THIS("[aidle] get models -> "<< model_keys.size());
    assert(model_keys.size()==aimodels.size());
}

template<class key_t, class val_t>
size_t AIDEL<key_t, val_t>::backward_train(const typename std::vector<key_t>::const_iterator &keys_begin, 
                                           const typename std::vector<val_t>::const_iterator &vals_begin,
                                           uint32_t size, int step)
{   
    // std::cout << "backward_train called: size=" << size << ", step=" << step << std::endl;
    
    if(size<=10){
        step = 1;
    } else {
        while(size<=step){
            step = int(step*learning_rate);
            // step = double(step*learning_rate);
        }
    }
    if (step < 1) {
        step = 1;
        // break;  // 避免无限调整
    }
    assert(step>0);
    // std::cout << "after assert backward_train called: size=" << size << ", step=" << step << std::endl;
    size_t start = 0;
    size_t end = size-step;
    int iter_count = 0;  // 新增：迭代计数器，控制打印频率
    while(end>0){
        lrmodel_type model;
        model.train(keys_begin, end);
        size_t err = model.get_maxErr();

        // // 新增：每 10 次迭代打印一次当前状态，避免日志过多
        // if (++iter_count % 1000000 == 0) {
        //     std::cout << "backward_train iter " << iter_count << ": end=" << end << ", err=" << err << ", step=" << step << std::endl;
        // }

        if(err<=maxErr){
            append_model(model, keys_begin, vals_begin, end, err);
            return end;
        }
        // end -= step;
        if (step < end){
            end -= step;
        } else{
            break;
        }
    }
    end = backward_train(keys_begin, vals_begin, end, int(step*learning_rate));
	return end;
}

template<class key_t, class val_t>
void AIDEL<key_t, val_t>::append_model(lrmodel_type &model, 
                                       const typename std::vector<key_t>::const_iterator &keys_begin, 
                                       const typename std::vector<val_t>::const_iterator &vals_begin, 
                                       size_t size, int err)
{
    key_t key = *(keys_begin+size-1);
    
    // set learning_step
    int n = size/10;
    learning_step = 1;
    while(n!=0){
        n/=10;
        learning_step*=10;
    }
     
    assert(err<=maxErr);
    aidelmodel_type aimodel(model, keys_begin, vals_begin, size, maxErr);

    model_keys.push_back(key);
    aimodels.push_back(aimodel);
}

template<class key_t, class val_t>
typename AIDEL<key_t, val_t>::aidelmodel_type* AIDEL<key_t, val_t>::find_model(const key_t &key)
{
    // root 
    size_t model_pos = binary_search_branchless(&model_keys[0], model_keys.size(), key);
    if(model_pos >= aimodels.size())
        model_pos = aimodels.size()-1;
    return &aimodels[model_pos];
}


// ===================== print data =====================
template<class key_t, class val_t>
void AIDEL<key_t, val_t>::print_models()
{
    
    for(int i=0; i<model_keys.size(); i++){
        std::cout<<"model "<<i<<" ,key:"<<model_keys[i]<<" ->";
        aimodels[i].print_model();
    }

    
    
}

template<class key_t, class val_t>
void AIDEL<key_t, val_t>::self_check()
{
    for(int i=0; i<model_keys.size(); i++){
        aimodels[i].self_check();
    }
    
}


// =================== search the data =======================
template<class key_t, class val_t>
inline result_f AIDEL<key_t, val_t>::find(const key_t &key, val_t &val)
{   
    /*size_t model_pos = root->find(key);
    if(model_pos >= aimodels.size())
        model_pos = aimodels.size()-1;
    return aimodels[model_pos].con_find(key, val);*/

    return find_model(key)[0].con_find_retrain(key, val);

}


// =================  scan ====================
template<class key_t, class val_t>
int AIDEL<key_t, val_t>::scan(const key_t &key, const size_t n, std::vector<std::pair<key_t, val_t>> &result)
{
    size_t remaining = n;
    size_t model_pos = binary_search_branchless(&model_keys[0], model_keys.size(), key);
    if(model_pos >= aimodels.size())
        model_pos = aimodels.size()-1;
    while(remaining>0 && model_pos < aimodels.size()){
        remaining = aimodels[model_pos].scan(key, remaining, result);
    }
    return remaining;
}



// =================== insert the data =======================
template<class key_t, class val_t>
inline result_f AIDEL<key_t, val_t>::insert(
        const key_t& key, const val_t& val)
{
    return find_model(key)[0].con_insert_retrain(key, val);
    //return find_model(key)[0].con_insert(key, val);
}


// ================ update =================
template<class key_t, class val_t>
inline result_f AIDEL<key_t, val_t>::update(
        const key_t& key, const val_t& val)
{
    return find_model(key)[0].update(key, val);
    //return find_model(key)[0].con_insert(key, val);
}


// ==================== remove =====================
template<class key_t, class val_t>
inline result_f AIDEL<key_t, val_t>::remove(const key_t& key)
{
    return find_model(key)[0].remove(key);
    //return find_model(key)[0].con_insert(key, val);
}

// ========================== using OptimalLPR train the model ==========================
template<class key_t, class val_t>
void AIDEL<key_t, val_t>::train_opt(const std::vector<key_t> &keys, 
                                    const std::vector<val_t> &vals, size_t _maxErr)
{
    using pair_type = typename std::pair<size_t, size_t>;

    assert(keys.size() == vals.size());
    maxErr = _maxErr;
    std::cout<<"training begin, length of training_data is:" << keys.size() <<" ,maxErr: "<< maxErr << std::endl;

    segments = make_segmentation(keys.begin(), keys.end(), maxErr);
    COUT_THIS("[aidle] get models -> "<< segments.size());

    /*
    // ===== predict the positions ===========
    std::vector<pair_type> predicts;
    predicts.reserve(keys.size());

    auto it = segments.begin();
    auto [slope, intercept] = it->get_floating_point_segment(it->get_first_x());
    //COUT_THIS(slope<<", "<<intercept<<", "<<keys[0]<<", "<<it->get_first_x());

    for (auto i = 0; i < keys.size(); ++i) {
        if (i != 0 && keys[i] == keys[i - 1])
            continue;
        if (std::next(it) != segments.end() && std::next(it)->get_first_x() <= keys[i]) {
            ++it;
            std::tie(slope, intercept) = it->get_floating_point_segment(it->get_first_x());
        }

        auto pos = (keys[i] - it->get_first_x()) * slope + intercept;
        pos = pos<=0? 0:pos;
        size_t e;
        if(i>pos) e = i-pos;
        else e=pos-i;
        predicts.push_back(pair_type(pos, e));
        // assert(e <= maxErr + 1);
    }

    //assert(model_keys.size()==aimodels.size());
    return predicts;*/
}

template<class key_t, class val_t>
size_t AIDEL<key_t, val_t>::model_size(){
    return segments.size();
}

template<class key_t, class val_t>
size_t AIDEL<key_t, val_t>::size() const{
    size_t size = sizeof(*this) + sizeof(key_t) * model_keys.size() + sizeof(canonical_segment) * segments.size();
    for (size_t i = 0; i < aimodels.size(); ++ i){
        size += aimodels[i].size();
    }
    return size;
}

}    // namespace aidel


#endif