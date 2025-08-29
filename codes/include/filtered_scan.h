#ifndef FILTERED_BRUTEFORCE_H
#define FILTERED_BRUTEFORCE_H

#include "storage.h"
#include "trie.h"
#include "distance.h"


namespace ANNS {

    class FilteredScan {
        public:
            FilteredScan() = default;
            ~FilteredScan() = default;

            // for baseline: process each query independently
            float search(std::shared_ptr<IStorage> base_storage, std::shared_ptr<IStorage> query_storage,
                         std::shared_ptr<DistanceHandler> distance_handler, std::string scenario, 
                         uint32_t num_threads, IdxType K, std::pair<IdxType, float>* results);

            // for computing groundtruth: process all query with the same label set together
            void run(std::shared_ptr<IStorage> base_storage, std::shared_ptr<IStorage> query_storage,
                     std::shared_ptr<DistanceHandler> distance_handler, std::string scenario, 
                     uint32_t num_threads, IdxType K, std::pair<IdxType, float>* results);

        private:

            // data
            std::shared_ptr<IStorage> _base_storage, _query_storage;    // 基础数据和查询数据的存储
            std::shared_ptr<DistanceHandler> _distance_handler;         // 距离计算处理对象
            std::pair<IdxType, float>* _results;                        // 向量 id 和 距离
            IdxType _K;                                                 // 查询需返回的最近邻的数量

            // trie index for label sets
            TrieIndex base_trie_index, query_trie_index;                // 存储数据和查询数据的标签的索引
            std::vector<std::vector<LabelType>> query_group_id_to_label_set;    // 查询的每个 group 对应的标签集合
            std::vector<std::vector<IdxType>> base_group_id_to_vec_ids, query_group_id_to_vec_ids;  // 树索引中每个 group 对应的向量，group id 从 1 开始

            // help function for answering all queries
            void init_trie_index(bool for_query=true);
            void compute_base_super_sets(std::string scenario, const std::vector<LabelType>& query_label_set, 
                                         std::vector<IdxType>& base_super_set_group_ids);
            float answer_one_query(IdxType query_vec_id, const std::vector<IdxType>& base_super_set_group_ids);
    };
}

#endif // FILTERED_BRUTEFORCE_H