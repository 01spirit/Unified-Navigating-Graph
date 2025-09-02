#ifndef UNG_H
#define UNG_H

#include "trie.h"
#include "graph.h"
#include "storage.h"
#include "distance.h"
#include "search_cache.h"
#include "label_nav_graph.h"
#include "vamana/vamana.h"

/*
 * 每个 group 的 storage 和 graph 是根据 new vec id range 截取的内存中向量的子序列，
 * storage 和 vamana graph index 使用的是从 0 开始的局部索引
 */

namespace ANNS {

    class UniNavGraph {
        public:
            UniNavGraph() = default;
            ~UniNavGraph() = default;

        /*
         * @param
         * scenario: 过滤场景的类型
         * num_cross_edges: 跨组边的数量
         * Lbuild: 构件图时候选集的大小
         * alpha: 控制图的稀疏性
         */
            void build(std::shared_ptr<IStorage> base_storage, std::shared_ptr<DistanceHandler> distance_handler, 
                       std::string scenario, std::string index_name, uint32_t num_threads, IdxType num_cross_edges,
                       IdxType max_degree, IdxType Lbuild, float alpha);

        /*
         * @param
         * Lsearch: 搜索时候选集的大小
         * num_entry_points: 入口点数量
         */
            void search(std::shared_ptr<IStorage> query_storage, std::shared_ptr<DistanceHandler> distance_handler, 
                        uint32_t num_threads, IdxType Lsearch, IdxType num_entry_points, std::string scenario,
                        IdxType K, std::pair<IdxType, float>* results, std::vector<float>& num_cmps);

            // I/O
        /*
         * @param
         * index_path_prefix: 索引文件目录前缀
         */
            void save(std::string index_path_prefix);
            void load(std::string index_path_prefix, const std::string& data_type);

        private:

            // data
            std::shared_ptr<IStorage> _base_storage, _query_storage;    // 存储基础数据和查询数据
            std::shared_ptr<DistanceHandler> _distance_handler;         // 距离计算的处理对象
            std::shared_ptr<Graph> _graph;                              // 统一导航图 UNG
            IdxType _num_points;                                        // 基础数据中向量数量

            // trie index and vector groups
            IdxType _num_groups;                                        // 向量 group 的数量
            TrieIndex _trie_index;                                      // 前缀树索引
            std::vector<IdxType> _new_vec_id_to_group_id;               // 排序后的新向量对应的 group id
            std::vector<std::vector<IdxType>> _group_id_to_vec_ids;     // group 中的一组向量，是重排序前的基础数据
            std::vector<std::vector<LabelType>> _group_id_to_label_set; // group 对应的标签集合
            // 构建前缀树，划分 group
            void build_trie_and_divide_groups();

            // label navigating graph   标签导航图
            std::shared_ptr<LabelNavGraph> _label_nav_graph = nullptr;
            void get_min_super_sets(const std::vector<LabelType>& query_label_set, std::vector<IdxType>& min_super_set_ids, 
                                    bool avoid_self=false, bool need_containment=true);
            void build_label_nav_graph();

            /*
             * 构建前缀树时会把向量根据标签集合划分为 group，此时的 old_vec_id 值是离散的
             * 重排序：把 group 中离散的 old_vec_id 重新整理成连续的 new_vec_id
             */

            // prepare vector storage for each group
            std::vector<IdxType> _new_to_old_vec_ids;                   // 重排序后每个新向量对应的基础数据的旧向量
            std::vector<std::pair<IdxType, IdxType>> _group_id_to_range;    // 重排序后每个 group 中 vec id 的范围，[start, end)
            std::vector<std::shared_ptr<IStorage>> _group_storages;     // group 的存储
            void prepare_group_storages_graphs();

            // graph indices for each group     每个 group 的图索引
            std::string _index_name;                                    // 索引名称，目前只支持 "Vamana"
            std::vector<std::shared_ptr<Graph>> _group_graphs;          // 图索引
            std::vector<IdxType> _group_entry_points;                   // 每个 group 的图索引的入口点
            void build_graph_for_all_groups();
            void build_complete_graph(std::shared_ptr<Graph> graph, IdxType num_points);
            std::vector<std::shared_ptr<Vamana>> _vamana_instances;     // 每个 group 的 vamana 实例

            // index parameters for each graph      每个图索引的参数
            IdxType _max_degree, _Lbuild;                               // 每个图节点的最大度数，构件图时的候选集大小
            float _alpha;                                               // 控制图的稀疏性
            uint32_t _num_threads;
            std::string _scenario;

            // cross-group edges    跨组边
            IdxType _num_cross_edges;                                   // 跨组边的数量
            std::vector<SearchQueue> _cross_group_neighbors;            // 跨组邻居
            void build_cross_group_edges();

            // obtain the final unified navigating graph
            void add_offset_for_uni_nav_graph();

            // obtain entry_points
            std::vector<IdxType> get_entry_points(const std::vector<LabelType>& query_label_set, 
                                                  IdxType num_entry_points, VisitedSet& visited_set);
            void get_entry_points_given_group_id(IdxType num_entry_points, VisitedSet& visited_set, 
                                                 IdxType group_id, std::vector<IdxType>& entry_points);

            // search in graph

            IdxType iterate_to_fixed_point(const char* query, std::shared_ptr<SearchCache> search_cache, 
                                           IdxType target_id, const std::vector<IdxType>& entry_points,
                                           bool clear_search_queue=true, bool clear_visited_set=true);

            // statistics   统计信息
            float _index_time, _label_processing_time, _build_graph_time;
            float _build_LNG_time = 0, _build_cross_edges_time = 0, _index_size;
            IdxType _graph_num_edges, _LNG_num_edges;
            void statistics();
    };
}

#endif // UNG_H