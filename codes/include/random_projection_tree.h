#ifndef RANDOM_PROJECTION_TREE_H
#define RANDOM_PROJECTION_TREE_H

#include <omp.h>
#include "storage.h"
#include "kmeans.h"
#include "search_queue.h"
#include "distance.h"
#include "third_party/annoy/annoylib.h"
#include "third_party/annoy/kissrandom.h"
#include <future>

namespace ANNS {

    struct RPTreeNode {
        float* randomDirection;    // 随机投影方向
        float medianProj;      // 节点中数据点在随机投影方向上的中位数
        std::shared_ptr<RPTreeNode> left;
        std::shared_ptr<RPTreeNode> right;

        IdxType depth;
        IdxType group_id;   // 叶子节点由从 1 递增的 id 标识，其他节点的 id 是 0
        IdxType group_size; // 节点中向量数量

        RPTreeNode() : randomDirection(nullptr), medianProj(0), left(nullptr), right(nullptr), depth(0), group_id(0), group_size(0) {};

        ~RPTreeNode() {
            delete [] randomDirection;
        };
    };

    class RPTree {
    public:
        RPTree() = delete;
        RPTree(const std::shared_ptr<IStorage>& base_storage, const std::shared_ptr<DistanceHandler>& distance_handler, IdxType max_node_size);
        ~RPTree();
        void build(IdxType num_threads = 1);
        void search(std::shared_ptr<IStorage> query_storage, IdxType K, std::pair<IdxType, float>* results, std::vector<IdxType>& num_cmps);

        void sampling(double rate);
        void sample_slice(const std::vector<IdxType>& points, std::vector<IdxType> &sampled_data, double rate);

        IdxType kmean_cluster(IdxType num_centers, IdxType max_reps);

        std::shared_ptr<RPTreeNode> get_root() { return _root; }
        void traverse_tree(const std::shared_ptr<RPTreeNode>& node);

    private:
        std::shared_ptr<IStorage> _base_storage, _query_storage;
        std::shared_ptr<DistanceHandler> _distance_handler;
        IdxType _max_node_size;
        IdxType _num_groups;
        IdxType _num_threads;
        IdxType _num_points;
        IdxType _num_queries;
        IdxType _dim;

        std::mutex _mutex;
        std::shared_ptr<RPTreeNode> _root;
        std::vector<std::vector<IdxType>> _points_in_node;  // 每个叶子节点中的原始向量 id（向量的全局索引）  group 中的一组向量，是重排序前的基础数据    group 索引从 1 开始
        std::vector<IdxType> _vec_id_to_group_id;  // 每个向量所属的叶子节点
        std::vector<std::shared_ptr<RPTreeNode>> _leaf_nodes;   // 所有叶子节点，用 group id 索引

        IdxType _num_samples;
        float* _sample_vecs;
        float* _pivot_data;
        std::vector<IdxType> _new_to_old_vec_ids;   // 采样数据的连续id对应原始数据id
        std::vector<std::vector<IdxType>> _closest_docs;    // 每个簇中的数据点
        std::vector<IdxType> _closest_center;   // 每个点所属的聚类中心

        std::shared_ptr<RPTreeNode> build_tree(const std::vector<IdxType>& vecs, IdxType depth, IdxType& new_group_id, IdxType num_threads = 1);
        IdxType search_tree(const std::shared_ptr<RPTreeNode>& node, SearchQueue& search_queue, IdxType query_vec, IdxType K, IdxType& num_cmps);
        void generate_random_direction(IdxType dim, float* dir);
        float project(const char* a, const float* dir, IdxType dim);
    };
}


#endif //RANDOM_PROJECTION_TREE_H
