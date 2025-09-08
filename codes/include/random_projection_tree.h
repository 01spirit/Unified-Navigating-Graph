#ifndef RANDOM_PROJECTION_TREE_H
#define RANDOM_PROJECTION_TREE_H

#include "storage.h"
#include "search_queue.h"
#include "distance.h"

namespace ANNS {

    inline IdxType max_node_size = 100;

    struct RPTreeNode {
        std::vector<IdxType> points;    // 节点内向量的全局索引
        std::vector<float> randomDirection;    // 随机投影方向
        float medianProj;      // 节点中数据点在随机投影方向上的中位数
        std::shared_ptr<RPTreeNode> left;
        std::shared_ptr<RPTreeNode> right;

        IdxType depth;
        IdxType group_id;
        IdxType group_size;

        RPTreeNode() : medianProj(0), left(nullptr), right(nullptr), depth(0), group_id(0), group_size(0) {};
        ~RPTreeNode() = default;
    };

    class RPTree {
    public:
        RPTree() = delete;
        RPTree(const std::shared_ptr<IStorage>& base_storage, const std::shared_ptr<DistanceHandler>& distance_handler);
        ~RPTree() = default;
        void build();
        void search(std::shared_ptr<IStorage> query_storage, IdxType K, std::pair<IdxType, float>* results, std::vector<IdxType>& num_cmps);

        std::shared_ptr<RPTreeNode> get_root() { return _root; }
        void traverse_tree(const std::shared_ptr<RPTreeNode>& node);

    private:
        std::shared_ptr<IStorage> _base_storage, _query_storage;
        std::shared_ptr<DistanceHandler> _distance_handler;
        IdxType _num_points;
        IdxType _num_queries;
        IdxType _dim;

        std::shared_ptr<RPTreeNode> _root;

        std::shared_ptr<RPTreeNode> build_tree(const std::vector<IdxType>& vecs, IdxType depth, IdxType& new_group_id);
        IdxType search_tree(const std::shared_ptr<RPTreeNode>& node, SearchQueue& search_queue, IdxType query_vec, IdxType K, IdxType& num_cmps);
        std::vector<float> generate_random_direction(IdxType dim);
        float project(const char* a, const std::vector<float>& dir);
    };
}


#endif //RANDOM_PROJECTION_TREE_H
