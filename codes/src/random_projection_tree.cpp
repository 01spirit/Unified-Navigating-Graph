#include "random_projection_tree.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <numeric>
#include <boost/range/end.hpp>
#include <utility>

#include "search_queue.h"
#include "utils.h"

namespace ANNS {
    RPTree::RPTree(const std::shared_ptr<IStorage>& base_storage, const std::shared_ptr<DistanceHandler>& distance_handler) {
        _base_storage = base_storage;
        _distance_handler = distance_handler;
        _num_points = base_storage->get_num_points();
        _dim = base_storage->get_dim();
        _num_queries = 0;

        srand(time(0));
        _root = std::make_shared<RPTreeNode>();
    }

    void RPTree::build() {
        std::cout<<SEP_LINE;
        std::cout<<"Building RPTree..."<<std::endl;
        std::vector<IdxType> vecs(_num_points);
        std::iota(vecs.begin(), vecs.end(), 0);
        IdxType new_group_id = 1;
        _root = build_tree(vecs, 0, new_group_id);
        std::cout<<"Building RPTree done"<<std::endl;
        std::cout<<SEP_LINE;
    }

    std::vector<float> RPTree::generate_random_direction(IdxType dim) {
        std::vector<float> dir(dim);
        float norm = 0;
        for (IdxType i = 0; i < dim; i++) {
            dir[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2 - 1;
            norm += dir[i] * dir[i];
        }
        norm = std::sqrt(norm);
        for (IdxType i = 0; i < dim; i++) {
            dir[i] /= norm;
        }
        return dir;
    }

    float RPTree::project(const char* a, const std::vector<float>& dir) {
        const float* vec = reinterpret_cast<const float*>(a);
        float proj = 0;
        for (IdxType i = 0; i < _dim; ++i) {
            proj += vec[i] * dir[i];
        }
        return proj;
    }

    std::shared_ptr<RPTreeNode> RPTree::build_tree(const std::vector<IdxType>& vecs, IdxType depth, IdxType& new_group_id) {
        // std::cout<<"BUILD: Depth: "<<depth<<" group id: "<<new_group_id<<" vecs size: "<<vecs.size()<<std::endl;
        IdxType vsize = vecs.size();
        if (vsize <= max_node_size) {
            std::shared_ptr<RPTreeNode> leaf = std::make_shared<RPTreeNode>();
            leaf->depth = depth;
            leaf->group_id = new_group_id;
            leaf->group_size = vsize;
            _points.resize(new_group_id+1);
            for (IdxType i = 0; i < vsize; i++) {
                _points[new_group_id].emplace_back(i);
            }
            new_group_id++;

            return leaf;
        }

        std::vector<float> dir = generate_random_direction(_dim);
        auto projections = std::vector<std::pair<IdxType, float>>(vsize);
        for (IdxType i = 0; i < vsize; i++) {
            float p = project(_base_storage->get_vector(vecs[i]), dir);
            projections[i] = std::make_pair(vecs[i], p);
        }

        std::sort(projections.begin(), projections.end(),
            [](const std::pair<IdxType, float>& a, const std::pair<IdxType, float>& b) {
                return a.second < b.second;
            });

        int mid = projections.size() / 2;
        float median = projections[mid].second;

        std::vector<IdxType> left_vecs, right_vecs;
        left_vecs.reserve(vsize / 2 + 1);
        right_vecs.reserve(vsize / 2 + 1);
        for (auto& pair : projections) {
            if (pair.second <= median) {
                left_vecs.emplace_back(pair.first);
            } else {
                right_vecs.emplace_back(pair.first);
            }
        }

        std::shared_ptr<RPTreeNode> node = std::make_shared<RPTreeNode>();
        node->randomDirection = dir;
        node->medianProj = median;
        node->depth = depth;
        node->group_id = 0;
        node->group_size = vecs.size();
        node->left = build_tree(left_vecs, depth + 1, new_group_id);
        node->right = build_tree(right_vecs, depth + 1, new_group_id);

        return node;
    }

    void RPTree::search(std::shared_ptr<IStorage> query_storage, IdxType K, std::pair<IdxType, float>* results, std::vector<IdxType>& num_cmps) {
        _query_storage = std::move(query_storage);
        _num_queries = _query_storage->get_num_points();
        std::cout<<SEP_LINE;
        std::cout<<"Start searching "<<_query_storage->get_num_points()<<" query vectors"<<std::endl;

        SearchQueue cur_result;
        cur_result.reserve(K);
        for (auto id = 0; id < _num_queries; ++id) {
            if (id % _num_queries == _num_queries / 10) {
                std::cout<<"SEARCH: query vec "<<id<<std::endl;
            }
            search_tree(_root, cur_result, id, K, num_cmps[id]);

            for (IdxType k = 0; k < K; ++k) {
                if (k < cur_result.size()) {
                    results[id*K+k].first = cur_result[k].id;
                    results[id*K+k].second = cur_result[k].distance;
                } else {
                    results[id*K+k].first = -1;
                }
            }
        }
        std::cout<<"Search done"<<std::endl;
        std::cout<<SEP_LINE;
    }

    IdxType RPTree::search_tree(const std::shared_ptr<RPTreeNode>& node, SearchQueue& search_queue, IdxType query_vec, IdxType K, IdxType& num_cmps) {
        if (!node) {
            return 0;
        }

        if (!node->left && !node->right) {
            // std::cout<<"SEARCH: Depth: "<<node->depth<<" group id: "<<node->group_id<<" vecs size: "<<node->group_size<<std::endl;
            for (auto& pt : _points[node->group_id]) {
                float dist = _distance_handler->compute(_base_storage->get_vector(pt), _query_storage->get_vector(query_vec), _dim);
                search_queue.insert(pt, dist);
            }
            return node->depth;
        }

        float proj = project(_query_storage->get_vector(query_vec), node->randomDirection);
        auto near = (proj <= node->medianProj) ? node->left : node->right;
        auto far = (proj <= node->medianProj) ? node->right : node->left;

        num_cmps += search_tree(near, search_queue, query_vec, K, num_cmps);
        num_cmps += search_tree(far, search_queue, query_vec, K, num_cmps);

        return num_cmps;
    }

    void RPTree::traverse_tree(const std::shared_ptr<RPTreeNode>& node) {
        if (!node) {
            return;
        }
        std::cout<<"Depth: "<<node->depth<<" group id: "<<node->group_id<<" vecs size: "<<node->group_size<<std::endl;
        traverse_tree(node->left);
        traverse_tree(node->right);
    }


}
