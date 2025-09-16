#include "random_projection_tree.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <numeric>
#include <boost/range/end.hpp>
#include <utility>

#include "kmeans.h"
#include "search_queue.h"
#include "utils.h"

namespace ANNS {
    RPTree::RPTree(const std::shared_ptr<IStorage>& base_storage, const std::shared_ptr<DistanceHandler>& distance_handler, IdxType max_node_size) {
        _base_storage = base_storage;
        _distance_handler = distance_handler;
        _max_node_size = max_node_size;
        _num_threads = 1;
        _num_points = base_storage->get_num_points();
        _dim = base_storage->get_dim();
        _num_queries = 0;

        srand(time(0));
        _root = std::make_shared<RPTreeNode>();
    }

    void RPTree::build(IdxType num_threads) {
        std::cout<<SEP_LINE;
        std::cout<<"Building RPTree..."<<std::endl;
        std::vector<IdxType> vecs(_num_points);
        std::iota(vecs.begin(), vecs.end(), 0);
        IdxType new_group_id = 1;
        _root = build_tree(vecs, 0, new_group_id, num_threads);
        std::cout<<"Building RPTree done"<<std::endl;
        std::cout<<SEP_LINE;
    }

    void RPTree::generate_random_direction(IdxType dim, float* dir) {
        // std::vector<float> dir(dim);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0, RAND_MAX);

        float norm = 0.0f;

        __m256 norm_vec = _mm256_setzero_ps();
        for (IdxType i = 0; i < dim; i += 8) {
            // float x = dis(gen) / RAND_MAX * 2.0f - 1.0f;
            // __m256 rand_vec = _mm256_set1_ps(x);
            __m256 rand_vec = _mm256_set_ps(
                dis(gen) / RAND_MAX * 2 - 1,
                dis(gen) / RAND_MAX * 2 - 1,
                dis(gen) / RAND_MAX * 2 - 1,
                dis(gen) / RAND_MAX * 2 - 1,
                 dis(gen) / RAND_MAX * 2 - 1,
                 dis(gen) / RAND_MAX * 2 - 1,
                dis(gen) / RAND_MAX * 2 - 1,
                dis(gen) / RAND_MAX * 2 - 1
            );

            _mm256_storeu_ps(&dir[i], rand_vec);
            __m256 squared_vec = _mm256_mul_ps(rand_vec, rand_vec);
            norm_vec = _mm256_add_ps(norm_vec, squared_vec);
        }
        norm_vec = _mm256_hadd_ps(norm_vec, norm_vec);
        norm_vec = _mm256_hadd_ps(norm_vec, norm_vec);
        float norm_array[8];
        _mm256_storeu_ps(norm_array, norm_vec);
        norm = norm_array[0] + norm_array[4];
        norm = std::sqrt(norm);

        __m256 norm_scalar = _mm256_set1_ps(norm);
        for (IdxType i = 0; i < dim; i += 8) {
            __m256 dir_vec = _mm256_loadu_ps(&dir[i]);
            dir_vec = _mm256_div_ps(dir_vec, norm_scalar);
            _mm256_storeu_ps(&dir[i], dir_vec);
        }

        // for (IdxType i = 0; i < dim; i++) {
        //     dir[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2 - 1;
        //     norm += dir[i] * dir[i];
        // }
        // norm = std::sqrt(norm);
        // for (IdxType i = 0; i < dim; ++i) {
        //     dir[i] /= norm;
        // }

    }

    float RPTree::project(const char* a, const float* dir, IdxType dim) {
        const float* vec = reinterpret_cast<const float*>(a);

        // float proj = 0;
        // for (IdxType i = 0; i < _dim; ++i) {
        //     proj += vec[i] * dir[i];
        // }
        // return proj;

        __m256 msum0 = _mm256_setzero_ps();
        while (dim >=8) {
            __m256 mx = _mm256_loadu_ps(vec);
            vec += 8;
            __m256 md = _mm256_loadu_ps(dir);
            dir += 8;
            msum0 = _mm256_add_ps(msum0, _mm256_mul_ps(mx, md));
            dim -= 8;
        }
        __m128 msum1 = _mm256_extractf128_ps(msum0, 1);
        __m128 msum2 = _mm256_extractf128_ps(msum0, 0);
        msum1 = _mm_add_ps(msum1, msum2);

        if (dim >= 4) {
            __m128 mx = _mm_loadu_ps(vec);
            vec += 4;
            __m128 md = _mm_load_ps(dir);
            dir += 4;
            msum1 = _mm_add_ps(msum1, _mm_mul_ps(mx, md));
            dim -= 4;
        }

        if (dim >0) {
            __m128 mx = masked_read(dim, vec);
            __m128 md = masked_read(dim, dir);
            msum1 = _mm_add_ps(msum1, _mm_mul_ps(mx, md));
        }

        msum1 = _mm_hadd_ps(msum1, msum1);
        msum1 = _mm_hadd_ps(msum1, msum1);

        return _mm_cvtss_f32(msum1);
    }

    std::shared_ptr<RPTreeNode> RPTree::build_tree(const std::vector<IdxType>& vecs, IdxType depth, IdxType& new_group_id, IdxType num_threads) {
        // std::cout<<"BUILD: Depth: "<<depth<<" group id: "<<new_group_id<<" vecs size: "<<vecs.size()<<std::endl;
        _num_threads = std::min(num_threads, std::thread::hardware_concurrency());
        omp_set_num_threads(_num_threads);

        IdxType vsize = vecs.size();
        if (vsize <= _max_node_size) {
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

        float* dir = new float[_dim];
        generate_random_direction(_dim, dir);
        auto projections = std::vector<std::pair<IdxType, float>>(vsize);
        #pragma omp parallel for schedule(dynamic, 1024)
        for (IdxType i = 0; i < vsize; i++) {
            float p = project(_base_storage->get_vector(vecs[i]), dir, _dim);
            projections[i] = std::make_pair(vecs[i], p);
        }
        delete[] dir;

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
        // node->randomDirection = dir;
        node->medianProj = median;
        node->depth = depth;
        node->group_id = 0;
        node->group_size = vecs.size();
        node->left = build_tree(left_vecs, depth + 1, new_group_id, _num_threads);
        node->right = build_tree(right_vecs, depth + 1, new_group_id, _num_threads);

        // std::future<std::shared_ptr<RPTreeNode>> left_future = std::async(std::launch::async, &RPTree::build_tree, this, left_vecs, depth + 1, std::ref(new_group_id), _num_threads);
        // std::future<std::shared_ptr<RPTreeNode>> right_future = std::async(std::launch::async, &RPTree::build_tree, this, right_vecs, depth + 1, std::ref(new_group_id), _num_threads);
        //
        // node->left = left_future.get();
        // node->right = right_future.get();

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

        float proj = project(_query_storage->get_vector(query_vec), node->randomDirection, _dim);
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
