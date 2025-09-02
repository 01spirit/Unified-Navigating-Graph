#include <omp.h>
#include <iostream>
#include <algorithm>
#include <boost/filesystem.hpp>
#include "utils.h"
#include "vamana.h"

namespace fs = boost::filesystem;



namespace ANNS {

    // 构建 vamana 索引
    void Vamana::build(std::shared_ptr<IStorage> base_storage, std::shared_ptr<DistanceHandler> distance_handler, 
                       std::shared_ptr<Graph> graph, IdxType max_degree, IdxType Lbuild, float alpha, 
                       uint32_t num_threads, IdxType max_candidate_size) {
        
        if (_verbose) {
            std::cout << "Building Vamana index ..." << std::endl;
            std::cout << "- max_degree: " << max_degree << std::endl;
            std::cout << "- Lbuild: " << Lbuild << std::endl;
            std::cout << "- alpha: " << alpha << std::endl;
            std::cout << "- max_candidate_size: " << max_candidate_size << std::endl;
            std::cout << "- num_threads: " << num_threads << std::endl;
        }
        
        _base_storage = base_storage;
        _distance_handler = distance_handler;
        _graph = graph;

        _num_threads = num_threads;
        _max_degree = max_degree;
        _Lbuild = Lbuild;
        _alpha = alpha;
        _max_candidate_size = max_candidate_size;

        if (_verbose)
            std::cout << "Computing entry point ..." << std::endl;
        _entry_point = _base_storage->choose_medoid(num_threads, distance_handler);     // 找出最接近几何中心的向量，作为图索引的入口

        if (_verbose)
            std::cout << "Linking the graph ..." << std::endl;
        link();     // 构造图索引

        if (_verbose)
            std::cout << "Finish." << std::endl << SEP_LINE;
    }


    // 为向量构造图索引，连接每个数据点的邻居
    void Vamana::link() {
        auto num_points = _base_storage->get_num_points();
        auto dim = _base_storage->get_dim();
        SearchCacheList search_cache_list(_num_threads, num_points, _Lbuild);

        // 并行处理数据点，找到其在图索引中的邻居，构建边的关系
        omp_set_num_threads(_num_threads);
        #pragma omp parallel for schedule(dynamic, 1)
        for (auto id = 0; id < num_points; ++id) {
            auto search_cache = search_cache_list.get_free_cache();     // 从缓存池中获取对象

            // search for point 
            const char* query = _base_storage->get_vector(id);
            iterate_to_fixed_point(query, search_cache, true, id);  // 找到该向量的固定点，记录过程中扩展的节点

            // prune for candidate neighbors
            std::vector<IdxType> pruned_list;
            prune_neighbors(id, search_cache->expanded_list, pruned_list, search_cache);    // 对扩展的节点剪枝，找到该向量的在度数限制内的邻居节点 pruned_list

            // update neighbors and insert the reversed edge
            {
                std::lock_guard<std::mutex> lock(_graph->neighbor_locks[id]);   // 作用域锁，出了大括号自动销毁
                _graph->neighbors[id] = pruned_list;    // 在图索引中记录该向量的邻居节点
            }
            inter_insert(id, pruned_list, search_cache);    // 插入反向边，是从各个邻居到该向量的，若过程中某个邻居的度数超限，会对其剪枝

            // clean and print
            search_cache_list.release_cache(search_cache);  // 放回缓存池
            if (_verbose && id % 10000 == 0)
                std::cout << "\r" << (100.0 * id) / num_points << "%" << std::flush;
        }

        if (_verbose)
            std::cout << "\rStarting final cleanup ..." << std::endl;
        // 并行处理数据点，检查每个点的邻居数量是否超出最大度数限制
        #pragma omp parallel for schedule(dynamic, 1)
        for (auto id = 0; id < num_points; ++id)
            if (_graph->neighbors[id].size() > _max_degree) {

                // prepare candidates
                std::vector<Candidate> candidates;  // 计算距离后加入候选集
                for (auto& neighbor : _graph->neighbors[id]) 
                    candidates.emplace_back(neighbor, _distance_handler->compute(_base_storage->get_vector(id), 
                                                                                 _base_storage->get_vector(neighbor), dim));
                
                // prune neighbors
                std::vector<IdxType> new_neighbors;
                auto search_cache = search_cache_list.get_free_cache();
                prune_neighbors(id, candidates, new_neighbors, search_cache);   // 剪枝
                _graph->neighbors[id] = new_neighbors;
            }
    }


    // 从图的入口点开始扩展到最近的节点，直到找到一个固定点，固定点是在当前查询向量下所能找到的最接近的点，把搜索过程中的所有近邻点按距离顺序存入 search_cache->search_queue
    // 返回比较次数    query : 查询向量， target_id : 目标节点 id，默认值 -1 表示没用到
    IdxType Vamana::iterate_to_fixed_point(const char* query, std::shared_ptr<SearchCache> search_cache, 
                                           bool record_expanded, IdxType target_id) {
        auto dim = _base_storage->get_dim();
        auto& search_queue = search_cache->search_queue;
        auto& visited_set = search_cache->visited_set;
        auto& expanded_list = search_cache->expanded_list;
        search_queue.clear();
        visited_set.clear();
        expanded_list.clear();
        std::vector<IdxType> neighbors;
        
        // entry point      把入口点插入搜索队列
        search_queue.insert(_entry_point, _distance_handler->compute(query, _base_storage->get_vector(_entry_point), dim));
        IdxType num_cmps = 1;

        // greedily expand closest nodes    贪心扩展到最近节点
        while (search_queue.has_unexpanded_node()) {    // 从队列中找到可扩展节点
            const Candidate& cur = search_queue.get_closest_unexpanded();   // 最近的可扩展节点
            if (record_expanded && target_id != cur.id)     // 记录扩展节点
                expanded_list.push_back(cur);

            // iterate neighbors
            {
                std::lock_guard<std::mutex> lock(_graph->neighbor_locks[cur.id]);
                neighbors = _graph->neighbors[cur.id];
            }
            for (auto i=0; i<neighbors.size(); ++i) {

                // prefetch     预取下一个邻居节点的数据到 L1 cache 中
                if (i+1 < neighbors.size()) {
                    visited_set.prefetch(neighbors[i+1]);
                    _base_storage->prefetch_vec_by_id(neighbors[i+1]);
                }

                // skip if visited
                auto& neighbor = neighbors[i];
                if (visited_set.check(neighbor))    // 跳过已访问节点
                    continue;
                visited_set.set(neighbor);  // 保存访问状态

                // push to search queue     插入搜索队列，用于进一步扩展；若距离太远，可能实际上并未插入
                search_queue.insert(neighbor, _distance_handler->compute(query, _base_storage->get_vector(neighbor), dim));
                num_cmps++;
            }
        }
        return num_cmps;
    }



    // 剪枝邻居，使每个节点（向量）的邻居数量不超过最大度数 _max_degree；  id : 当前节点的 id，candidates : 候选向量集合， pruned_list : 剪枝列表，保留最相关的邻居
    void Vamana::prune_neighbors(IdxType id, std::vector<Candidate>& candidates, std::vector<IdxType>& pruned_list, 
                                    std::shared_ptr<SearchCache> search_cache) {
        auto dim = _base_storage->get_dim();
        pruned_list.clear();
        pruned_list.reserve(_max_degree);

        // init candidates      按照距离排序候选向量
        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            return a.distance < b.distance;
        });
        auto candidate_size = std::min((IdxType)(candidates.size()), _max_candidate_size);

        // init occlude factor  初始化邻居的遮挡因子
        auto& occlude_factor = search_cache->occlude_factor;
        occlude_factor.clear();
        occlude_factor.insert(occlude_factor.end(), candidate_size, 0.0f);

        // prune neighbors      剪枝
        float cur_alpha = 1;    // alpha 越大，图越稀疏，剪枝过程中会增大，用于控制剪枝的严格程度
        while (cur_alpha <= _alpha && pruned_list.size() < _max_degree) {
            for (auto i=0; i<candidate_size && pruned_list.size() < _max_degree; ++i) {
                if (occlude_factor[i] > cur_alpha)  // 跳过遮挡因子更大的邻居
                    continue;

                // set to float::max so that is not considered again
                occlude_factor[i] = std::numeric_limits<float>::max();  // 不再添加该邻居到剪枝列表中
                if (candidates[i].id != id)     // 将该邻居节点添加到剪枝列表中
                    pruned_list.push_back(candidates[i].id);

                // update occlude factor for the following candidates   更新后续候选向量的遮挡因子
                for (auto j=i+1; j<candidate_size; ++j) {
                    if (occlude_factor[j] > _alpha)
                        continue;
                    auto distance_ij = _distance_handler->compute(_base_storage->get_vector(candidates[i].id), 
                                                                _base_storage->get_vector(candidates[j].id), dim);      // 计算后续节点与当前节点的距离，更新遮挡因子
                    occlude_factor[j] = (distance_ij == 0) ? std::numeric_limits<float>::max() 
                                                        : std::max(occlude_factor[j], candidates[j].distance / distance_ij);
                }
            }
            cur_alpha *= 1.2f;
        }
    }


    // 在图中插入反向边，确保图的对称性，两个节点应互为邻居，必要时对邻居剪枝； src : 当前节点的 id，src_neighbors : 当前节点的邻居
    void Vamana::inter_insert(IdxType src, std::vector<IdxType>& src_neighbors, std::shared_ptr<SearchCache> search_cache) {
        auto dim = _base_storage->get_dim();

        // insert the reversed edge     插入反向边
        for (auto& dst : src_neighbors) {
            bool need_prune = false;
            std::vector<Candidate> candidates;

            // try to add edge dst -> src   添加从邻居到源节点的反向边
            {
                std::lock_guard<std::mutex> lock(_graph->neighbor_locks[dst]);
                auto& dst_neighbors = _graph->neighbors[dst];       // 该邻居的所有邻居节点
                if (std::find(dst_neighbors.begin(), dst_neighbors.end(), src) == dst_neighbors.end()) {    // 若其中不包含源节点，需要添加一条边
                    if (dst_neighbors.size() < (IdxType)(default_paras::GRAPH_SLACK_FACTOR * _max_degree)) 
                        dst_neighbors.push_back(src);
                    else {
                        candidates.reserve(dst_neighbors.size() + 1);   // 度数超限，需要剪枝，距离先初始化为 0
                        for (auto& neighbor : dst_neighbors)     // 把该邻居的所有邻居和待连接的源节点都放入候选集
                            candidates.emplace_back(neighbor, 0);
                        candidates.emplace_back(src, 0);
                        need_prune = true;
                    }
                }
            }

            // prune the neighbors of dst   对邻居进行剪枝
            if (need_prune) {
                for (auto& candidate : candidates)
                    candidate.distance = _distance_handler->compute(_base_storage->get_vector(dst), 
                                                                    _base_storage->get_vector(candidate.id), dim);
                std::vector<IdxType> new_dst_neighbors;
                prune_neighbors(dst, candidates, new_dst_neighbors, search_cache);  // 剪枝后最相关的邻居
                {
                    std::lock_guard<std::mutex> lock(_graph->neighbor_locks[dst]);
                    _graph->neighbors[dst] = new_dst_neighbors;     // 直接更新全部
                }
            }
        }
    }



    void Vamana::statistics() {
        float num_points = _base_storage->get_num_points();
        std::cout << "Number of points: " << num_points << std::endl;

        float num_edges = 0;
        IdxType min_degree = std::numeric_limits<IdxType>::max(), max_degree = 0;
        for (auto id=0; id<num_points; ++id) {
            num_edges += _graph->neighbors[id].size();
            min_degree = std::min(min_degree, (IdxType)_graph->neighbors[id].size());
            max_degree = std::max(max_degree, (IdxType)_graph->neighbors[id].size());
        }
        std::cout << "Number of edges: " << num_edges << std::endl;
        std::cout << "Min degree: " << min_degree << std::endl;
        std::cout << "Max degree: " << max_degree << std::endl;

        float avg_degree = num_edges / num_points;
        std::cout << "Average degree: " << avg_degree << std::endl;
    }



    void Vamana::save(std::string& index_path_prefix) {
        fs::create_directories(index_path_prefix);  // 创建存储索引文件的路径
        std::cout << "Saving index to " << index_path_prefix << " ..." << std::endl;

        // save meta data
        std::map<std::string, std::string> meta_data;
        meta_data["max_degree"] = std::to_string(_max_degree);
        meta_data["Lbuild"] = std::to_string(_Lbuild);
        meta_data["alpha"] = std::to_string(_alpha);
        meta_data["max_candidate_size"] = std::to_string(_max_candidate_size);
        meta_data["build_num_threads"] = std::to_string(_num_threads);
        meta_data["entry_point"] = std::to_string(_entry_point);
        std::string meta_filename = index_path_prefix + "meta";
        write_kv_file(meta_filename, meta_data);

        // save graph data      序列化图索引
        std::string graph_filename = index_path_prefix + "graph";
        _graph->save(graph_filename);

        // print
        std::cout << "- Index saved." << std::endl;
    }

    

    void Vamana::load(std::string& index_path_prefix, std::shared_ptr<Graph> graph) {
        std::cout << "Loading index from " << index_path_prefix << " ..." << std::endl;
            
        // load meta data
        std::string meta_filename = index_path_prefix + "meta";
        auto meta_data = parse_kv_file(meta_filename);
        _entry_point = std::stoi(meta_data["entry_point"]);

        // load graph data
        std::string graph_filename = index_path_prefix + "graph";
        _graph = graph;
        _graph->load(graph_filename);

        // print
        std::cout << "- Index loaded." << std::endl;
    }



    void Vamana::search(std::shared_ptr<IStorage> base_storage, std::shared_ptr<IStorage> query_storage, 
                        std::shared_ptr<DistanceHandler> distance_handler, IdxType K, IdxType Lsearch, 
                        uint32_t num_threads, std::pair<IdxType, float>* results, std::vector<IdxType>& num_cmps) {
        auto num_points = base_storage->get_num_points();   // 基础数据的向量数量
        auto num_queries = query_storage->get_num_points(); // 查询向量的数量
        _base_storage = base_storage;
        _query_storage = query_storage;
        _distance_handler = distance_handler;

        // preparation
        if (K > Lsearch) {  // 结果数量应少于候选集大小
            std::cerr << "Error: K should be less than or equal to Lsearch" << std::endl;
            exit(-1);
        }
        SearchCacheList search_cache_list(num_threads, num_points, Lsearch);    // 缓存池

        // run queries      并行执行所有查询
        omp_set_num_threads(num_threads);
        #pragma omp parallel for schedule(dynamic, 1)
        for (auto id = 0; id < num_queries; ++id) {
            auto search_cache = search_cache_list.get_free_cache(); 
            const char* query = _query_storage->get_vector(id);
            num_cmps[id] = iterate_to_fixed_point(query, search_cache);     // 寻找临近点，返回比较次数

            // write results then clean     记录所有临近点的信息，search_queue 中保存了最多 Lsearch 个距离查询点最近的点
            for (auto k=0; k<K; ++k) {
                if (k < search_cache->search_queue.size()) {
                    results[id*K+k].first = search_cache->search_queue[k].id;
                    results[id*K+k].second = search_cache->search_queue[k].distance;
                } else
                    results[id*K+k].first = -1;     // 临近点不足 K 个
            }
            search_cache_list.release_cache(search_cache);
        }
    }
}