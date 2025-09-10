#include <gtest/gtest.h>
#include <numeric>
#include <boost/mpl/distance.hpp>
#include <boost/range/distance.hpp>
#include <statgrab.h>
#include <atomic>

#include "random_projection_tree.h"
#include "graph.h"
#include "utils.h"
#include "vamana/vamana.h"

#define ANNOYLIB_MULTITHREADED_BUILD

namespace ANNS {
    class RPTreeTest : public ::testing::Test {

    };

    // 全局变量，用于控制后台线程的运行
    std::atomic<bool> running(true);
    std::mutex mtx;
    std::condition_variable cv;

    // 后台线程函数，用于实时监控CPU占用
    void monitor_cpu_usage() {
        sg_init(0); // 初始化libstatgrab
        sg_cpu_percents *cpu = nullptr;
        while (running) {
            cpu = sg_get_cpu_percents(nullptr); // 获取CPU使用率
            if (cpu != nullptr) {
                std::lock_guard<std::mutex> lock(mtx);
                std::cout << "User: " << cpu->user << "%" << "\t";
                std::cout << "System: " << cpu->kernel << "%" << "\t";
                std::cout << "Idle: " << cpu->idle << "%" << std::endl;
                // sg_free_cpu_percents(cpu); // 释放资源
            } else {
                std::cerr << "Failed to get CPU stats" << std::endl;
            }

            // 等待一段时间后再次获取CPU使用率
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        sg_shutdown(); // 关闭libstatgrab
    }

    TEST_F(RPTreeTest, TEST1) {

        // 创建并启动后台线程来监控CPU使用率
        std::thread cpu_monitor_thread(monitor_cpu_usage);

        std::string data_type = "float";
        std::string dist_fn = "L2";
        std::string base_bin_file = "../../data/sift/sift_base.bin";
        std::string base_label_file = "../../data/sift/sift_base_12_labels_zipf.txt";

        // load base data
        std::shared_ptr<ANNS::IStorage> base_storage = ANNS::create_storage(data_type);
        base_storage->load_from_file(base_bin_file, base_label_file);

        // preparation
        auto start_time = std::chrono::high_resolution_clock::now();
        std::shared_ptr<ANNS::DistanceHandler> distance_handler = ANNS::get_distance_handler(data_type, dist_fn);

        // build vamana index
        // std::shared_ptr<ANNS::Graph> graph = std::make_shared<ANNS::Graph>(base_storage->get_num_points());
        // ANNS::Vamana index;
        // index.build(base_storage, distance_handler, graph, 32, 100, 1.2, 32);
        // std::cout << "Index time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << "ms" << std::endl;

        start_time = std::chrono::high_resolution_clock::now();
        std::shared_ptr<RPTree> rp_tree = std::make_shared<RPTree>(base_storage, distance_handler);
        rp_tree->build();
        std::cout << "Index time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << "ms" << std::endl;

        // rp_tree->traverse_tree(rp_tree->get_root());

        // 通知后台线程停止运行
        {
            std::lock_guard<std::mutex> lock(mtx);
            running = false;
        }
        cv.notify_all();

        // 等待后台线程结束
        cpu_monitor_thread.join();
    }

    TEST_F(RPTreeTest, TEST2) {
        std::string data_type = "float";
        std::string dist_fn = "L2";
        std::string base_bin_file = "../../data/sift/sift_base.bin";
        std::string base_label_file = "../../data/sift/sift_base_12_labels_zipf.txt";
        std::string query_bin_file = "../../data/sift/sift_query.bin";
        std::string query_label_file = "../../data/sift/sift_query_12_labels_zipf_containment.txt";

        // load base data
        std::shared_ptr<ANNS::IStorage> base_storage = ANNS::create_storage(data_type);
        std::shared_ptr<ANNS::IStorage> query_storage = ANNS::create_storage(data_type);
        base_storage->load_from_file(base_bin_file, base_label_file);
        query_storage->load_from_file(query_bin_file, query_label_file);

        std::shared_ptr<IStorage> sub_query_storage = ANNS::create_storage(query_storage, 0, 10);

        // preparation
        auto start_time = std::chrono::high_resolution_clock::now();
        std::shared_ptr<ANNS::DistanceHandler> distance_handler = ANNS::get_distance_handler(data_type, dist_fn);

        start_time = std::chrono::high_resolution_clock::now();
        std::shared_ptr<RPTree> rp_tree = std::make_shared<RPTree>(base_storage, distance_handler);
        rp_tree->build();
        std::cout << "Index time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << "ms" << std::endl;


        IdxType K = 10;
        std::string gt_file = "../../data/sift/sift_gt_12_labels_zipf_containment.bin";

        // preparation
        auto num_queries = sub_query_storage->get_num_points();
        auto gt = new std::pair<ANNS::IdxType, float>[num_queries * K];
        ANNS::load_gt_file(gt_file, gt, num_queries, K);

        auto results = new std::pair<ANNS::IdxType, float>[num_queries * K];
        std::vector<ANNS::IdxType> num_cmps(num_queries);

        // search
        start_time = std::chrono::high_resolution_clock::now();
        rp_tree->search(sub_query_storage, K, results, num_cmps);
        auto time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count();

        // statistics
        std::cout << "- Time cost: " << time_cost << "ms" << std::endl;
        std::cout << "- QPS: " << num_queries * 1000.0 / time_cost << std::endl;
        float total_cmps = std::accumulate(num_cmps.begin(), num_cmps.end(), 0);
        std::cout << "- Average number of comparisons: " << total_cmps / num_queries << std::endl;

        // calculate recall
        auto recall = ANNS::calculate_recall(gt, results, num_queries, K);
        std::cout << "- Recall: " << recall << "%" << std::endl;

    }

    TEST_F(RPTreeTest, TEST3) {
        std::string data_type = "float";
        std::string dist_fn = "L2";
        std::string base_bin_file = "../../data/sift/sift_base.bin";
        std::string base_label_file = "../../data/sift/sift_base_12_labels_zipf.txt";

        // load base data
        std::shared_ptr<ANNS::IStorage> base_storage = ANNS::create_storage(data_type);
        base_storage->load_from_file(base_bin_file, base_label_file);

        // preparation
        auto start_time = std::chrono::high_resolution_clock::now();
        std::shared_ptr<ANNS::DistanceHandler> distance_handler = ANNS::get_distance_handler(data_type, dist_fn);

        IdxType dim = 128;
        Annoy::AnnoyIndex<IdxType, float, Annoy::Euclidean, Annoy::Kiss32Random, Annoy::AnnoyIndexMultiThreadedBuildPolicy> annoy_index =
            Annoy::AnnoyIndex<IdxType ,float, Annoy::Euclidean, Annoy::Kiss32Random, Annoy::AnnoyIndexMultiThreadedBuildPolicy>(base_storage->get_dim());
        // omp_set_num_threads(32);
        // #pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < base_storage->get_num_points(); ++i) {
            const float* vec = reinterpret_cast<const float*> (base_storage->get_vector(i));
            annoy_index.add_item(i, vec);
        }
        annoy_index.build(1);

        std::cout << "Index time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << "ms" << std::endl;

        std::cout<<"items num: "<<annoy_index.get_n_items()<<" tree num: "<<annoy_index.get_n_trees()<<std::endl;

        start_time = std::chrono::high_resolution_clock::now();
        std::vector<IdxType> closest;
        std::vector<float> distance;
        annoy_index.get_nns_by_item(1, 10, 1, &closest, &distance);
        for (int i = 0; i < closest.size(); ++i) {
            std::cout<<closest[i]<<" "<<distance[i]<<std::endl;
        }
        std::cout << "Search time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << "ms" << std::endl;

    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}