#include "kmeans.h"
#include "vamana/vamana.h"
#include "uni_nav_graph.h"
#include <statgrab.h>
#include <thread>
#include <gtest/gtest.h>
#include <condition_variable>

namespace ANNS {
    class KmeansTest : public ::testing::Test {

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

    TEST_F(KmeansTest, TEST1) {
        int num_threads = 32; // 设置线程数为
        // 设置后续并行区域中使用的线程数
        omp_set_num_threads(num_threads);

        std::thread cpu_monitor_thread(monitor_cpu_usage);

        std::string data_type = "float";
        std::string dist_fn = "L2";
        std::string base_bin_file = "../../data/sift/sift_base.bin";
        std::string base_label_file = "../../data/sift/sift_base_12_labels_zipf.txt";

        // load base data
        std::shared_ptr<ANNS::IStorage> base_storage = ANNS::create_storage(data_type);
        base_storage->load_from_file(base_bin_file, base_label_file);

        int num_parts = 30;
        int max_k_means_reps = 100;
        IdxType num_points = base_storage->get_num_points();
        IdxType dim = base_storage->get_dim();

        char *data = base_storage->get_vector(0);
        float *vecs = reinterpret_cast<float *>(data);
        float *pivot_data = new float[num_parts * dim];
        std::vector<std::vector<IdxType>> closest_docs;
        closest_docs.resize(num_parts);
        std::vector<IdxType> closest_center;

        double rate = 1;
        float *sample_vecs = nullptr;
        IdxType num_samples = 0;
        IdxType max_num_samples = num_points * rate;
        sample_vecs = new float[max_num_samples * dim];

        auto start_time = std::chrono::high_resolution_clock::now();
        kmeans_sampling(vecs, num_points, dim, rate, sample_vecs, num_samples);
        std::cout << "Sampling time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << " ms" << std::endl;

        closest_center.resize(num_samples);
        start_time = std::chrono::high_resolution_clock::now();
        kmeanspp_selecting_pivots(sample_vecs, num_samples, dim, pivot_data, num_parts);
        auto residual = run_lloyds(sample_vecs, num_samples, dim, pivot_data, num_parts, max_k_means_reps, closest_docs, closest_center);

        std::cout << "K-means time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << " ms" << std::endl;
        std::cout<< "Residual: " << residual << std::endl;


        delete[] pivot_data;
        delete[] sample_vecs;

        {
            std::lock_guard<std::mutex> lock(mtx);
            running = false;
        }
        cv.notify_all();

        // 等待后台线程结束
        cpu_monitor_thread.join();
    }

}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}