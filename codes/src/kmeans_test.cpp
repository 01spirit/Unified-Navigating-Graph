#include "kmeans.h"
#include "vamana/vamana.h"
#include "uni_nav_graph.h"

#include <gtest/gtest.h>

namespace ANNS {
    class KmeansTest : public ::testing::Test {

    };

    TEST_F(KmeansTest, TEST1) {
        std::string data_type = "float";
        std::string dist_fn = "L2";
        std::string base_bin_file = "../../data/sift/sift_base.bin";
        std::string base_label_file = "../../data/sift/sift_base_12_labels_zipf.txt";

        // load base data
        std::shared_ptr<ANNS::IStorage> base_storage = ANNS::create_storage(data_type);
        base_storage->load_from_file(base_bin_file, base_label_file);

        int num_parts = 10;
        int max_k_means_reps = 10;
        IdxType num_points = base_storage->get_num_points();
        IdxType dim = base_storage->get_dim();

        char *data = base_storage->get_vector(0);
        float *vecs = reinterpret_cast<float *>(data);
        float *pivot_data = new float[num_parts * dim];

        auto start_time = std::chrono::high_resolution_clock::now();
        kmeanspp_selecting_pivots(vecs, num_points, dim, pivot_data, num_parts);
        auto residual = run_lloyds(vecs, num_points, dim, pivot_data, num_parts, max_k_means_reps, NULL, NULL);

        std::cout << "K-means time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << "ms" << std::endl;
        std::cout<< "Residual: " << residual << std::endl;

        delete[] pivot_data;
        delete[] vecs;
    }

}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}