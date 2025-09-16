#ifndef KMEANS_H
#define KMEANS_H

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <random>
#include <queue>
#include <sys/stat.h>
#include <vector>

#include "config.h"

namespace ANNS {
    struct PivotContainer
    {
        PivotContainer() = default;

        PivotContainer(size_t pivo_id, float pivo_dist) : piv_id{pivo_id}, piv_dist{pivo_dist}
        {
        }

        bool operator<(const PivotContainer &p) const
        {
            return p.piv_dist < piv_dist;
        }

        bool operator>(const PivotContainer &p) const
        {
            return p.piv_dist > piv_dist;
        }

        size_t piv_id;
        float piv_dist;
    };

    // 计算向量间的距离
    __m128 masked_read(IdxType dim, const float *x);
    float calc_distance(const char *x, const char *y, IdxType dim );
    float calc_distance(const float *x, const float *y, IdxType dim);

    // 计算每个向量的平方 L2 范数 （向量到原点的距离的平方）
    /*
     * @param
     * vecs_l2sq : 计算结果
     * data : 向量数据
     */
    void compute_vecs_l2sq(float *vecs_l2sq, float *data, const size_t num_points, const size_t dim);

    void rotate_data_randomly(float *data, size_t num_points, size_t dim, float *rot_mat, float *&new_mat,
                              bool transpose_rot = false);


    // 找出每个 block 内数据点的最近的 k 个聚类中心（默认 1 个）；调用者把数据分块并行计算，该函数在每个 block 中单独计算
    /*
     * @param
     * data : 向量数据
     * centers : 聚类中心，向量
     * docs_l2sq : 预计算的数据点的平方 L2 范数
     * centers_l2sq : 预计算的聚类中心的平方 L2 范数
     * center_index : 输出结果，每个数据点的 k 个最近聚类中心的索引（是第几个中心）
     * dist_matrix : 输出结果，每个数据点到每个聚类中心的平方距离
     */
    void compute_closest_centers_in_block(const float *const data, const size_t num_points, const size_t dim,
                                          const float *const centers, const size_t num_centers,
                                          const float *const docs_l2sq, const float *const centers_l2sq,
                                          uint32_t *center_index, float *const dist_matrix, size_t k = 1);


    // 计算所有数据点的 k 最近邻聚类中心
    /*
     * @param
     * data : 向量数据
     * pivot_data : 聚类中心
     * k : 每个数据点的最近邻聚类中心数量
     * closest_centers_ivf : 输出结果，每个数据点的 k 个最近聚类中心的索引
     * inverted_index : 输出结果，每个聚类中心的倒排索引，是该聚类中的所有数据点的索引
     * pts_norms_squared : 预计算的数据点的平方 L2 范数
     */
    void compute_closest_centers(float *data, size_t num_points, size_t dim, float *pivot_data, size_t num_centers,
                                 size_t k, uint32_t *closest_centers_ivf, std::vector<size_t> *inverted_index = NULL,
                                 float *pts_norms_squared = NULL);

    // if to_subtract is 1, will subtract nearest center from each row. Else will
    // add. Output will be in data_load iself.
    // Nearest centers need to be provided in closst_centers.

    void process_residuals(float *data_load, size_t num_points, size_t dim, const float *cur_pivot_data, size_t num_centers,
                           uint32_t *closest_centers, bool to_subtract);


    /* ------------------ K-means --------------------*/

    // 执行 K-Means 算法的一次迭代，计算每个数据点到最近的聚类中心的距离，更新聚类中心，返回总残差（所有数据点到最近聚类中心的平方距离之和）
    /*
     * @param
     * data : 向量数据
     * centers : 聚类中心，向量
     * docs_l2sq : 预计算的数据点的平方 L2 范数
     * closest_docs : 每个聚类中的所有数据点
     * closest_center : 每个数据点的最近聚类中心的索引
     */
    float lloyds_iter(float *data, size_t num_points, size_t dim, float *centers, size_t num_centers, float *docs_l2sq,
                      std::vector<size_t> *closest_docs, uint32_t *&closest_center);

    // Run Lloyds until max_reps or stopping criterion
    // If you pass NULL for closest_docs and closest_center, it will NOT return
    // the results, else it will assume appriate allocation as closest_docs = new
    // vector<size_t> [num_centers], and closest_center = new size_t[num_points]
    // Final centers are output in centers as row major num_centers * dim
    //
    // 运行迭代，直到达到最大迭代次数或满足停止条件（参差变化率 < 0.00001）
    /*
     * @param
     * data : 向量数据
     * centers : 聚类中心，向量
     * max_reps : 最大迭代次数
     * closest_docs : 每个聚类中的所有数据点
     * closest_center : 每个数据点的最近聚类中心
     */
    float run_lloyds(float *data, size_t num_points, size_t dim, float *centers, const size_t num_centers,
                     const size_t max_reps, std::vector<size_t> *closest_docs, uint32_t *closest_center);


    // 选取聚类中心
    void selecting_pivots(float *data, size_t num_points, size_t dim, float *pivot_data, size_t num_centers);

    // kmeans++，优化的 kmeans 方法
    void kmeanspp_selecting_pivots(float *data, size_t num_points, size_t dim, float *pivot_data, size_t num_centers);
}


#endif //KMEANS_H
