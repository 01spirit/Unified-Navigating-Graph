#include <iostream>
#include "distance.h"

// 一个 float 有 32 位

namespace ANNS {

    // 只能算浮点值的欧式距离
    std::unique_ptr<DistanceHandler> get_distance_handler(const std::string& data_type, const std::string& dist_fn) {
        if (data_type == "float") {
            if (dist_fn == "L2")
                return std::make_unique<FloatL2DistanceHandler>();
            else if (dist_fn == "IP") {
                std::cerr << "Not implement distance function: " << dist_fn << " for data type: " << data_type << std::endl;
                exit(-1);
            } else if (dist_fn == "cosine") {
                std::cerr << "Not implement distance function: " << dist_fn << " for data type: " << data_type << std::endl;
                exit(-1);
            } else {
                std::cerr << "Error: invalid distance function: " << dist_fn << " and data type: " << data_type << std::endl;
                exit(-1);
            }
        } else if (data_type == "int8") {
            std::cerr << "Not implement distance function: " << dist_fn << " for data type: " << data_type << std::endl;
            exit(-1);
        } else if (data_type == "uint8") {
            std::cerr << "Not implement distance function: " << dist_fn << " for data type: " << data_type << std::endl;
            exit(-1);
        } else {
            std::cerr << "Not implement distance function: " << dist_fn << " for data type: " << data_type << std::endl;
            exit(-1);
        }
    }

    // float L2 distance 计算浮点数向量之间 L2 距离（欧几里得距离），提供了多种实现方式  a/b : 两个向量的指针 dim : 向量维度
    float FloatL2DistanceHandler::compute(const char *a, const char *b, IdxType dim) const {
        const float *x = reinterpret_cast<const float *>(a);
        const float *y = reinterpret_cast<const float *>(b);

        /*
         *  朴素循环
         */
        // naive
        // float ans = 0;
        // for (IdxType i = 0; i < dim; i++)
        //     ans += (x[i] - y[i]) * (x[i] - y[i]);    // 逐个计算每个维度的差的平方，并累加到结果中
        // return ans;

        /*
         *  OpenMP SIMD 并行化
         */
        // simd
        // x = (const float *)__builtin_assume_aligned(x, 32);  // 告诉编译器内存地址是 32 字节对齐的，这可以提高内存访问的效率
        // y = (const float *)__builtin_assume_aligned(y, 32);
        // float ans = 0;
        // #pragma omp simd reduction(+ : ans) aligned(x, y : 32)   // 用 OpenMP 的 SIMD 并行化指令 #pragma omp simd 来加速计算
        // for (int32_t i = 0; i < (int32_t)dim; i++)
        //     ans += (x[i] - y[i]) * (x[i] - y[i]);
        // return ans;

        //  AVX-2
        __m256 msum0 = _mm256_setzero_ps();     // 处理 8 个浮点数（256 位），初始化一个 256 位的 SIMD 寄存器，用于累加中间结果

        while (dim >= 8) {  // 每次处理 8 个维度的浮点数
            __m256 mx = _mm256_loadu_ps(x); // 一次从内存中加载 8 个浮点数到寄存器
            x += 8;
            __m256 my = _mm256_loadu_ps(y);
            y += 8;
            const __m256 a_m_b1 = _mm256_sub_ps(mx, my);    // 对两个寄存器进行逐元素减法
            msum0 = _mm256_add_ps(msum0, _mm256_mul_ps(a_m_b1, a_m_b1)); // 对两个寄存器进行逐元素乘法，计算差的平方，累加中间结果
            dim -= 8;
        }

        // 从 256 位的 SIMD 寄存器中提取高 128 位和低 128 位，分别存储到两个 128 位的 SIMD 寄存器中
        // 将两个 128 位的 SIMD 寄存器中的元素逐元素相加，得到一个 128 位的 SIMD 寄存器 msum1，其中包含了所有中间结果的累加和
        __m128 msum1 = _mm256_extractf128_ps(msum0, 1);
        __m128 msum2 = _mm256_extractf128_ps(msum0, 0);
        msum1 = _mm_add_ps(msum1, msum2);

        if (dim >= 4) { // 还剩 4 - 8 个浮点数，先处理 4 个，128 位
            __m128 mx = _mm_loadu_ps(x);
            x += 4;
            __m128 my = _mm_loadu_ps(y);
            y += 4;
            const __m128 a_m_b1 = _mm_sub_ps(mx, my);
            msum1 = _mm_add_ps(msum1, _mm_mul_ps(a_m_b1, a_m_b1));
            dim -= 4;
        }

        if (dim > 0) { // 还剩不到 4 个浮点数
            __m128 mx = masked_read(dim, x);    // 手动加载数据
            __m128 my = masked_read(dim, y);
            __m128 a_m_b1 = _mm_sub_ps(mx, my);
            msum1 = _mm_add_ps(msum1, _mm_mul_ps(a_m_b1, a_m_b1));
        }

        // 对 128 位的 SIMD 寄存器中的元素进行水平相加，将相邻的两个元素相加，结果存储在低 128 位中
        // 第一次由 4 个浮点数加为 2 个，第二次剩下 1 个
        msum1 = _mm_hadd_ps(msum1, msum1);
        msum1 = _mm_hadd_ps(msum1, msum1);
        return _mm_cvtss_f32(msum1); // 将 128 位 SIMD 寄存器中的第一个浮点数转换为标量浮点数，返回最终的 L2 距离

        // AVX-512
        // __m512 msum0 = _mm512_setzero_ps();

        // while (dim >= 16) {
        //     __m512 mx = _mm512_loadu_ps(x);
        //     x += 16;
        //     __m512 my = _mm512_loadu_ps(y);
        //     y += 16;
        //     const __m512 a_m_b1 = mx - my;
        //     msum0 += a_m_b1 * a_m_b1;
        //     dim -= 16;
        // }

        // __m256 msum1 = _mm512_extractf32x8_ps(msum0, 1);
        // msum1 += _mm512_extractf32x8_ps(msum0, 0);

        // if (dim >= 8) {
        //     __m256 mx = _mm256_loadu_ps(x);
        //     x += 8;
        //     __m256 my = _mm256_loadu_ps(y);
        //     y += 8;
        //     const __m256 a_m_b1 = mx - my;
        //     msum1 += a_m_b1 * a_m_b1;
        //     dim -= 8;
        // }

        // __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
        // msum2 += _mm256_extractf128_ps(msum1, 0);

        // if (dim >= 4) {
        //     __m128 mx = _mm_loadu_ps(x);
        //     x += 4;
        //     __m128 my = _mm_loadu_ps(y);
        //     y += 4;
        //     const __m128 a_m_b1 = mx - my;
        //     msum2 += a_m_b1 * a_m_b1;
        //     dim -= 4;
        // }

        // if (dim > 0) {
        //     __m128 mx = masked_read(dim, x);
        //     __m128 my = masked_read(dim, y);
        //     __m128 a_m_b1 = mx - my;
        //     msum2 += a_m_b1 * a_m_b1;
        // }

        // msum2 = _mm_hadd_ps(msum2, msum2);
        // msum2 = _mm_hadd_ps(msum2, msum2);
        // return _mm_cvtss_f32(msum2);
    }

    // 从一个浮点数数组 x 中读取指定数量的浮点数，并将它们加载到一个 128 位的 SIMD 寄存器中
    __m128 FloatL2DistanceHandler::masked_read(IdxType dim, const float *x) {
        __attribute__((__aligned__(16))) float buf[4] = {0, 0, 0, 0};   // 使用一个 16 字节对齐的缓冲区 buf，初始化为 0
        switch (dim) {  // 根据 dim 的值，将 x 中的前 dim 个浮点数复制到 buf 中
            case 3:     // 没有 break，会依次顺序执行 buf 的赋值
                buf[2] = x[2];
            case 2:
                buf[1] = x[1];
            case 1:
                buf[0] = x[0];
        }
        return _mm_load_ps(buf);    // 将 buf 加载到 128 位的 SIMD 寄存器中
    }
}