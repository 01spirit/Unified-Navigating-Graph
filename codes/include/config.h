#ifndef ANNS_CONFIG_H
#define ANNS_CONFIG_H

#include <cstdint>
#include <cstddef>

/**
 *  候选集：用于在搜索过程中高效地找到与查询向量最相似的向量，通常是一个包含多个候选向量的集合
 *          直接计算查询向量与所有数据点之间的距离是非常耗时的。通过选择一个较小的候选集，可以显著减少需要计算的距离数量，从而提高搜索效率
 *          在基于图的 ANNS 算法（如 Vamana）中，候选集通常是通过图结构来生成的。从一个起始节点（如入口点）开始，
 *          通过图的边逐步扩展，找到与查询向量相似的节点。这些节点及其邻居节点构成了候选集。
 *
 *      在构建图时，每个节点会从候选集中选择最相似的节点进行连接。候选集的大小由参数 Lbuild 控制
 *      在搜索时，从入口点开始，通过图的边逐步扩展，找到与查询向量相似的节点。这些节点及其邻居节点构成了候选集。候选集的大小由参数 Lsearch 控制
 *
 *  ALPHA ： 用于控制图的稀疏性的参数。它影响图中边的权重，从而影响搜索过程中的路径选择
 *          ALPHA 越小，图越密集，每个节点有更多邻居，搜索时可选路径越多，但计算量和内存占用更大；提高搜索精度
 *          ALPHA 越大，图越稀疏，；提高搜索效率，但是可能错过最优路径
 *
 *  GRAPH_SLACK_FACTOR ：用于控制图的松弛程度的参数。它影响图的构建和搜索过程中的行为，特别是在处理边界条件和动态调整时。
 *          越小，图越严格，搜索时更倾向于选择最短路径，提高搜索精度，增加计算量
 *          越小，图越松弛，搜索时允许更多的路径选择，提高搜索效率
 *
 **/

namespace ANNS {

    // type for storing the id of vectors   向量 id
    using IdxType = uint32_t;

    // type for storing the label of vectors    向量的标签，标签用数值表示
    using LabelType = uint16_t;

    // type for marks which are refreshed in each search    每次搜索中刷新的标记
    using MarkType = uint16_t;

    enum DataType {
        FLOAT = 0,
        UINT8 = 1,
        INT8 = 2
    };

    // 距离度量
    enum Metric {
        L2 = 0,     // 欧式距离
        INNER_PRODUCT = 1,  // 内积
        COSINE = 2  // 余弦相似度
    };

    // default parameters
    namespace default_paras {
        const uint32_t NUM_THREADS = 1;

        // general graph indices
        const IdxType MAX_DEGREE = 64;  // 每个节点的最大度数
        const IdxType L_BUILD = 100;    // 构件图时的候选集大小
        const IdxType L_SEARCH = 100;   // 搜索时的候选集大小

        // for vamana
        const IdxType MAX_CANDIDATE_SIZE = 750;     // 最大候选集大小
        const float ALPHA = 1.2;    // 控制图的稀疏性
        const float GRAPH_SLACK_FACTOR = 1.3;   // 图的松弛因子

        // for Unified Navigating Graph
        const IdxType NUM_ENTRY_POINTS = 16;    // UNG 入口点数量
        const IdxType NUM_CROSS_EDGES = 6;      // UNG 交叉边数量
    }
}

#endif // ANNS_CONFIG_H