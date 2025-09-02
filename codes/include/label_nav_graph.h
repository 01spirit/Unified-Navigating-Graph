#ifndef LABEL_NAV_GRAPH_H
#define LABEL_NAV_GRAPH_H

#include <vector>
#include "config.h"

/*
 * 标签导航图 LNG ：前面用前缀树把向量的标签集合划分成了 group，每个 group 对应树中的一个终端节点，也对应 LNG 中的一个图节点
 * LNG 节点的出边指向包含其标签的超集节点，所有出边邻居的标签集合都是该节点的超集，所有入边邻居的标签集合都是该节点的子集
 */

namespace ANNS {

    class LabelNavGraph {

        public:
            // 图索引从 1 开始
            LabelNavGraph(IdxType num_nodes) {
                in_neighbors.resize(num_nodes+1);
                out_neighbors.resize(num_nodes+1);
            };

            // 节点的出入边邻居节点，第一个 vector 是图中的节点，第二个 vector 是每个节点的邻居节点
            std::vector<std::vector<IdxType>> in_neighbors, out_neighbors;

            ~LabelNavGraph() = default;

        private:
            
    };
}



#endif // LABEL_NAV_GRAPH_H