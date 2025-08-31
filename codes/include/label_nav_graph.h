#ifndef LABEL_NAV_GRAPH_H
#define LABEL_NAV_GRAPH_H

#include <vector>
#include "config.h"


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