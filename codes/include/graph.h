#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <mutex>
#include <fstream>
#include <sstream>
#include "config.h"


namespace ANNS {

    class Graph {
            
        public:
            std::vector<IdxType>* neighbors;    // 动态分配的节点数组，每个vector表示一个节点的邻居节点
            std::mutex* neighbor_locks;

            Graph() = default;

            Graph(IdxType num_points) {
                _num_points = num_points;
                neighbors = new std::vector<IdxType>[num_points];
                neighbor_locks = new std::mutex[num_points];
            };

            // 提取子图
            Graph(std::shared_ptr<Graph> graph, IdxType start, IdxType end) {
                neighbors = graph->neighbors + start;
                neighbor_locks = graph->neighbor_locks + start;
                _num_points = end - start;
            };

            // 遍历所有节点，将每个节点及其邻居节点写入文件
            void save(std::string& filename) {
                std::ofstream out(filename);
                for (IdxType i = 0; i < _num_points; i++) { // 第 i 个节点的所有邻居节点
                    out << i << " ";
                    for (auto& neighbor : neighbors[i])
                        out << neighbor << " ";
                    out << std::endl;
                }
                out.close();
            }

            void load(std::string& filename) {
                std::ifstream in(filename);
                std::string line;
                IdxType id, neighbor;
                while (std::getline(in, line)) {
                    std::istringstream iss(line);
                    iss >> id;
                    neighbors[id].clear();
                    while (iss >> neighbor) 
                        neighbors[id].push_back(neighbor);
                }
                in.close();
            }

            // 图索引的大小，每个节点的邻居节点数量 * sizeof(IdxType)
            float get_index_size() {
                float index_size = 0;
                for (IdxType i = 0; i < _num_points; i++)
                    index_size += neighbors[i].size() * sizeof(IdxType);
                return index_size;
            }

            void clean() {
                delete[] neighbors;
                delete[] neighbor_locks;
            }

            ~Graph() = default;

        private:

            IdxType _num_points;    // 图中节点数量
            
    };
}

#endif // GRAPH_H