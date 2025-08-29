#ifndef TRIE_TREE_H
#define TRIE_TREE_H

#include <vector>
#include <map>
#include <memory>
#include "config.h"

/*
 * 用前缀树作为数值型标签的索引，可用于查找标签集合及其超集等
 * 前缀树的每个终端节点代表一个完整的标签集合，用 group 把这个标签集合与数据点关联起来
 */

namespace ANNS {

    // trie tree node
    struct TrieNode {
        LabelType label;    // 每个节点对应一个标签
        IdxType group_id;                       // group_id>0, and 0 if not a terminal node     每个终端节点有唯一的 group_id
        LabelType label_set_size;               // number of elements in the label set if it is a terminal node     标签集合中的标签数量
        IdxType group_size;                     // number of elements in the group if it is a terminal node     终端节点的 group 中的数据点数量

        std::shared_ptr<TrieNode> parent;
        std::map<LabelType, std::shared_ptr<TrieNode>> children;    // 该节点的多个子节点，每个子节点由一个标签映射，代表只有该子节点存储了这个标签

        TrieNode(LabelType x, std::shared_ptr<TrieNode> y)
            : label(x), parent(y), group_id(0), label_set_size(0), group_size(0) {}
        TrieNode(LabelType a, IdxType b, LabelType c, IdxType d)
            : label(a), group_id(b), label_set_size(c), group_size(d) {}
        ~TrieNode() = default;
    };


    // trie tree construction and search for super sets
    class TrieIndex {

        public:
            TrieIndex();

            // construction
            IdxType insert(const std::vector<LabelType>& label_set, IdxType& new_label_set_id);

            // query
            LabelType get_max_label_id() const { return _max_label_id; }
            // 找到标签集合完全匹配的终端节点
            std::shared_ptr<TrieNode> find_exact_match(const std::vector<LabelType>& label_set) const;
            // 找到这个标签集合的所有超集的入口节点，用参数指定是否包含精确匹配自身的节点；假设输入标签集合已经按升序排列
            void get_super_set_entrances(const std::vector<LabelType>& label_set, 
                                         std::vector<std::shared_ptr<TrieNode>>& super_set_entrances, 
                                         bool avoid_self=false, bool need_containment=true) const;

            // I/O
            void save(std::string filename) const;
            void load(std::string filename);
            float get_index_size();

        private:
            LabelType _max_label_id = 0;
            std::shared_ptr<TrieNode> _root;
            std::vector<std::vector<std::shared_ptr<TrieNode>>> _label_to_nodes;    // 标签到前缀树节点的倒排索引，第一个 vector 是数值型标签，第二个 vector 是每个标签关联的倒排索引节点

            // help function for get_super_set_entrances
            // 该节点是否是包含该标签集合的最小节点，他的父节点中不会有标签集合中的值标签
            bool examine_smallest(const std::vector<LabelType>& label_set, const std::shared_ptr<TrieNode>& node) const;
            // 该节点是否是标签集合的超集，必须包含集合外的元素
            bool examine_containment(const std::vector<LabelType>& label_set, const std::shared_ptr<TrieNode>& node) const;
    };
}

#endif // TRIE_TREE_H