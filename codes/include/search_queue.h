#ifndef SEARCH_QUQUE
#define SEARCH_QUQUE

#include <vector>
#include <memory>
#include "config.h"

/*
 * bool expanded：在搜索过程中，一个向量可能通过多个路径访问到（扩展），只需要被扩展一次，避免重复计算
 * int32_t _cur_unexpanded：跟踪搜索队列中第一个未扩展的候选向量，每次处理一个向量后指向下一个未扩展向量
 * */

namespace ANNS {

    // for storing each candidate, prefer those with minimal distances  候选向量
    struct Candidate {
        IdxType id;     // 向量索引
        float distance; // 候选向量到查询向量的距离
        bool expanded;  // 扩展标记

        Candidate() = default;
        Candidate(IdxType a, float b) : id{a}, distance{b}, expanded(false) {}

        // 候选向量比较，先选距离近的，再选 id 小的
        inline bool operator<(const Candidate &other) const {
            return distance < other.distance || (distance == other.distance && id < other.id);
        }
        inline bool operator==(const Candidate &other) const { return (id == other.id); }
    };


    // search queue for ANNS, preserve the closest vectors  候选向量的队列，向量按距离从小到大排列，插入时保证顺序
    class SearchQueue {
        
        public:
            SearchQueue() : _size(0), _capacity(0), _cur_unexpanded(0) {};
            ~SearchQueue() = default;

            // size
            int32_t size() const { return _size; };
            int32_t capacity() const { return _capacity; };
            void reserve(int32_t capacity);

            // read and write
            Candidate operator[](int32_t idx) const { return _data[idx]; }
            Candidate& operator[](int32_t idx) { return _data[idx]; };
            bool exist(IdxType id) const;
            void insert(IdxType id, float distance);    // 按大小顺序插入
            void clear() { _size = 0; _cur_unexpanded = 0; };

            // expand
            bool has_unexpanded_node() const { return _cur_unexpanded < _size; };
            // 找到下一个距离最近的未扩展向量
            const Candidate& get_closest_unexpanded();
            
        private:

            int32_t _size, _capacity, _cur_unexpanded;  // 当前候选向量的数量，容量，当前未扩展的向量的位置
            std::vector<Candidate> _data;   // 候选向量
    };
}

#endif // SEARCH_QUQUE