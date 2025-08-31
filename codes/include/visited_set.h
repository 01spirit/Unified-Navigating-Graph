#ifndef VISITED_SET_H
#define VISITED_SET_H

#include <cstring>
#include "config.h"

/**
 * 管理访问状态，标记和检查节点是否被访问过
 * 每次搜索或操作开始时，_curValue 会增加。这确保了每次搜索或操作使用的标记值是唯一的
 * 通过增加 _curValue，可以避免不同搜索或操作之间的标记冲突
 */
namespace ANNS {
    class VisitedSet {
        public:
            VisitedSet() = default;

            void init(IdxType num_elements) {
                _curValue = -1;
                _num_elements = num_elements;
                if (_marks != nullptr)
                    delete[] _marks;
                _marks = new MarkType[num_elements];
            }

            void clear() {
                _curValue++;
                if (_curValue == 0) {
                    memset(_marks, 0, sizeof(MarkType) * _num_elements);
                    _curValue++;
                }
            }

            // 预取指定索引的标记
            inline void prefetch(IdxType idx) const {
                _mm_prefetch((char *)_marks + idx, _MM_HINT_T0); // 将数据提前加载到缓存中，减少访问延迟
            }

            inline void set(IdxType idx) { 
                _marks[idx] = _curValue; 
            }

            inline bool check(IdxType idx) const { 
                return _marks[idx] == _curValue; 
            }

            ~VisitedSet() { 
                delete[] _marks; 
            }

        private:
            MarkType _curValue;     // 当前标记值
            MarkType* _marks = nullptr;     // 每个元素的访问状态
            IdxType _num_elements;  // 需要管理的元素数量
    };
}

#endif // VISITED_SET_H