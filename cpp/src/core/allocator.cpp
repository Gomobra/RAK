#include "rak/core/allocator.h"

#include <new>

namespace rak::core {

void* ArenaAllocator::allocate(size_t n) {
  if (n == 0) return nullptr;
  const size_t align = 64;  // align cache line untuk SIMD
  size_t padded = (n + align - 1) / align * align;
  void* p = ::operator new(padded);
  blocks_.push_back({p, padded});
  total_allocated_ += padded;
  return p;
}

void ArenaAllocator::reset() {
  for (auto& b : blocks_) {
    ::operator delete(b.ptr);
  }
  blocks_.clear();
  total_allocated_ = 0;
}

}  // namespace rak::core
