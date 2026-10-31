#ifndef RAK_CORE_ALLOCATOR_H
#define RAK_CORE_ALLOCATOR_H

#include <cstddef>
#include <vector>

namespace rak::core {

// Alokator arena: alokasi murah, memori terprediksi, dibebaskan sekaligus.
// Digunakan untuk buffer tensor sehingga alokasi per-op tidak memecah heap.
class ArenaAllocator {
 public:
  ArenaAllocator() = default;
  ~ArenaAllocator() = default;

  ArenaAllocator(const ArenaAllocator&) = delete;
  ArenaAllocator& operator=(const ArenaAllocator&) = delete;
  ArenaAllocator(ArenaAllocator&&) = default;
  ArenaAllocator& operator=(ArenaAllocator&&) = default;

  // Alokasi n byte, mengembalikan pointer ter-align 64 byte.
  void* allocate(size_t n);
  // Bebaskan semua blok yang telah dialokasi (reset arena).
  void reset();
  size_t total_allocated() const { return total_allocated_; }

 private:
  struct Block {
    void* ptr;
    size_t size;
  };
  std::vector<Block> blocks_;
  size_t total_allocated_ = 0;
};

}  // namespace rak::core

#endif  // RAK_CORE_ALLOCATOR_H
