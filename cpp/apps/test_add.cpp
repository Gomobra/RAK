#include <cstdio>
#include <memory>
#include <vector>
#include "rak/core/tensor.h"
#include "rak/core/autograd.h"
#include "rak/ops/elementwise.h"

using namespace rak::core;
using namespace rak::ops;

int main() {
  auto a = std::make_shared<Tensor>(std::vector<size_t>{3}, true);
  auto b = std::make_shared<Tensor>(std::vector<size_t>{3}, true);
  printf("tensors created, numel=%zu\n", a->numel());
  auto c = add(a, b);
  printf("after add call\n");
  printf("add OK, numel=%zu\n", c->numel());
  printf("c[0]=%f\n", c->at(0));
  return 0;
}
