#include "rak/core/autograd.h"

#include <algorithm>
#include <unordered_set>
#include <iostream>

namespace rak::core {

Autograd g_autograd;

void Autograd::backward(std::shared_ptr<Tensor> loss, bool retain_graph) {
  if (!loss) throw std::invalid_argument("Autograd::backward loss null");
  if (loss->numel() != 1) {
    throw std::invalid_argument("Autograd::backward loss harus skalar (numel == 1)");
  }

  // Topological sort iteratif (DFS dengan stack eksplicite), aman untuk graf
  // dalam yang ribuan node tanpa overflow stack.
  std::vector<std::shared_ptr<GradNode>> order;
  std::unordered_set<const GradNode*> visited;
  std::stack<std::pair<std::shared_ptr<GradNode>, bool>> stk;

  auto seed = loss->grad_fn;
  if (seed) stk.push({seed, false});

  while (!stk.empty()) {
    auto [node, processed] = stk.top();
    stk.pop();
    if (processed) {
      order.push_back(node);
      continue;
    }
    if (visited.count(node.get())) continue;
    visited.insert(node.get());
    stk.push({node, true});
    for (auto& w : node->inputs()) {
      if (auto t = w.lock()) {
        if (t->grad_fn) stk.push({t->grad_fn, false});
      }
    }
  }

  // grad_map: node -> grad_output (akumulasi untuk graf berdiamond).
  std::unordered_map<const GradNode*, std::vector<float>> grad_map;
  grad_map[seed.get()] = {1.0f};  // d loss / d loss = 1

  // Traverse urutan topological terbalik (root terlebih dahulu).
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    auto& node = *it;
    auto found = grad_map.find(node.get());
    if (found == grad_map.end()) continue;
    auto& grad_out = found->second;
    auto grad_inputs = node->apply(grad_out);
    const auto& inputs = node->inputs();
    for (size_t i = 0; i < inputs.size() && i < grad_inputs.size(); ++i) {
      if (auto t = inputs[i].lock()) {
        if (t->grad_fn) {
          // Intermediate tensor: propagasikan gradien ke grad_fn-nya untuk
          // melanjutkan backprop melalui rantai operasi.
          auto& acc = grad_map[t->grad_fn.get()];
          if (acc.empty()) acc.assign(grad_inputs[i].size(), 0.0f);
          if (acc.size() != grad_inputs[i].size()) {
            throw std::runtime_error("Autograd: ukuran gradien tidak cocok saat akumulasi");
          }
          for (size_t j = 0; j < grad_inputs[i].size(); ++j) acc[j] += grad_inputs[i][j];
        } else if (t->requires_grad) {
          // Leaf tensor yang membutuhkan gradien: tulis ke buffer grad.
          if (t->grad.size() != grad_inputs[i].size()) {
            t->grad.assign(grad_inputs[i].size(), 0.0f);
          }
          for (size_t j = 0; j < grad_inputs[i].size(); ++j) t->grad[j] += grad_inputs[i][j];
        }
        // Skip leaf tensor yang tidak requires_grad (tidak ada buffer grad).
      }
    }
  }

  if (!retain_graph) {
    reset();
  }
}

void Autograd::reset() {
  nodes_.clear();
}

std::shared_ptr<Tensor> make_op_output(
    std::vector<std::shared_ptr<Tensor>> inputs,
    std::vector<float> forward_values,
    std::vector<size_t> shape,
    GradNode::BackwardFn backward) {
  auto out = std::make_shared<Tensor>(shape, std::move(forward_values), false);
  std::vector<std::weak_ptr<Tensor>> weak_inputs;
  weak_inputs.reserve(inputs.size());
  for (auto& in : inputs) weak_inputs.push_back(in);
  auto node = std::make_shared<GradNode>(std::move(backward), std::move(weak_inputs));
  out->grad_fn = node;
  g_autograd.record(node);
  return out;
}

}  // namespace rak::core
