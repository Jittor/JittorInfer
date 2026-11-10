#include "forch/graph.hpp"
#include "forch/exception.hpp"
#include "graph.hpp"

#include <memory>
#include <string>
#include <utility>

namespace forch {

Graph::Graph(TensorList outputs)
    : _impl(std::make_shared<Graph::Impl>(std::move(outputs))) {}

std::string Graph::name() const {
  return _impl->graph_name;
}

Graph &Graph::set_name(const std::string &name) {
  _impl->graph_name = name;
  return *this;
}

TensorList Graph::outputs() const {
  return _impl->outputs;
}

Backend::Backend(std::shared_ptr<Impl> impl) : _impl(std::move(impl)) {}

Backend::Graph::Graph(std::shared_ptr<Impl> impl) : _impl(std::move(impl)) {}

TensorList Backend::Graph::Output::filter_name(const std::string &name) const {
  TensorList out;
  out.reserve(this->size());
  for (const auto &t : *this) {
    if (t.get_name() == name) {
      out.push_back(t);
    }
  }
  return out;
}

Tensor Backend::Graph::Output::get(const std::string &name) const {
  auto filtered = filter_name(name);
  if (filtered.size() != 1) {
    throw exception(std::string("Expected exactly one output tensor with name ")
      + name + ", but got " + std::to_string(filtered.size()));
  }
  return filtered[0];
}

Backend::Graph &Backend::Graph::compile() {
  _impl->compile();
  return *this;
}

Backend::Graph &Backend::Graph::load() {
  _impl->load();
  return *this;
}

Backend::Graph::Output Backend::Graph::run(const TensorList &inputs) {
  return Output(_impl->run(inputs));
}

Backend::Graph Backend::build(const forch::Graph &graph) {
  return _impl->build(_impl, graph);
}

Tensor ToDevice::operator()(const Tensor &t) const {
  return backend.move_to_device(t);
}

Tensor Backend::move_to_device(const Tensor &tensor) const {
  return _impl->move_to_device(tensor);
}

} // namespace forch