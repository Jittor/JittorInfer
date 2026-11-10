#pragma once

#include "forch/exception.hpp"
#include "forch/forch.hpp"

#include <map>
#include <memory>
#include <string>
#include <type_traits>

namespace forch::nn {

struct Loader {
  virtual ~Loader()                        = default;
  virtual Tensor load(const std::string &) = 0;
};

struct Loadable {
  virtual ~Loadable()                                          = default;
  virtual void load(Loader *loader, const std::string &prefix) = 0;

  virtual Loadable *set_prefix_name(const std::string &prefix) {
    _name = prefix;
    return this;
  }

  virtual std::string prefix_name(const std::string &extend = "") const {
    if (extend.empty()) {
      return _name;
    }
    if (_name.empty()) {
      return extend;
    }
    return _name + "." + extend;
  }

  virtual std::string prefix_name(size_t index) const {
    return prefix_name(std::to_string(index));
  }

private:
  std::string _name;
};

struct Module : Loadable {
  ~Module() override = default;

  Module()                          = default;
  Module(const Module &)            = delete;
  Module &operator=(const Module &) = delete;
  Module(Module &&)                 = default;
  Module &operator=(Module &&)      = default;

  template <class T>
  T *register_loadable(const std::string &name, T &&parameter) {
    static_assert(
      std::is_base_of_v<Loadable, T>, "T must inherit from ParameterBase");

    if (_children.contains(name)) {
      throw exception(
        std::string("register_paramter: duplicate child name: ") + name);
    }

    auto  child_holder = std::make_unique<T>(std::forward<T>(parameter));
    auto *ptr          = child_holder.get();
    _children[name]    = std::move(child_holder);
    return ptr;
  }
  template <class T>
  T *register_loadable(T &) = delete;

  void load(Loader *loader, const std::string &prefix = "") override {
    set_prefix_name(prefix);
    for (auto &[n, module] : _children) {
      module->load(loader, prefix_name(n));
    }
  }

private:
  std::map<std::string, std::unique_ptr<Loadable>> _children;
};

template <class T, typename = std::enable_if_t<std::is_base_of_v<Loadable, T>>>
struct ModuleList : public Loadable, private std::vector<std::unique_ptr<T>> {
  ModuleList(std::vector<std::unique_ptr<T>> &&modules)
      : std::vector<std::unique_ptr<T>>(std::move(modules)) {}

  using std::vector<std::unique_ptr<T>>::size;
  using std::vector<std::unique_ptr<T>>::operator[];
  using std::vector<std::unique_ptr<T>>::begin;
  using std::vector<std::unique_ptr<T>>::end;
  using typename std::vector<std::unique_ptr<T>>::iterator;
  using typename std::vector<std::unique_ptr<T>>::const_iterator;

  void load(Loader *loader, const std::string &prefix) override {
    set_prefix_name(prefix);
    for (size_t i = 0; i < this->size(); ++i) {
      (*this)[i]->load(loader, prefix_name(i));
    }
  }
};

} // namespace forch::nn
