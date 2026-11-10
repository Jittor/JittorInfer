#include "forch/huggingface/pretrained.hpp"
#include "forch/exception.hpp"
#include "forch/forch.hpp"
#include "forch/huggingface/safetensors.hpp"

#include <filesystem>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>

namespace forch::huggingface {

namespace fs = std::filesystem;

struct Pretrained::Impl {
  fs::path model_path;

  bool                               is_shared;
  std::map<std::string, Safetensors> sf_by_tensor;

  Impl(const std::string &path) {
    model_path = fs::path(path);

    if (!fs::exists(model_path)) {
      throw exception(
        std::string("Model path does not exist: ") + model_path.string());
    }
    if (!fs::is_directory(model_path)) {
      throw exception(
        std::string("Model path is not a directory: ") + model_path.string());
    }

    auto index_path = model_path / "model.safetensors.index.json";
    is_shared       = fs::exists(index_path);

    if (!is_shared) {
      logger()->info(
        "load model '{}' from single safetensors file", model_path.string());

      sf_by_tensor.emplace(
        "", Safetensors((model_path / "model.safetensors").string()));
    } else {
      logger()->info(
        "load model '{}' from multiple safetensors files with index file: {}",
        model_path.string(), index_path.string());

      std::ifstream  index_file(index_path);
      nlohmann::json j;
      index_file >> j;

      auto load_safetensors_by_name =
        [cache = std::map<fs::path, Safetensors>()](
          const fs::path &file_name) mutable -> Safetensors {
        auto it = cache.find(file_name);
        if (it != cache.end()) {
          return it->second;
        }
        auto sf = Safetensors(file_name.string());
        cache.emplace(file_name, sf);
        return sf;
      };

      for (const auto &[tensor_name, info] : j["weight_map"].items()) {
        sf_by_tensor.emplace(tensor_name,
          load_safetensors_by_name(model_path / info.get<std::string>()));
      }
    }
  };
};

Pretrained::Pretrained(const std::string &path)
    : impl(std::make_shared<Impl>(path)) {}

Tensor Pretrained::load(const std::string &name) {
  auto it = impl->sf_by_tensor.find(impl->is_shared ? name : "");
  if (it == impl->sf_by_tensor.end()) {
    throw exception(std::string("Tensor not found in model: ") + name);
  }
  return it->second.load(name);
}

} // namespace forch::huggingface