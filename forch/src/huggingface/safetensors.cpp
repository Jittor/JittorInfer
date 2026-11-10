#include "forch/huggingface/safetensors.hpp"
#include "forch/exception.hpp"
#include "forch/forch.hpp"
#include "forch/ops/const.hpp"

#include <map>
#include <nlohmann/json.hpp>
#include <sys/mman.h>
#include <sys/stat.h>

namespace forch::huggingface {

namespace {

DType parse_dtype(const std::string &dtype_str) {
  // safetensors 规范里的 dtype 字符串是 F64 / F32 / F16 / BF16 / I8 / I32 /
  // I64 / BOOL（见其 Dtype 枚举）；这里原先只认 FP64 / FP32 / FP16 / INT8…
  // 的别名，只有 BF16 恰好同名 —— 于是 fp16/fp32 检查点会直接抛
  // "Unsupported dtype: F16"。两种写法都收，规范名在前。
  if (dtype_str == "F64" || dtype_str == "FP64") {
    return DType::FP64;
  } else if (dtype_str == "F32" || dtype_str == "FP32") {
    return DType::FP32;
  } else if (dtype_str == "F16" || dtype_str == "FP16") {
    return DType::FP16;
  } else if (dtype_str == "BF16") {
    return DType::BF16;
  } else if (dtype_str == "I8" || dtype_str == "INT8") {
    return DType::INT8;
  } else if (dtype_str == "I32" || dtype_str == "INT32") {
    return DType::INT32;
  } else if (dtype_str == "I64" || dtype_str == "INT64") {
    return DType::INT64;
  } else if (dtype_str == "BOOL") {
    return DType::BOOL;
  } else {
    throw exception(std::string("Unsupported dtype: ") + dtype_str);
  }
}

} // namespace

struct Safetensors::Impl {
  struct File {
    const std::string filename;
    int               fd;
    struct stat       sb;
    Buffer            data_raw;
    Buffer            data_tensor;
    size_t            header_size;

    File(std::string filename) : filename(std::move(filename)) {
      logger()->trace("load safetensors: filename={}", filename);

      fd = open(this->filename.c_str(), O_RDONLY);
      if (fd < 0) {
        throw exception(std::string("Failed to open file: ") + this->filename);
      }

      if (fstat(fd, &sb) < 0) {
        close(fd);
        throw exception(std::string("Failed to stat file: ") + this->filename);
      }

      void *map = mmap(nullptr, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
      if (map == MAP_FAILED) {
        close(fd);
        throw exception(std::string("Failed to mmap file: ") + this->filename);
      }
      data_raw = static_cast<Buffer>(map);

      // load first 8-bytes in little-endian as header size
      if (sb.st_size < 8) {
        throw exception(
          std::string("illegal safetensors file: ") + this->filename);
      }
      header_size = 0;
      for (int i = 0; i < 8; ++i) {
        header_size += (static_cast<size_t>(data_raw[i]) << (8 * i));
      }

      data_tensor = data_raw + 8 + header_size;
    }

    ~File() {
      munmap(data_raw, sb.st_size);
      close(fd);
    }

    std::string_view header_string() const {
      const auto *c = reinterpret_cast<const char *>(data_raw + 8);
      return {c, header_size};
    }
  } file;

  struct MetaInfo {
    struct Tensor {
      DType  dtype;
      Shape  shape;
      size_t start;
      size_t end;
    };

    std::map<std::string, Tensor> tensors;

    explicit MetaInfo(const std::string_view json_str) {
      auto j = nlohmann::json::parse(json_str);
      for (const auto &[name, info] : j.items()) {
        if (name == "__metadata__") {
          continue;
        }
        logger()->trace(
          "load tensor meta: name={}, meta={}", name, info.dump());
        tensors[name] = {
          .dtype = parse_dtype(info.at("dtype")),
          .shape = info.at("shape").get<Shape>(),
          .start = info.at("data_offsets").at(0).get<size_t>(),
          .end   = info.at("data_offsets").at(1).get<size_t>(),
        };
      }
    }
  } meta;

  Impl(std::string filename)
      : file(std::move(filename)), meta(file.header_string()) {}
};

Safetensors::Safetensors(const std::string &path)
    : impl(std::make_shared<Impl>(path)) {}

Tensor Safetensors::load(const std::string &name) {
  logger()->trace("load tensor: name={}", name);

  auto   tensor_info = impl->meta.tensors.at(name);
  Buffer data_ptr    = impl->file.data_tensor + tensor_info.start;
  return op::load_from_buffer(data_ptr, tensor_info.shape, tensor_info.dtype);
}

} // namespace forch::huggingface