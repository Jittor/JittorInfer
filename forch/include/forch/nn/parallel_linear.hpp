#pragma once

#include "forch/distributed.hpp"
#include "forch/nn/linear.hpp"
#include "forch/nn/module.hpp"
#include "forch/nn/parameter.hpp"
#include "forch/ops/batched_mat_mul.hpp"
#include "forch/ops/collective.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/mat_mul.hpp"
#include <array>
#include <cstring>
#include <limits>

namespace forch::nn {
struct WeightShard {
  // Own bytes rather than converting through FP32: BF16/FP16 checkpoint bits
  // must survive loading exactly, and temporary loaders may be destroyed.
  struct ShardConstant : op::Const {
    mutable std::vector<uint8_t> data;
    Shape                        dimensions;
    DType                        type;
    ShardConstant(std::vector<uint8_t> bytes, Shape shape, DType dtype)
        : data(std::move(bytes)), dimensions(std::move(shape)), type(dtype) {}
    Buffer buffer() const override {
      return data.data();
    }
    Shape shape(const TensorList &) const override {
      return dimensions;
    }
    DType dtype(const TensorList &) const override {
      return type;
    }
  };

  static size_t multiply(size_t a, size_t b) {
    if (b && a > std::numeric_limits<size_t>::max() / b)
      throw exception("ParallelLinear: weight byte size overflow");
    return a * b;
  }
  static size_t element_size(DType type) {
    switch (type) {
    case DType::FP64:
    case DType::INT64:
      return 8;
    case DType::FP32:
    case DType::INT32:
      return 4;
    case DType::FP16:
    case DType::BF16:
      return 2;
    case DType::INT8:
    case DType::BOOL:
      return 1;
    }
    throw exception("ParallelLinear: unsupported storage dtype");
  }
  static Tensor shard(
    const Tensor &source, size_t axis, int64_t start, int64_t count) {
    auto *constant = dynamic_cast<const op::Const *>(source.get_op());
    if (!constant)
      throw exception("ParallelLinear: loader must return a host constant");
    auto shape = source.shape();
    if (axis >= shape.size() || start < 0 || count <= 0
      || start > shape[axis] - count)
      throw exception("ParallelLinear: invalid weight slice");
    size_t outer = 1, inner = element_size(source.dtype());
    for (size_t i = 0; i < axis; ++i)
      outer = multiply(outer, shape[i]);
    for (size_t i = axis + 1; i < shape.size(); ++i)
      inner = multiply(inner, shape[i]);
    const auto source_stride = multiply(shape[axis], inner);
    const auto local_stride  = multiply(count, inner);
    multiply(outer, source_stride);
    std::vector<uint8_t> bytes(multiply(outer, local_stride));
    if (!constant->buffer())
      throw exception("ParallelLinear: null weight buffer");
    for (size_t row = 0; row < outer; ++row)
      std::memcpy(bytes.data() + row * local_stride,
        constant->buffer() + row * source_stride + multiply(start, inner),
        local_stride);
    shape[axis] = count;
    return Tensor({},
      std::make_unique<ShardConstant>(
        std::move(bytes), std::move(shape), source.dtype()));
  }
};

// Consumes replicated inputs and produces local output-channel shards.
class ColumnParallelLinear : public Module {
public:
  ColumnParallelLinear(int64_t in_features, int64_t out_features,
    ProcessGroup group, bool bias = true, DType dtype = DType::FP32)
      : in_(in_features), out_(out_features), group_(group) {
    if (in_ <= 0 || out_ <= 0)
      throw exception("ColumnParallelLinear: feature sizes must be positive");
    if (out_ % group_.world_size())
      throw exception(
        "ColumnParallelLinear: output features must divide world_size");
    local_in_  = in_;
    local_out_ = out_ / group_.world_size();
    weight_ =
      register_loadable("weight", Parameter({local_out_, local_in_}, dtype));
    if (bias)
      bias_ = register_loadable("bias", Parameter({local_out_}, dtype));
  }
  int64_t in_features() const {
    return in_;
  }
  int64_t out_features() const {
    return out_;
  }
  int64_t local_in_features() const {
    return local_in_;
  }
  int64_t local_out_features() const {
    return local_out_;
  }
  const ProcessGroup &group() const {
    return group_;
  }
  const Tensor &weight() const {
    return weight_->get();
  }

  void load(Loader *loader, const std::string &prefix = "") override {
    if (!loader)
      throw exception("ColumnParallelLinear: null loader");
    ShardLoader shards(loader, *this, prefix);
    Module::load(&shards, prefix);
  }
  Tensor forward(const Tensor &input) const {
    if (input.dim() < 2 || input.shape().back() != local_in_)
      throw exception(
        "ColumnParallelLinear: input must end in local_in_features");
    auto output = input.dim() > 2
      ? op::batched_mat_mul(false, true)(input, weight_->get())
      : op::mat_mul(false, true)(input, weight_->get());
    if (bias_)
      output = op::add(output, bias_->get());
    return output;
  }

private:
  struct ShardLoader : Loader {
    Loader                     *source;
    const ColumnParallelLinear &layer;
    std::string                 weight_name;
    ShardLoader(Loader *loader, const ColumnParallelLinear &linear,
      const std::string &prefix)
        : source(loader), layer(linear),
          weight_name(prefix.empty() ? "weight" : prefix + ".weight") {}
    Tensor load(const std::string &name) override {
      auto       tensor = source->load(name);
      const bool weight = name == weight_name;
      if (tensor.shape()
        != (weight ? Shape{layer.out_, layer.in_} : Shape{layer.out_}))
        throw exception(
          "ColumnParallelLinear: checkpoint shape mismatch for " + name);
      auto count = tensor.shape()[0] / layer.group_.world_size();
      return WeightShard::shard(
        tensor, 0, layer.group_.rank_id() * count, count);
    }
  };
  int64_t      in_, out_, local_in_, local_out_;
  ProcessGroup group_;
  Parameter   *weight_ = nullptr;
  Parameter   *bias_   = nullptr;
};

// Head-aware Q/K/V projections with separate checkpoint tensors. Q may contain
// several packed components per head (e.g. Qwen3.5's [query, gate]).
class QKVParallelLinear : public Module {
public:
  static int64_t local_kv_heads(
    int64_t q_heads, int64_t kv_heads, const ProcessGroup &group) {
    const auto size = group.world_size();
    if (q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads || q_heads % size
      || (kv_heads >= size ? kv_heads % size : size % kv_heads))
      throw exception("QKVParallelLinear: incompatible Q/KV heads and TP size");
    return kv_heads >= size ? kv_heads / size : 1;
  }

  QKVParallelLinear(int64_t in_features, int64_t q_heads, int64_t kv_heads,
    int64_t head_dim, ProcessGroup group, bool bias = true,
    DType dtype = DType::FP32, int64_t q_components = 1)
      : in_(in_features), q_heads_(q_heads), kv_heads_(kv_heads),
        head_dim_(head_dim), q_components_(q_components), group_(group) {
    local_kv_ = local_kv_heads(q_heads, kv_heads, group);
    if (in_ <= 0 || head_dim_ <= 0 || q_components_ <= 0)
      throw exception("QKVParallelLinear: feature sizes must be positive");
    local_q_  = q_heads_ / group_.world_size();
    kv_start_ = kv_heads_ >= group_.world_size()
      ? group_.rank_id() * local_kv_
      : group_.rank_id() / (group_.world_size() / kv_heads_);
    q_proj    = register_loadable(
      "q_proj", Linear(in_, local_q_ * head_dim_ * q_components_, bias, dtype));
    k_proj = register_loadable(
      "k_proj", Linear(in_, local_kv_ * head_dim_, bias, dtype));
    v_proj = register_loadable(
      "v_proj", Linear(in_, local_kv_ * head_dim_, bias, dtype));
  }

  int64_t local_query_heads() const {
    return local_q_;
  }
  int64_t local_key_value_heads() const {
    return local_kv_;
  }
  int64_t key_value_head_start() const {
    return kv_start_;
  }

  std::array<Tensor, 3> forward(const Tensor &input) const {
    if (input.dim() < 2 || input.shape().back() != in_)
      throw exception("QKVParallelLinear: input must end in in_features");
    return {
      q_proj->forward(input), k_proj->forward(input), v_proj->forward(input)};
  }

  void load(Loader *loader, const std::string &prefix = "") override {
    if (!loader)
      throw exception("QKVParallelLinear: null loader");
    ShardLoader shards(loader, *this, prefix);
    Module::load(&shards, prefix);
  }

  Linear *q_proj, *k_proj, *v_proj;

private:
  struct ShardLoader : Loader {
    Loader                  *source;
    const QKVParallelLinear &layer;
    std::string              prefix;
    ShardLoader(
      Loader *source, const QKVParallelLinear &layer, const std::string &prefix)
        : source(source), layer(layer),
          prefix(prefix.empty() ? "" : prefix + ".") {}
    Tensor load(const std::string &name) override {
      const auto suffix = name.substr(prefix.size());
      const bool q      = suffix == "q_proj.weight" || suffix == "q_proj.bias";
      const bool weight = suffix.ends_with(".weight");
      auto       tensor = source->load(name);
      const auto width  = layer.head_dim_ * (q ? layer.q_components_ : 1);
      const auto heads  = q ? layer.q_heads_ : layer.kv_heads_;
      if (tensor.shape()
        != (weight ? Shape{heads * width, layer.in_} : Shape{heads * width}))
        throw exception(
          "QKVParallelLinear: checkpoint shape mismatch for " + name);
      const auto count = q ? layer.local_q_ : layer.local_kv_;
      const auto start = q ? layer.group_.rank_id() * count : layer.kv_start_;
      return WeightShard::shard(tensor, 0, start * width, count * width);
    }
  };
  int64_t      in_, q_heads_, kv_heads_, head_dim_, q_components_;
  int64_t      local_q_, local_kv_, kv_start_;
  ProcessGroup group_;
};

// Consumes input shards and sums their contributions into a full output.
class RowParallelLinear : public Module {
public:
  RowParallelLinear(int64_t in_features, int64_t out_features,
    ProcessGroup group, bool bias = true, DType dtype = DType::FP32)
      : in_(in_features), out_(out_features), group_(group) {
    if (in_ <= 0 || out_ <= 0)
      throw exception("RowParallelLinear: feature sizes must be positive");
    if (in_ % group_.world_size())
      throw exception(
        "RowParallelLinear: input features must divide world_size");
    local_in_  = in_ / group_.world_size();
    local_out_ = out_;
    weight_ =
      register_loadable("weight", Parameter({local_out_, local_in_}, dtype));
    if (bias)
      bias_ = register_loadable("bias", Parameter({local_out_}, dtype));
  }
  int64_t in_features() const {
    return in_;
  }
  int64_t out_features() const {
    return out_;
  }
  int64_t local_in_features() const {
    return local_in_;
  }
  int64_t local_out_features() const {
    return local_out_;
  }
  const ProcessGroup &group() const {
    return group_;
  }
  const Tensor &weight() const {
    return weight_->get();
  }

  void load(Loader *loader, const std::string &prefix = "") override {
    if (!loader)
      throw exception("RowParallelLinear: null loader");
    ShardLoader shards(loader, *this, prefix);
    Module::load(&shards, prefix);
  }
  Tensor forward(const Tensor &input) const {
    if (input.dim() < 2 || input.shape().back() != local_in_)
      throw exception("RowParallelLinear: input must end in local_in_features");
    auto output = input.dim() > 2
      ? op::batched_mat_mul(false, true)(input, weight_->get())
      : op::mat_mul(false, true)(input, weight_->get());
    if (group_.world_size() > 1)
      output = op::all_reduce(output, group_);
    // Replicated bias must be added after reduction, exactly once.
    if (bias_)
      output = op::add(output, bias_->get());
    return output;
  }

private:
  struct ShardLoader : Loader {
    Loader                  *source;
    const RowParallelLinear &layer;
    std::string              weight_name;
    ShardLoader(Loader *loader, const RowParallelLinear &linear,
      const std::string &prefix)
        : source(loader), layer(linear),
          weight_name(prefix.empty() ? "weight" : prefix + ".weight") {}
    Tensor load(const std::string &name) override {
      auto       tensor = source->load(name);
      const bool weight = name == weight_name;
      if (tensor.shape()
        != (weight ? Shape{layer.out_, layer.in_} : Shape{layer.out_}))
        throw exception(
          "RowParallelLinear: checkpoint shape mismatch for " + name);
      if (weight)
        return WeightShard::shard(
          tensor, 1, layer.group_.rank_id() * layer.local_in_, layer.local_in_);
      return WeightShard::shard(tensor, 0, 0, layer.out_);
    }
  };
  int64_t      in_, out_, local_in_, local_out_;
  ProcessGroup group_;
  Parameter   *weight_ = nullptr;
  Parameter   *bias_   = nullptr;
};
} // namespace forch::nn
