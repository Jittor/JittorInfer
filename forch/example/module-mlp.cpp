#include "forch/forch.hpp"
#include "forch/nn/module.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/function.hpp"
#include "forch/ops/input.hpp"

#include "forch/huggingface/pretrained.hpp"
#include "forch/nn/linear.hpp"

#include "forch/backend/forch-ge.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

using namespace forch;

// clang-format off
/*
class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
*/
// clang-format on

struct MyModelConfig {
  int64_t hidden_size;
  int64_t intermediate_size;
  int64_t num_layers;
  DType   dtype = DType::FP32;
};

struct MyModelMLP : public nn::Module {
  nn::Linear *gate_proj;
  nn::Linear *up_proj;
  nn::Linear *down_proj;
  DType       dtype;

  MyModelMLP(const MyModelConfig &config) {
    dtype     = config.dtype;
    gate_proj = register_loadable("gate_proj",
      nn::Linear(config.hidden_size, config.intermediate_size, false, dtype));
    up_proj   = register_loadable("up_proj",
        nn::Linear(config.hidden_size, config.intermediate_size, false, dtype));
    down_proj = register_loadable("down_proj",
      nn::Linear(config.intermediate_size, config.hidden_size, false, dtype));
  }

  Tensor forward(const Tensor &input) const {
    auto down_proj = this->down_proj->forward(
      op::mul(op::silu(this->gate_proj->forward(input)),
        this->up_proj->forward(input)));
    return down_proj;
  }
};

struct MyModelLayer : public nn::Module {
  MyModelMLP *mlp;

  MyModelLayer(const MyModelConfig &config) {
    mlp = register_loadable("mlp", MyModelMLP(config));
  }
};

struct MyModel : public nn::Module {
  nn::ModuleList<MyModelLayer> *layers;

  MyModel(const MyModelConfig &config) {
    std::vector<std::unique_ptr<MyModelLayer>> layer_list;
    layer_list.reserve(config.num_layers);
    for (int i = 0; i < config.num_layers; ++i) {
      layer_list.push_back(std::make_unique<MyModelLayer>(config));
    }
    layers = register_loadable(
      "layers", nn::ModuleList<MyModelLayer>(std::move(layer_list)));
  }

  Tensor forward(const Tensor &input) const {
    auto x = input;
    for (const auto &layer : *layers) {
      x = layer->mlp->forward(x);
    }
    return x;
  }
};

int main(int argc, char **argv) {
  if (argc < 5) {
    logger()->error("Usage: module-mlp <model_path> <module_prefix> "
                    "<hidden_size> <intermediate_size> [batch=1]");
    return 1;
  }

  std::string model_path        = argv[1];
  std::string module_prefix     = argv[2];
  int64_t     hidden_size       = std::atoll(argv[3]);
  int64_t     intermediate_size = std::atoll(argv[4]);
  int64_t     batch             = (argc > 5) ? std::atoll(argv[5]) : 1;

  auto config = MyModelConfig{
    .hidden_size       = hidden_size,
    .intermediate_size = intermediate_size,
    .num_layers        = 4,
    .dtype             = DType::BF16,
  };
  auto model  = std::make_unique<MyModel>(config);
  auto loader = huggingface::Pretrained(model_path);
  model->load(&loader, module_prefix);

  auto input =
    op::create_input({(int64_t)batch, (int64_t)config.hidden_size}, DType::FP32)
      .set_name("input");
  auto output = model->forward(input).set_name("output");

  Backend      backend = GEBackend(AscendDevice::from_env());
  forch::Graph graph_ir({output});
  graph_ir.set_name("ge-mlp");
  auto graph = backend.build(graph_ir);

  graph.compile();
  graph.load();

  std::vector<float> input_data(
    static_cast<size_t>(config.hidden_size * batch));
  for (size_t i = 0; i < input_data.size(); ++i) {
    // input_data[i] = static_cast<float>(i % 100) / 100.0F;
    input_data[i] =
      static_cast<float>(i / config.hidden_size) + 1.0F; // batch index + 1
  }
  auto input_tensor =
    op::load_from_buffer(reinterpret_cast<Buffer>(input_data.data()),
      {(int64_t)config.hidden_size, (int64_t)batch}, DType::FP32)
      .set_name("input");

  auto outputs = graph.run({input_tensor});

  const auto *out_ptr = reinterpret_cast<const float *>(outputs[0].get_data());
  int64_t     out_len = config.hidden_size * batch;
  std::cout << "Output (first 8 values):" << std::endl;

  for (int64_t bi = 0; bi < batch; bi++) {
    for (int64_t i = 0; i < std::min<int64_t>(out_len, 8); ++i) {
      std::cout << out_ptr[bi * config.hidden_size + i]
                << (i + 1 == std::min<int64_t>(out_len, 8) ? '\n' : ' ');
    }
  }

  return 0;
}
