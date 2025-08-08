<div align="center" id="HWInfer-top">
<!-- TODO: 请替换为您的项目 Logo -->
<img src="./assets/logo.svg" alt="logo" width="400" margin="10px"></img>
</div>

--------------------------------------------------------------------------------

![til](./assets/demo.gif)

## 📰 最新动态
- **[2025.08.08]** 🔥 `JittorInfer` v0.1.0 发布！

## 📖 关于
`JittorInfer` 是一个在华为昇腾（Ascend）AI处理器上，为大型语言模型（如 DeepSeek）设计的高性能 C++ 推理框架。它的目标是提供极致的推理速度和效率。

核心功能包括:
- **高性能后端**: 采用先进的并行计算技术和昇腾CANN的Graph Engine技术，以实现最快的推理速度。
- **易于使用**: 提供简洁的运行方法，方便直接部署为可调用的服务。
- **可扩展性**: 方便添加对新模型的支持。

## 📦 环境依赖

- CMake (推荐版本 >= 3.22)
- GCC/G++ (推荐版本 >= 10.3.1)
- 华为 [CANN](https://www.hiascend.com/developer/download/community/result?module=cann) 工具包 (推荐版本 >= 8.2.RC1.alpha001)

## 🛠️ 开始使用
### [llama-server 使用指南](./examples/README.md)

llama-server 是一个高性能推理服务端，针对昇腾环境进行了特殊优化，提供与 OpenAI API 兼容的接口。服务器支持多种大语言模型，如 DeepSeek-V2-Lite 等，并提供文本生成和聊天功能。

详细编译与使用教程请参考[此处](./examples/README.md)。

## 🚀 性能测试
- 单卡DeepSeek V2 Lite 测试结果

    | 并发数 | vLLM-Ascend (v0.7.3) | vLLM-Ascend (v0.9.1) | Ours (8-6) | 加速比   |
    |:-------:|:---------------------:|:---------------------:|:-----------:|:---------:|
    | 1       | 12.4                  | 12.2                  | 73.3        | 500.8%    |
    | 2       | 22.7                  | 22.4                  | 114.43      | 410.8%    |
    | 4       | 43.5                  | 40.4                  | 166.82      | 312.9%    |
    | 8       | 87.7                  | 82.5                  | 249.28      | 202.2%    |

- 单机八卡DeepSeek V2 测试结果

    | 并发数 | vLLM-Ascend (v0.7.3) | Ours (8-6) | 加速比 |
    |:-------:|:---------------------:|:-----------:|:-------:|
    | 1       | 4.3                   | 9.3         | 116.2%   |
    | 2       | 7.83                  | 15.37       | 96.3%   |
    | 4       | 15.62                 | 25.80       | 65.2%   |
    | 8       | 26.3                  | 45.69       | 73.7%   |

## 📢 交流
- 欢迎加入JittorInfer技术交流群，请扫描下方二维码.

<div align="center">
<img src="./assets/qrcode.png" alt="qrcode" width="200" margin="10px"></img>
</div>


## 💖 致谢
`JittorInfer` 的开发借鉴了以下优秀开源项目的思想和代码：[llama.cpp](https://github.com/ggml-org/llama.cpp)，[ggml](https://github.com/ggml-org/ggml)，[torchair](https://github.com/Ascend/torchair)

