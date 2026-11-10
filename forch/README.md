# 环境配置

建议参考华为官方文档，用 conda/mamba 创建 cann 9.0.0 环境。

加载环境：

```bash
micromamba activate forch
source $CONDA_PREFIX/Ascend/cann/set_env.sh
```

# 代码组织

+ include 包含头文件，在保证不引入 backend 相关实现的情况下可以写一些 inline function
+ src 包含实现，src/{backend-name} 包含各个后端的实现。各个后端作为单独的编译单元动态链接。
+ example 包含使用用例，几乎全是 ai 生成。建议都使用 ai 生成，因为如果一个功能 ai 用不好那一定是设计的有问题。
+ src/{backend-name}/tests 包含后端相关的测试，也都是 ai 写的。

# 开发入口

+ [AGENTS.md](AGENTS.md)：工程约束、构建和测试命令，以及按任务读取的 skill 索引。
+ `AGENTS.local.md`：本机环境与资源配置，不纳入版本控制；上面的 `forch` 是当前开发环境名。
+ [forch-ge 算子开发](.agents/skills/forch-ge-operator/SKILL.md)：公共接口、GE 映射、注册、测试与验证流程，供开发者和 agents 共用。

# 设计文档

讲某一块"为什么这么分"，不重复上面的操作流程。

+ [Device 与设备内存](docs/device.md)：逻辑卡绑定、Buffer 所有权、GE 析构后的 Tensor 读回与测试入口。
+ [多卡通信层](docs/collectives.md)：集合通信与 matmul 融合算子的分层、进程初始化的环境契约、多卡启动器。
