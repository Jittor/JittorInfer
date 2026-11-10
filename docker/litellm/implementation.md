# finer + LiteLLM Compose 实现说明

部署操作见 [部署指南](README.md)。本文说明 Compose 的服务划分、配置来源和启动依赖。

## 服务与网络

| 服务 | 职责 | 配置来源 |
| --- | --- | --- |
| `finer` | Rust 前端提供 OpenAI 兼容 API，C++ 后端执行推理 | `config/finer.json` |
| `litellm` | 网关、模型路由和管理页面 | 环境变量与 PostgreSQL |
| `db` | PostgreSQL 16，保存 LiteLLM 数据 | 环境变量与 `postgres_data` 卷 |

请求路径为 `客户端 → LiteLLM:4000 → finer:8000/v1`。
只有 LiteLLM 发布宿主端口，finer 和 PostgreSQL 通过 Compose 内部网络访问。
LiteLLM 的模型路由使用 `hosted_vllm/` provider，目标地址为 `http://finer:8000/v1`。

## 配置与持久化

- `.env` 提供镜像、设备、宿主模型路径、监听地址和凭据。
- `config/finer.json` 只配置 finer，以只读方式挂载到 `/config/finer.json`。
- LiteLLM 不挂载配置文件，也不使用 `--config`；模型由启动后的 API 请求创建。
  `STORE_MODEL_IN_DB=True` 与 `DATABASE_URL` 使模型配置保存在 PostgreSQL 中。
- PostgreSQL 使用命名卷 `postgres_data`。恢复部署需要保留数据库及 `.env` 中的凭据，
  包括用于解密已存凭据的 `LITELLM_SALT_KEY`。

## Ascend 运行环境

finer 使用 ARM64 镜像，由镜像 ENTRYPOINT 加载 CANN 和 HPCKit 环境，
再执行 `script/run-server.sh /config/finer.json` 启动前后端。

Compose 映射选定的 `/dev/davinciN` 及三个管理设备，按宿主原路径只读挂载驱动文件，
并设置 `shm_size: 1g`。模型目录只读挂载到 `/model`，宿主缓存目录挂载到 `/root/.cache`。
当前配置为单卡；镜像及多卡运行说明见 [Docker Quick Start](../../docs/docker.md)。

## 启动与停止

LiteLLM 依赖 finer 的 `/health` 和 PostgreSQL 的 `pg_isready` 健康检查。
finer 首次启动需编译计算图，健康检查的 `start_period` 为 30 分钟，
与 `server.engine_ready_timeout_secs=1800` 对应。修改启动等待时间时应同步调整两处。
finer 的停止宽限期为 150 秒。

## 参考

- [LiteLLM Docker Quick Start](https://docs.litellm.ai/docs/proxy/docker_quick_start)
  与 [Compose 示例](https://github.com/BerriAI/litellm/blob/main/docker/docker-compose.quickstart.yml)
- [vLLM Ascend Quick Start](https://docs.vllm.ai/projects/ascend/en/latest/getting_started/quick_start.html)
- [LiteLLM vLLM provider](https://docs.litellm.ai/docs/providers/vllm)
- [LiteLLM 模型管理 API](https://docs.litellm.ai/docs/proxy/ui_store_model_db_setting)
