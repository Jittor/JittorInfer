# finer + LiteLLM Compose 部署指南

在 Ascend 主机上部署 finer、LiteLLM 和 PostgreSQL，提供 OpenAI 兼容 API。
默认使用单张 Ascend 910B 加载 Qwen3-4B。Compose 的结构和配置设计见
[实现说明](implementation.md)。

## 1. 准备环境

主机需要 aarch64 Linux、Docker Compose v2、可用的 Ascend 910B 驱动、
一张空闲 NPU，以及完整的本地模型文件。使用预构建镜像，无须在宿主机编译。

以下命令在本目录的 Bash 中执行。首次部署创建 `.env`，已有该文件时直接编辑：

```bash
cp .env.example .env
chmod 600 .env
mkdir -p "$HOME/.cache"
```

### 部署环境变量

在与 `compose.yaml` 同目录的 `.env` 中按 `变量名=值` 填写，无须执行 `source .env`。
Compose 会自动读取该文件；已导出的同名 shell 环境变量会覆盖 `.env` 的值。
规则见 [Docker Compose 环境变量说明](https://docs.docker.com/compose/how-tos/environment-variables/variable-interpolation/)。

| 变量 | 默认值 / 是否必填 | 用途与设置方式 |
| --- | --- | --- |
| `MODEL_DIR` | 必填 | 直接包含模型配置、tokenizer 和权重的宿主绝对路径；替换示例中的占位路径 |
| `DEVICE` | `/dev/davinci0` | 选择空闲 NPU 的设备路径，例如 `/dev/davinci3` |
| `FINER_IMAGE` | `ghcr.io/huawei-llm-inference-framework/finer:0.2.0-beta-2026.11.02` | finer 镜像，可替换为所需版本 |
| `LITELLM_IMAGE` | `docker.litellm.ai/berriai/litellm:latest` | LiteLLM 镜像，可指定 tag 或 digest |
| `LITELLM_BIND` | `127.0.0.1` | 宿主监听地址；`0.0.0.0` 表示监听所有 IPv4 接口 |
| `LITELLM_PORT` | `4000` | 宿主端口，例如 `14000`；修改后同步调整浏览器、客户端和 SSH 转发使用的端口 |
| `LITELLM_MASTER_KEY` | 必填 | 管理 API 的 Bearer token，也是默认 `admin` 登录密码 |
| `LITELLM_SALT_KEY` | 必填 | 加密数据库中保存的凭据，首次生成后保持不变 |
| `POSTGRES_PASSWORD` | 必填 | PostgreSQL 密码；使用下方命令生成十六进制值，避免数据库连接 URL 中的特殊字符 |

例如，使用 NPU 3 并将网关映射到本机 `14000` 端口：

```dotenv
MODEL_DIR=/data/models/Qwen3-4B
DEVICE=/dev/davinci3
LITELLM_BIND=127.0.0.1
LITELLM_PORT=14000
```

运行以下命令，将输出填入 `.env` 对应字段：

```bash
printf 'LITELLM_MASTER_KEY=sk-%s\n' "$(openssl rand -hex 32)"
printf 'LITELLM_SALT_KEY=sk-%s\n' "$(openssl rand -hex 32)"
printf 'POSTGRES_PASSWORD=%s\n' "$(openssl rand -hex 24)"
```

妥善保存 `.env`，已有数据库使用的 salt 和密码应保持不变。
仅修改 `.env` 中的 `POSTGRES_PASSWORD` 不会更改已有数据库内的密码。
同一主机部署多套服务时，可在各自 `.env` 中另设 `COMPOSE_PROJECT_NAME`，
并使用不同的设备和端口；后续管理同一套服务时保持项目名不变。
如需调整模型名称、上下文长度或批处理参数，编辑 `config/finer.json`。

### Compose 已设置的容器变量

以下变量已由 `compose.yaml` 提供，无须重复填入 `.env`：

| 变量 | 当前设置 |
| --- | --- |
| `STORE_MODEL_IN_DB` | 固定为 `"True"`，启用模型配置入库；UI 显示差异见文末 |
| `DATABASE_URL` | 根据 `POSTGRES_PASSWORD` 生成，连接内部 `db:5432/litellm` |
| `POSTGRES_USER`、`POSTGRES_DB` | 均为 `litellm` |

`.env` 中的变量不会自动全部传入容器。新增 LiteLLM 环境变量时，还需在
`compose.yaml` 的 `litellm.environment` 中声明；例如，只在 `.env` 添加
`UI_USERNAME` 或 `UI_PASSWORD` 不会改变当前登录凭据。
缓存目录使用宿主当前用户的 `$HOME/.cache`，无须修改 `HOME`。

## 2. 启动服务

```bash
docker compose config --quiet
docker compose pull
docker compose up -d
```

首次启动需要加载模型并编译计算图，请等待服务就绪。可查看状态和日志：

```bash
docker compose ps
docker compose logs -f finer litellm
```

## 3. 登录管理页面

打开 `http://127.0.0.1:4000/ui/`，使用以下凭据登录：

- 用户名：`admin`
- 密码：部署目录 `.env` 中的 `LITELLM_MASTER_KEY` 值，包含 `sk-` 前缀。

监听端口按 `.env` 调整；远程访问方法见下文。页面提示
`Environment-credential login is enabled` 表示当前允许使用环境变量凭据登录，
不影响模型管理。如需个人账号，可在 **Internal Users → Invite User** 创建管理用户，通过邀请链接设置密码。详见
[LiteLLM 管理页面说明](https://docs.litellm.ai/docs/proxy/ui)。

## 4. 注册和管理模型

LiteLLM 启动后通过 API 或管理页面注册模型，配置保存在数据库中。
两种方式操作同一份数据，选择一种即可，无须重复添加。
以下命令复用同一 Bash 会话。Compose 读取 `.env` 不会把变量载入当前 shell，
调用 API 前还需设置 `LITELLM_URL`，并输入 `.env` 中的 `LITELLM_MASTER_KEY`。
`LITELLM_URL` 是调用端访问网关的地址，仅供下方 curl 命令使用，不是容器配置。
例如本地转发端口为 `14000` 时，应设置为 `http://127.0.0.1:14000`。

```bash
LITELLM_URL=http://127.0.0.1:4000
read -r -s -p 'LiteLLM master key: ' LITELLM_MASTER_KEY
printf '\n'

curl --noproxy '*' --fail-with-body -sS "$LITELLM_URL/model/new" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model_name": "qwen3-4b",
    "litellm_params": {
      "model": "hosted_vllm/qwen3-4b",
      "api_base": "http://finer:8000/v1"
    }
  }'
```

`model_name` 是客户端使用的别名，`hosted_vllm/` 后的名称应与
`config/finer.json` 的 `server.served_model_name` 一致。
保留数据库数据卷时，重启后无须重复注册。

在 UI 中，进入 **Models + Endpoints → Add Model**，选择 vLLM provider，
按上述请求填写模型参数并保存：

| 参数 | 值 |
| --- | --- |
| 对外模型名称（`model_name`） | `qwen3-4b` |
| LiteLLM 模型标识（`litellm_params.model`） | `hosted_vllm/qwen3-4b` |
| API Base | `http://finer:8000/v1` |

如果表单会自动添加 provider 前缀，模型名称只填 `qwen3-4b`；保存后的完整标识应为
`hosted_vllm/qwen3-4b`。该 finer 服务未设置上游 API key，不需要填写。
这里的 API Base 是 LiteLLM 容器访问 finer 的地址，不是浏览器访问管理页面的地址。

常用 API 与 UI 操作对应如下，URL 均相对于 LiteLLM 网关地址：

| 操作 | HTTP API | 管理页面操作 |
| --- | --- | --- |
| 添加模型 | `POST /model/new` | **Models + Endpoints → Add Model** |
| 查看模型配置 | `GET /model/info` | **Models + Endpoints → All Models**，点击 Model ID；在 **Raw JSON** 查看完整定义 |
| 修改模型 | `PATCH /model/{model_id}/update` | 模型详情页 → **Edit Settings** → 保存 |
| 删除模型 | `POST /model/delete`，请求体为 `{"id":"<model_id>"}` | 模型详情页 → **Delete Model** |
| 发送对话 | `POST /v1/chat/completions` | **Playground**，选择模型并发送消息 |
| 查看入库开关 | `GET /config/list?config_type=general_settings` | **Models + Endpoints** 的设置中查看 **Store Model in DB** |
| 修改入库开关 | `POST /config/field/update` | 同一设置中修改 **Store Model in DB** 并保存 |

模型管理接口详见 [LiteLLM 模型管理说明](https://docs.litellm.ai/docs/proxy/ui_store_model_db_setting)。

## 5. 使用服务

OpenAI API base URL 为 `http://127.0.0.1:4000/v1`，管理页面为
`http://127.0.0.1:4000/ui`。调用示例：

```bash
curl --noproxy '*' --fail-with-body -sS "$LITELLM_URL/v1/chat/completions" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-4b",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 128
  }'
```

默认仅允许部署主机本地访问。从其他机器访问时，可建立 SSH 转发：

```bash
ssh -N -L 14000:127.0.0.1:4000 user@ascend-host
```

替换 SSH 用户和主机后，通过本机 `http://127.0.0.1:14000` 访问。
如需直接对外提供服务，在 `.env` 中设置 `LITELLM_BIND` 为主机监听地址。

## 6. 添加用户与额度控制

### 创建账号

1. 管理员进入 **Internal Users → Invite User**，填写用户邮箱并选择角色。
2. 创建后复制页面生成的邀请链接，由用户打开并按提示设置密码。
   当前 Compose 未配置邮件服务，可直接分享链接；用户需要能访问链接中的网关地址。
3. 用户使用自己的邮箱和密码登录 `/ui/`。需要重设密码时，管理员在用户行的操作菜单中选择
   **Reset password**，将生成的链接交给用户。

| 角色 | 用途 |
| --- | --- |
| `internal_user` | 普通使用者，可管理自己的密钥并查看自己的用量 |
| `internal_user_viewer` | 查看自己的密钥和用量，不可创建或删除密钥 |
| `proxy_admin` | 管理模型、用户和平台配置 |
| `proxy_admin_viewer` | 查看全局密钥和用量，不可修改 |

普通使用者选择 `internal_user`，需要维护部署的人员才授予 `proxy_admin`。
账号保存在数据库中，无须为每个用户设置 `UI_USERNAME` 或 `UI_PASSWORD`。
角色与邀请流程见 [LiteLLM 用户管理](https://docs.litellm.ai/docs/proxy/self_serve)。

### 设置额度和限流

在 **Internal Users** 中编辑用户可设置个人预算；在 **Teams** 中创建或编辑团队可设置
共享预算并添加成员。应用调用使用 **Virtual Keys / Keys → Create New Key** 创建的虚拟密钥，
选择所属用户、团队（如有）和允许访问的模型，再填写额度及限流参数。
管理员代建密钥时应明确指定所属用户，以便归属用量。详见
[LiteLLM 虚拟密钥](https://docs.litellm.ai/docs/proxy/virtual_keys)。

| 设置 | 含义 | 示例 |
| --- | --- | --- |
| Models | 允许调用的模型别名，大小写应与模型列表一致 | `qwen3-4b` |
| Max Budget（`max_budget`） | 以美元计的费用额度，不是 token 总数 | `10` |
| Budget Duration（`budget_duration`） | 额度重置周期 | `30d`：每 30 天重置，并非每月 1 日 |
| RPM（`rpm_limit`） | 每分钟请求数上限 | `10` |
| TPM（`tpm_limit`） | 每分钟 token 数上限 | `20000` |
| Key Duration（`duration`） | 密钥有效期，与额度重置周期独立 | `90d` |

例如，为个人应用创建只允许 `qwen3-4b` 的密钥，设置 10 美元 / 30 天、10 RPM、
20000 TPM、90 天有效期。调用时将前文 curl 中的 master key 替换为该虚拟密钥。
这些数值只是填写示例，应按实际容量和使用需求调整。

个人预算用于未关联团队的个人密钥；团队密钥使用团队预算，不能依赖用户的个人预算
限制团队消费。需要限制团队内单个成员时，在团队成员设置中配置成员预算。
密钥还可设置自己的额度。超出适用的预算或速率限制后，请求会被拒绝。
进入 **Usage** 或用户、团队、密钥详情查看累计用量、费用和预算重置时间。
详见 [LiteLLM 预算与限流](https://docs.litellm.ai/docs/proxy/users)。

### 自托管模型的费用统计

finer 的模型需要有正确的输入、输出 token 单价，金额预算才有意义。
在 **Models + Endpoints → 模型详情 → Edit Settings** 中设置输入和输出价格，
注意 UI 的单价单位为每百万 token；通过 API 配置时应按对应字段要求换算。
若模型未配置或未匹配到有效价格，不能仅凭请求成功就认为费用统计正确；
价格为零时，金额预算也无法限制其使用量。可使用 RPM/TPM 控制调用速率，
并在 **Usage / Logs** 中检查实际 token 数和费用。详见
[LiteLLM 自定义价格](https://docs.litellm.ai/docs/proxy/custom_pricing)。

| 用户管理操作 | HTTP API | 对应 UI 操作 |
| --- | --- | --- |
| 创建用户 | `POST /user/new` | **Internal Users → Invite User / New User** |
| 生成邀请链接 | `POST /invitation/new` | 创建用户后复制邀请链接 |
| 修改用户额度 | `POST /user/update` | **Internal Users → 编辑用户** |
| 创建或修改团队 | `POST /team/new`、`POST /team/update` | **Teams → 创建或编辑团队** |
| 创建或修改密钥 | `POST /key/generate`、`POST /key/update` | **Virtual Keys / Keys → 创建或编辑密钥** |

## 7. 更新和停止

修改 `.env` 或 Compose 配置后执行 `docker compose up -d`；
`docker compose restart` 不会加载新的容器环境变量。
修改 `config/finer.json` 后执行 `docker compose up -d --force-recreate finer`。
更换模型时，还需通过 [LiteLLM 模型管理 API](https://docs.litellm.ai/docs/proxy/ui_store_model_db_setting)
更新模型路由。

停止并移除容器和网络：

```bash
docker compose down
```

该命令保留数据库数据卷。`docker compose down -v` 会删除数据库数据。
备份或迁移时，同时保留数据卷和 `.env`。

## 已知显示问题：Store Model in DB

在本次使用的 LiteLLM 镜像中，仅通过环境变量设置 `STORE_MODEL_IN_DB=True` 时，
管理页面仍可能显示 **Store Model in DB = false**。其设置接口返回
`field_value: null`、`source: unset`，未反映环境变量的生效值；
这不代表模型入库功能关闭，通过 API 创建的模型仍会保存到 PostgreSQL。

在 UI 中将开关保存为 `true` 后，该设置会写入数据库，页面随之显示 `true`；
删除数据卷后，这项设置也会清除。部署无需为修正显示而额外修改开关，
也不需要把模型改为配置文件管理。
