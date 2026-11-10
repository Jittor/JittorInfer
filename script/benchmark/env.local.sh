# 机器相关配置。换机器只改这个文件。
# bench-performance.sh、bench-correctness.sh、install-opencompass.sh 会 source 它。
#
# 第一次使用先建环境，再装 OpenCompass：
#   conda create -n opencompass python=3.10 pip
#   bash script/benchmark/install-opencompass.sh

# conda activate 会读到未设置的变量，source 期间先关掉 nounset
case $- in *u*) _bench_nounset=1; set +u ;; *) _bench_nounset=0 ;; esac

# OpenCompass 装在这个环境里。
CONDA_ENV=opencompass
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
# conda activate 不会把 $CONDA_PREFIX/lib 放进 LD_LIBRARY_PATH。
# 不设的话，nltk → sqlite3 会加载系统里过旧的 libstdc++，评测准确率为空。
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# 已启动 server 的地址，以及 /v1/models 里的模型 id。
export BENCH_URL=http://127.0.0.1:8000
export MODEL_NAME=qwen3-0.6b

# inference-benchmarker 的 tokenizer，必须是 HF repo id（tokenizers 只走 hub）。
export TOKENIZER_NAME=Qwen/Qwen3-0.6B

# OpenCompass 截断计数用的 tokenizer，本地模型目录。
export TOKENIZER_PATH=/beegfs/model/Qwen/Qwen3-0.6B

# OpenCompass 的 max_seq_len，必须 ≤ server 的 n_context（换模型/换 server 配置时一起改）。
# gsm8k 2048~4096 够用；MRCR（≤32k 档）需要 34816。
export MAX_SEQ_LEN=8192

# 数据集可以在本地缓存（gsm8k 走 OpenCompass OSS，MRCR 三个 bin 在 HF cache）。
# 默认离线：load_dataset 不发请求直接命中缓存，省掉断网时每个数据集 ~23s 的重试。
# 首次下载新数据集时临时 HF_HUB_OFFLINE=0（或注释掉这两行）跑一次。
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

# 正确性评测的 endpoint，一行一个：简称|chat/completions 完整地址|模型 id
# 多行 = 一次跑多个 server，汇总表一列一个。
export BENCH_ENDPOINTS="jittorinfer|${BENCH_URL}/v1/chat/completions|${MODEL_NAME}"

if [ "$_bench_nounset" = 1 ]; then set -u; fi
unset _bench_nounset
