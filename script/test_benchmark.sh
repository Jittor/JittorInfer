#!/bin/bash

# 测试 localhost:8080 的推理性能测试脚本
# 基于 inference-benchmarker 工具

# 设置默认参数
#TOKENIZER_NAME="deepseek-ai/DeepSeek-V2-Lite"  # 默认使用 deepseek 分词器
TOKENIZER_NAME="deepseek-ai/DeepSeek-V2-Lite"
URL="http://localhost:8088"  # 目标服务器地址
BENCHMARK_KIND="throughput"  # 默认使用 throughput 模式
DURATION="90s"  # 每个测试步骤持续时间
WARMUP="15s"  # 预热时间
MAX_VUS=1  # 最大并发用户数

export HTTPS_PROXY=ddns.shadowhome.top:15999

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    -t|--tokenizer)
      TOKENIZER_NAME="$2"
      shift 2
      ;;
    -u|--url)
      URL="$2"
      shift 2
      ;;
    -b|--benchmark)
      BENCHMARK_KIND="$2"
      shift 2
      ;;
    -d|--duration)
      DURATION="$2"
      shift 2
      ;;
    -w|--warmup)
      WARMUP="$2"
      shift 2
      ;;
    -m|--max-vus)
      MAX_VUS="$2"
      shift 2
      ;;
    -h|--help)
      echo "用法: $0 [选项]"
      echo "选项:"
      echo "  -t, --tokenizer NAME    使用的分词器名称 (默认: gpt2)"
      echo "  -u, --url URL           测试的服务器URL (默认: http://localhost:8080)"
      echo "  -b, --benchmark TYPE    测试类型: throughput, sweep, optimum (默认: sweep)"
      echo "  -d, --duration TIME     每个测试步骤的持续时间 (默认: 60s)"
      echo "  -w, --warmup TIME       预热时间 (默认: 15s)"
      echo "  -m, --max-vus NUM       最大并发用户数 (默认: 64)"
      echo "  -h, --help              显示此帮助信息"
      exit 0
      ;;
    *)
      echo "未知选项: $1"
      exit 1
      ;;
  esac
done

echo "开始对 $URL 进行性能测试..."
echo "使用分词器: $TOKENIZER_NAME"
echo "测试类型: $BENCHMARK_KIND"
echo "持续时间: $DURATION"
echo "预热时间: $WARMUP"
echo "最大并发用户数: $MAX_VUS"
echo ""

# 运行测试
/root/zjp/inference-benchmarker/target/release/inference-benchmarker \
  --tokenizer-name "$TOKENIZER_NAME" \
  --url "$URL" \
  --benchmark-kind "$BENCHMARK_KIND" \
  --duration "$DURATION" \
  --warmup "$WARMUP" \
  --max-vus "$MAX_VUS" \
  --extra-meta "test_script=benchmark_test,server=localhost:8080" \
  --prompt-options "num_tokens=5,max_tokens=10,min_tokens=1,variance=10" \
  --decode-options "num_tokens=100,max_tokens=120,min_tokens=110,variance=10" \
  --model-name "/root/cache/modelscope/hub/models/deepseek-ai/DeepSeek-V2-Chat"
