
- 转换脚本（只支持Qwen3，更多模型参考：https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py）
```bash
python convert_qwen3_to_gguf.py /root/data/qwen3-8b --outtype f16 --outfile /root/data/qwen3-8b-merged-fp16.gguf --outtype f16
```

- 合并qkv的转换脚本
```bash
python convert_qwen3_to_gguf.py /root/data/qwen3-8b --outtype f16 --outfile /root/data/qwen3-8b-merged-fp16.gguf --outtype f16 --merge-qkv --merge-ffn
```