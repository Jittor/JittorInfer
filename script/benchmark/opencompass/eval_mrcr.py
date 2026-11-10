# 长上下文正确性评测：OpenAI MRCR（多轮共指消解，"多针找针"）。
# 数据集 giulio98/MRCR_v2_common，needle_in_x_y 表示 prompt 长度落在 x~y token 之间。
#
# 跑法：bash script/benchmark/bench-correctness.sh mrcr
#
# 注意：
#   1. prompt 长度必须 ≤ 后端 n_context，否则被截断、准确率是假的。
#      默认配置 n_context=8192 连最小的 4096_8192 档都可能截断，
#      建议 32k 配置（MAX_SEQ_LEN 同步改成 32768）并只保留 ≤32768 的档。
#   2. 每个 bin 100 条 4k~32k 的 prompt，批量跑起来要几十分钟。

from opencompass.models import OpenAI
from opencompass.datasets.mrcr import MRCRDataset, MRCREvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_raw_prompt_template import RawPromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
ENDPOINTS = []
for raw in __import__('os').environ.get('BENCH_ENDPOINTS', '').splitlines():
    line = raw.strip()
    if not line or line.startswith('#'):
        continue
    name, api_base, model_name = (part.strip() for part in line.split('|', 2))
    ENDPOINTS.append((name, api_base, model_name))
TOKENIZER_PATH = __import__('os').environ.get('TOKENIZER_PATH', '')
if not TOKENIZER_PATH:
    raise RuntimeError('未设置 TOKENIZER_PATH，改 script/benchmark/env.local.sh')
MAX_SEQ_LEN = int(__import__('os').environ.get('MAX_SEQ_LEN', '34816'))   # ≤ 后端 n_context；可用环境变量 MAX_SEQ_LEN 覆盖
MAX_OUT_LEN = 1024
QUERY_PER_SECOND = 64            # OpenCompass 自己的限速（默认 1 QPS，会拖到几十分钟）
BATCH_SIZE = 64                  # 适当调大 利用continuous batching
# 长度档，可选：{2,4,8}needle_in_{4096_8192, 8192_16384, 16384_32768, 32768_65536, ...}
MRCR_SUBSETS = [
    '2needle_in_4096_8192',
    '2needle_in_8192_16384',
    '2needle_in_16384_32768',
]

mrcr_reader_cfg = dict(input_columns=['context', 'question'], output_column='gold')

mrcr_infer_cfg = dict(
    prompt_template=dict(
        type=RawPromptTemplate,
        messages=[{'role': 'user', 'content': '{context}\n{question}'}]),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

mrcr_eval_cfg = dict(evaluator=dict(type=MRCREvaluator))

datasets = []
for subset in MRCR_SUBSETS:
    datasets.append(dict(
        abbr=f'mrcr_v2_{subset}',
        type=MRCRDataset,
        path='giulio98/MRCR_v2_common',
        subset=subset,
        reader_cfg=mrcr_reader_cfg,
        infer_cfg=mrcr_infer_cfg,
        eval_cfg=mrcr_eval_cfg,
    ))

models = []
for name, api_base, model_name in ENDPOINTS:
    models.append(dict(
        type=OpenAI,
        path=model_name,
        key='EMPTY',                       # server 不校验 key，占位即可
        openai_api_base=api_base,
        max_seq_len=MAX_SEQ_LEN,
        query_per_second=QUERY_PER_SECOND,
        tokenizer_path=TOKENIZER_PATH,
        temperature=0,
        abbr=f'{name}-{model_name}',       # 汇总表里一列一个 endpoint
        max_out_len=MAX_OUT_LEN,
        batch_size=BATCH_SIZE,
        run_cfg=dict(num_gpus=0),
    ))
