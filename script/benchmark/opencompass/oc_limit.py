# 抽样数据集：给 eval_quick.py 快速验证链路用，固定 16 条。
# 抽样准确率噪声大，只用来确认链路通，不代表模型水平。
from datasets import DatasetDict
from opencompass.datasets import GSM8KDataset
from opencompass.registry import LOAD_DATASET

LIMIT = 16


@LOAD_DATASET.register_module()
class GSM8KLimitDataset(GSM8KDataset):
    """取数据集前 LIMIT 条（原类返回 DatasetDict{train,test}，评测用 test）"""

    @staticmethod
    def load(path):
        ds = GSM8KDataset.load(path)
        if LIMIT:
            ds = DatasetDict({k: v.select(range(min(LIMIT, len(v)))) for k, v in ds.items()})
        return ds
