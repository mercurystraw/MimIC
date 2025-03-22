import evaluate

import os
from datasets import load_dataset
from tqdm import tqdm

import src.paths as paths
from interface import DatasetBase
from testbed.data import postprocess_generation
from src.utils import get_expand_runname


class Dataset(DatasetBase):
    support_datasets = ["coco", "flickr"]

    def __init__(self, data_cfg):
        super().__init__(data_cfg)
        if self.name == "coco":
            ds_init_args = dict(
                path=os.path.join(paths.testbed_dir, "data", "coco"),
                data_dir=paths.karpathy_coco_caption_dir,
                images_dir=paths.coco_dir,
            )
        elif self.name == "flickr":
            ds_init_args = dict(
                path=os.path.join(paths.testbed_dir, "data", "flickr"),
                data_dir=paths.flickr30k_dir,
                images_dir=paths.flickr30k_images_dir,
                name="flickr30k",
            )

        dataset = load_dataset(**ds_init_args, trust_remote_code=True)
        self._support_set = dataset["train"]
        self._query_set = dataset["validation"]

    # 每个数据样本（一轮对话）中涉及的角色数量
    @property
    def num_role_in_round(self):
        # [
        #   { "role" : "image",
        #     "content" :  ... },
        #   { "role" : "caption",
        #     "content" : ... },
        # ]
        return 2
    
    @staticmethod
    def metric_key():
        return "CIDEr"

    # coco和flickr数据集中的每个图像通常对应多个描述文本，因此，可以选择第一个描述文本作为答案
    def extract_answer(self, item):
        # we use the first answer as grounding truth
        return item["sentences_raw"][0]

    @property
    def instruction(self):
        if self.cfg.is_icl:
            return "provide a short caption of the input image."
        return None

    def eval(
        self,
        eval_cfg,
        model,
    ):
        result = []
        # 加载CIDEr评估指标
        # 当调用 metric = evaluate.load(...) 时，metric 是一个 Metric 类的实例（对象）
        # 状态：存储待处理的预测（predictions）和参考答案（references）。
        # 方法：提供 add()、compute() 等接口用于添加数据和计算结果。
        # metric.add(prediction=prediction, reference=answer)
        # compute() 基于缓存的所有预测和参考，计算最终得分（如 CIDEr 分数），并返回一个字典（{"cider": 0.55}）。
        metric = evaluate.load(
            os.path.join(paths.testbed_dir, "evaluate", "metrics", "CIDEr")
        )
        eval_dl = self.validation_dataloader(eval_cfg.batch_size)
        # 选择评估eval_cfg.iterations个batch进行评估或者全部的batch
        iterations = eval_cfg.iterations or len(eval_dl)
        generation_args = eval_cfg.generation_args
        # 强制限制模型生成的最大文本长度
        generation_args["max_new_tokens"] = 20
        # 使用 zip 的主要目的是 将两个可迭代对象（range(iterations) 和 eval_dl）对齐，确保循环最多执行 iterations 次，同时借助 tqdm 显示进度条。
        for _, batch in zip(
            range(iterations),
            tqdm(
                eval_dl,
                total=iterations,
                desc=f"Evaluating {model.model_name} with {get_expand_runname(eval_cfg)} ...",
            ),
        ):
            predictions = self.get_prediction(
                model,
                batch,
                max_skip_oom=eval_cfg.max_skip_oom,
                **generation_args,
            )
            if predictions is None:
                continue

            # zip配对
            for pred, context in zip(predictions, batch):
                last_item = context[-1]
                answer = last_item["sentences_raw"]
                prediction = postprocess_generation(
                    self.name,
                    pred,
                    ["\n", "Caption", "Image", "<", "Short"],
                )
                metric.add(prediction=prediction, reference=answer)
                record = {
                    "raw_output": pred,
                    "filename": last_item["filename"],
                    "sentences": last_item["sentences_raw"],
                    "prediction": prediction,
                }
                if self.name == "coco":
                    record.update(cocoid=last_item["cocoid"])
                result.append(record)

        return result, metric.compute()
