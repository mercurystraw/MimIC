import random

import evaluate
import os
from datasets import load_dataset
from tqdm import tqdm

import src.paths as paths
from src.dataset_utils import DatasetBase

from testbed.data import (
    postprocess_generation,
    register_postprocess,
    register_dataset_retriever
)
from src.utils import get_expand_runname

@register_dataset_retriever("boolq")
def retriever(item, is_last):
    return [
        {
            "role": "passage",
            "content":item["passage"]
        },
        {
            "role":"question",
            "content":item["question"],
        },
        (
            {"role": "answer"}
            if is_last
            else {
                "role": "answer",
                "content": str(item["answer"]),
            }
        ),
    ]


@register_postprocess("boolq")
def postprocess(pred):
    pred = pred.lower()
    if "yes" in pred or "true" in pred:
        return 1
    elif "no" in pred or "false" in pred:
        return 0
    return -1

class Dataset(DatasetBase):
    support_datasets = ["boolq"]

    def __init__(self, data_cfg):
        super().__init__(data_cfg)
        ds_init_args = dict(
            path="parquet",
            data_dir=paths.boolq_dir,
        )
        assert (
            data_cfg.num_query_samples
        ), f"num_query_samples must be specified and greater than 0, but got {data_cfg.num_query_samples}"

        dataset = load_dataset(**ds_init_args, trust_remote_code=True)

        self._support_set = dataset["train"]
        self._query_set = dataset["validation"]

    @property
    def num_role_in_round(self):
        # BoolQ为纯文本QA任务，角色结构：
        # [
        #   {"role": "question", "content": ...},
        #   {"role": "answer", "content": ...}
        # ]
        return 2

    @staticmethod
    def metric_key():
        return "accuracy"

    def extract_answer(self,item):
        # BoolQ的答案为True/False，转换为小写字符串
        return str(item["answer"]).lower()

    @property
    def instruction(self):
        if self.cfg.is_icl:
            return "Read the passage carefully and Provide an answer in 'true' or 'false' to the question based on the passage. "
        return None

    def eval(
        self,
        eval_cfg,
        model
    ):
        result = []
        metric = evaluate.load("accuracy")
        eval_dl = self.validation_dataloader(eval_cfg.batch_size)
        iterations = eval_cfg.iterations or len(eval_dl)

        # generation_args = eval_cfg.generation_args
        # generation_args["max_new_tokens"] = 5  # 限制生成长度

        for _, batch in zip(
            range(iterations),
            tqdm(
                eval_dl,
                total=iterations,
                desc=f"Evaluating {model.model_name} with {get_expand_runname(eval_cfg)}...",
            ),
        ):
            predictions = self.get_prediction(
                model,
                batch,
                max_skip_oom=eval_cfg.max_skip_oom,
                **eval_cfg.generation_args,
            )
            if predictions is None:
                continue

            for pred, context in zip(predictions, batch):
                last_item = context[-1]
                prediction = postprocess_generation(
                    self.name,
                    pred,
                    stop_words=["\n", "Passage", "Question", "Answer"]
                )

                # 将模型输出映射到True/False
                # if "yes" in prediction or "true" in prediction:
                #     prediction = "true"
                # elif "no" in prediction or "false" in prediction:
                #     prediction = "false"
                # else:  # 无法解析时随机猜测
                #     prediction = random.choice(["true", "false"])

                gt_answer = str(last_item["answer"])
                # prediction = postprocess_generation(self.name, pred, stop_words=["\n"])

                metric.add(predictions=prediction, references=gt_answer)
                result.append({
                    "question": last_item["question"],
                    "passage": last_item["passage"],
                    "raw_output": pred,
                    "prediction": prediction,
                    "answer": gt_answer,

                })

        return result, metric.compute()