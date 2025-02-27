import os
import random
import sys
import re
import datasets

sys.path.insert(0, "..")
import config

from tqdm import tqdm
from pathlib import Path
import evaluate

from testbed.data import prepare_dataloader, postprocess_generation, prepare_input
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils import *


def get_prediction(cfg, lmm, dataset_name, batch, instruction, **generate_args):
    text, images = prepare_input(dataset_name, batch, instruction=instruction)
    if isinstance(images, list):
        if isinstance(images[0], list):
            images = [[img.convert("RGB") for img in img_list] for img_list in images]
        else:
            images = [img.convert("RGB") for img in images]
    try:
        return lmm.generate(images, text, **generate_args)
    except torch.cuda.OutOfMemoryError:
        num_skip_oom = getattr(get_prediction, "__num_skip_oom", 0)
        if num_skip_oom > cfg.eval.max_skip_oom:
            raise
        else:
            get_prediction.__num_skip_oom = num_skip_oom + 1
    return None


def eval_coco(cfg, lmm, iterations, dataloader, hparams, runname):
    result = []
    metric = evaluate.load(
        os.path.join(config.testbed_dir, "evaluate", "metrics", "CIDEr")
    )
    for _, batch in zip(
        range(iterations),
        tqdm(
            dataloader,
            desc=f"Evaluating {lmm.model_name} with {runname} ...",
        ),
    ):
        predictions = get_prediction(
            cfg,
            lmm,
            "coco",
            batch,
            cfg.data.caption_instruction,
            **hparams["generate_args"],
        )
        if predictions is None:
            continue

        for pred, context in zip(predictions, batch):
            last_item = context[-1]
            answer = last_item["sentences_raw"]
            prediction = postprocess_generation(
                cfg.eval.query_set,
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
            if cfg.eval.query_set == "coco":
                record.update(cocoid=last_item["cocoid"])
            result.append(record)

    return result, metric.compute()


def eval_hm(cfg, lmm, iterations, dataloader, hparams, runname):
    result = []
    metric = evaluate.load("roc_auc")
    for _, batch in zip(
        range(iterations),
        tqdm(
            dataloader,
            desc=f"Evaluating {lmm.model_name} with {runname} ...",
        ),
    ):
        predictions = get_prediction(
            cfg,
            lmm,
            "hateful_memes",
            batch,
            cfg.data.hm_instruction,
            **hparams["generate_args"],
        )
        if predictions is None:
            continue

        for pred, context in zip(predictions, batch):
            last_item = context[-1]
            answer = last_item["label"]
            prediction = postprocess_generation(
                "hateful_memes", pred, ["\n", "Answer", "Short"]
            )
            metric.add(prediction_scores=prediction, references=answer)
            result.append(
                {
                    "id": last_item["id"],
                    "answer": last_item["label"],
                    "raw_output": pred,
                    "prediction": prediction,
                }
            )

    return result, metric.compute()


def eval_ocr_vqa(cfg, lmm, iterations, dataloader, hparams, runname):
    result = []
    metric = evaluate.load("exact_match")
    for _, batch in zip(
        range(iterations),
        tqdm(
            dataloader,
            desc=f"Evaluating {lmm.model_name} with {runname} ...",
        ),
    ):
        predictions = get_prediction(
            cfg,
            lmm,
            "ocr_vqa",
            batch,
            cfg.data.vqa_instruction,
            **hparams["generate_args"],
        )
        if predictions is None:
            continue
        for pred, context in zip(predictions, batch):
            last_qa = context[-1]
            prediction = postprocess_generation(
                cfg.eval.query_set,
                pred,
                ["\n", "Question", "Answer", "Image", "Short"],
            )
            gt_answer = last_qa["answer"]
            metric.add(prediction=prediction.lower(), reference=gt_answer.lower())
            result.append(
                {
                    "question_id": last_qa["question_id"],
                    "raw_output": pred,
                    "question": last_qa["question"],
                    "prediction": prediction,
                    "answer": last_qa["answer"],
                }
            )

    return result, metric.compute()


def eval_vqa(cfg, lmm, iterations, dataloader, hparams, runname):
    result = []
    metric = evaluate.load(
        os.path.join(config.testbed_dir, "evaluate", "metrics", "vqa_accuracy")
    )
    for _, batch in zip(
        range(iterations),
        tqdm(
            dataloader,
            desc=f"Evaluating {lmm.model_name} with {runname} ...",
        ),
    ):
        predictions = get_prediction(
            cfg,
            lmm,
            cfg.data.name,
            batch,
            cfg.data.vqa_instruction,
            **hparams["generate_args"],
        )
        if predictions is None:
            continue
        for pred, context in zip(predictions, batch):
            last_qa = context[-1]
            prediction = postprocess_generation(
                cfg.eval.query_set,
                pred,
                ["\n", "Question", "Answer", "Image", "Short"],
            )
            gt_answer = [item["answer"] for item in last_qa["answers"]]
            metric.add(
                prediction=prediction,
                reference=gt_answer,
                question_types=last_qa["question_type"],
                answer_types=last_qa["answer_type"],
            )
            result.append(
                {
                    "question_id": last_qa["question_id"],
                    "raw_output": pred,
                    "question": last_qa["question"],
                    "question_type": last_qa["question_type"],
                    "answer_type": last_qa["answer_type"],
                    "prediction": prediction,
                    "answers": last_qa["answers"],
                }
            )

    return result, metric.compute()


def eval_seed_bench(cfg, lmm, iterations, dataloader, hparams, runname):
    result = []
    metric = evaluate.load("exact_match")
    for _, batch in zip(
        range(iterations),
        tqdm(
            dataloader,
            desc=f"Evaluating {lmm.model_name} with {runname} ...",
        ),
    ):
        predictions = get_prediction(
            cfg,
            lmm,
            "seed_bench",
            batch,
            cfg.data.vqa_instruction,
            **hparams["generate_args"],
        )
        if predictions is None:
            continue
        for pred, context in zip(predictions, batch):
            last_qa = context[-1]
            prediction = postprocess_generation(
                "mmmu-pro", pred, stop_words=["\n", "."]
            )
            if prediction.upper() not in ["A", "B", "C", "D"]:
                prediction = random.choice(["A", "B", "C", "D"])
            gt_answer = last_qa["answer"]
            metric.add(
                prediction=prediction,
                reference=gt_answer,
            )
            result.append(
                {
                    "question": last_qa["question"],
                    "question_id": last_qa["question_id"],
                    "raw_output": pred,
                    "question": last_qa["question"],
                    "choice_a": last_qa["choice_a"],
                    "choice_b": last_qa["choice_b"],
                    "choice_c": last_qa["choice_c"],
                    "choice_d": last_qa["choice_d"],
                    "prediction": prediction,
                    "answer": last_qa["answer"],
                }
            )

    return result, metric.compute()


def eval_mme(cfg, lmm, iterations, dataloader, hparams, runname):
    result = []
    metric = evaluate.load("accuracy")
    for _, batch in zip(
        range(iterations),
        tqdm(
            dataloader,
            desc=f"Evaluating {lmm.model_name} with {runname} ...",
        ),
    ):
        predictions = get_prediction(
            cfg,
            lmm,
            "mme",
            batch,
            cfg.data.mme_instruction,
            **hparams["generate_args"],
        )
        if predictions is None:
            continue
        for pred, context in zip(predictions, batch):
            last_qa = context[-1]
            gt_answer = last_qa["answer"]
            prediction = postprocess_generation("mme", pred, stop_words=["\n"])
            metric.add(prediction=prediction, reference=gt_answer.lower() == "yes")
            result.append(
                {
                    "prediction": prediction,
                    "answer": gt_answer,
                    "question": last_qa["question"],
                    "raw_output": pred,
                    "question_id": last_qa["question_id"],
                }
            )

    return result, metric.compute()


def eval_mmmu_pro(cfg, lmm, iterations, dataloader, hparams, runname):
    result = []
    metric = evaluate.load("exact_match")
    for _, batch in zip(
        range(iterations),
        tqdm(
            dataloader,
            desc=f"Evaluating {lmm.model_name} with {runname} ...",
        ),
    ):
        predictions = get_prediction(
            cfg,
            lmm,
            "mmmu-pro",
            batch,
            cfg.data.mme_instruction,
            **hparams["generate_args"],
        )
        if predictions is None:
            continue
        for pred, context in zip(predictions, batch):
            last_qa = context[-1]
            gt_answer = last_qa["answer"]
            prediction = postprocess_generation(
                "mmmu-pro", pred, stop_words=["\n", ".", "Qeustion", "Answer"]
            )
            metric.add(prediction=prediction, reference=gt_answer)
            result.append(
                {
                    "prediction": prediction,
                    "answer": gt_answer,
                    "question": last_qa["question"],
                    "raw_answer": pred,
                    "id": last_qa["id"],
                }
            )

    return result, metric.compute()


def prepare_dataset(name, split, train_size=None, seed=None):
    """
    Prepare dataset for training or evaluation
    Args:
        name (str): dataset name
        split (str): "train" or "validation"
        train_size (int, *optional*): size of the training set. If not set,
            the whole dataset will be used.
        seed (int, *optional*): random seed, used with train_size.

    Returns:
        dataset: dataset object
    """

    assert split in ["train", "validation"]
    if split == "train":
        args_key = "support_set_args"
    elif "query_set_args":
        # if query_set_args is same as train_set_args
        # you can use the same args_key
        args_key = (
            "query_set_args"
            if "query_set_args" in DATASET_MAPPING[name]
            else "support_set_args"
        )

    if name in ["vqav2", "ok_vqa", "ocr_vqa", "hm", "coco", "flickr"]:
        dataset = datasets.load_dataset(
            split=split,
            **DATASET_MAPPING[name][args_key],
            trust_remote_code=True,
        ).shuffle(seed=seed)
        if (
            train_size is not None
            and isinstance(train_size, int)
            and train_size < len(dataset)
            and split == "train"
        ):
            dataset = dataset.select(range(train_size))

    elif name in ["seed", "mme", "mmmu-pro"]:
        dataset = datasets.load_dataset(
            split="test",
            **DATASET_MAPPING[name][args_key],
            trust_remote_code=True,
        ).shuffle(seed=seed)

        if name == "mmmu-pro":
            # for simplity, only support one image
            dataset = dataset.filter(lambda x: x["image_2"] is None)
        if (
            train_size is not None
            and isinstance(train_size, int)
            and train_size < len(dataset)
        ):
            dataset = dataset.train_test_split(
                train_size=train_size, seed=seed, shuffle=False
            )["train" if split == "train" else "test"]
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    return dataset


def prepare_icl_dataset(
    support_set_name, query_set_name=None, train_size=None, seed=None
):
    """
    Prepare dataset for icl evaluation
    Args:
        support_set_name (str): support set name
        query_set_name (str, *optional*): query set name
        train_size (int, *optional*): size of the training set. It only used for
            seed bench, mme, and mmmu-pro datasets, since these datasets may only contains
            test set. This argument has no effect on other datasets.
        seed (int, *optional*): random seed, used with train_size.

    Returns:
        support_set: support set object
        query_set: query set object
    """
    if query_set_name is None:
        query_set_name = support_set_name
    support_set = prepare_dataset(support_set_name, "train", train_size, seed)
    query_set = prepare_dataset(query_set_name, "validation", train_size, seed)
    return support_set, query_set


DATASET_MAPPING = {
    "ok_vqa": {
        "support_set_args": dict(
            path=os.path.join(config.testbed_dir, "data", "ok_vqa"),
            data_dir=config.ok_vqa_dir,
            images_dir=config.coco_dir,
        ),
        "eval_fn": eval_vqa,
    },
    "vqav2": {
        "support_set_args": dict(
            path=os.path.join(config.testbed_dir, "data", "vqav2"),
            data_dir=config.vqav2_dir,
            images_dir=config.coco_dir,
        ),
        "query_set_args": dict(
            path=os.path.join(config.testbed_dir, "data", "vqav2"),
            data_dir=os.path.join(Path(__file__).parent, "..", "dataset", "vqav2"),
            images_dir=config.coco_dir,
        ),
        "eval_fn": eval_vqa,
    },
    "coco": {
        "support_set_args": dict(
            path=os.path.join(config.testbed_dir, "data", "coco"),
            data_dir=config.karpathy_coco_caption_dir,
            images_dir=config.coco_dir,
        ),
        "eval_fn": eval_coco,
    },
    "flickr": {
        "support_set_args": dict(
            path=os.path.join(config.testbed_dir, "data", "flickr"),
            data_dir=config.flickr30k_dir if hasattr(config, "flickr30k_dir") else None,
            images_dir=(
                config.flickr30k_images_dir
                if hasattr(config, "flickr30k_images_dir")
                else None
            ),
            name="flickr30k",
        ),
        "eval_fn": eval_coco,
    },
    "ocr_vqa": {
        "support_set_args": dict(
            path=os.path.join(config.testbed_dir, "data", "ocr_vqa"),
            data_dir=config.ocr_vqa_dir if hasattr(config, "ocr_vqa_dir") else None,
            images_dir=(
                config.ocr_vqa_images_dir
                if hasattr(config, "ocr_vqa_images_dir")
                else None
            ),
        ),
        "query_set_args": dict(
            path=os.path.join(config.testbed_dir, "data", "ocr_vqa"),
            data_dir=os.path.join(Path(__file__).parent, "..", "dataset", "ocr_vqa"),
            images_dir=(
                config.ocr_vqa_images_dir
                if hasattr(config, "ocr_vqa_images_dir")
                else None
            ),
        ),
        "eval_fn": eval_ocr_vqa,
    },
    "seed": {
        "support_set_args": dict(
            path=os.path.join(config.testbed_dir, "data", "seed_bench"),
            data_dir=(config.seed_dir if hasattr(config, "seed_dir") else None),
        ),
        "eval_fn": eval_seed_bench,
    },
    "hm": {
        "support_set_args": dict(
            path=os.path.join(config.testbed_dir, "data", "hateful_memes"),
            data_dir=(
                config.hateful_memes_dir
                if hasattr(config, "hateful_memes_dir")
                else None
            ),
        ),
        "eval_fn": eval_hm,
    },
    "mme": {
        "support_set_args": dict(
            path="parquet",
            data_dir=(config.mme_dir if hasattr(config, "mme_dir") else None),
        ),
        "eval_fn": eval_mme,
    },
    "mmmu-pro": {
        "support_set_args": dict(
            path="parquet",
            data_dir=(config.mmmu_pro_dir if hasattr(config, "mmmu_pro_dir") else None),
        ),
        "eval_fn": eval_mmmu_pro,
    },
}


@register_dataset_retriever("mme")
def retriever(item, is_last: bool):
    return (
        [
            {"role": "image", "content": [{"type": "image"}]},
            {
                "role": "question",
                "content": item["question"],
            },
            (
                {"role": "answer"}
                if is_last
                else {
                    "role": "answer",
                    "content": item["answer"],
                }
            ),
        ],
        item["image"],
    )


@register_postprocess("mme")
def postprocess(text):
    if text.lower() == "yes":
        return 1
    elif text.lower() == "no":
        return 0
    return -1


@register_dataset_retriever("mmmu-pro")
def retriever(item, is_last: bool):
    text_parts = re.split(r"(<image 1>)", item["question"], 1)
    question_contents = []
    for text in text_parts:
        if text:
            if text == "<image 1>":
                question_contents.append({"type": "image"})
            else:
                question_contents.append({"type": "text", "text": text})

    question_contents.append({"type": "text", "text": "Answer with the letter."})

    return (
        [
            {
                "role": "question",
                "content": question_contents,
            },
            {
                "role": "choices",
                "content": [
                    {"type": "text", "text": f"{chr(65+i)}. {item}"}
                    for i, item in enumerate(eval(item["options"]))
                ],
            },
            (
                {"role": "answer"}
                if is_last
                else {
                    "role": "answer",
                    "content": item["answer"],
                }
            ),
        ],
        item["image_1"],
    )


@register_postprocess("mmmu-pro")
def postprocess(text):
    if text != "" and text[0].upper() in ["A", "B", "C", "D"]:
        return text[0].upper()
    return random.choice(["A", "B", "C", "D"])
