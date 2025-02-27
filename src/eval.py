from functools import partial
import random
import omegaconf
import torch
from torch.utils.data import SequentialSampler, RandomSampler
import os
import sys
import re
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, "..")
import config

from shift_encoder import AttnApproximator, AttnFFNShift, ShiftStrategy
import hydra

from tqdm import tqdm
from pathlib import Path
import evaluate

from testbed.data import prepare_dataloader, postprocess_generation, prepare_input
from omegaconf import DictConfig, OmegaConf
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils import *
from mydatasets import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_dataloader(cfg, batch_size, num_shot, is_icl):
    eval_cfg = cfg.eval
    query_set_name, support_set_name, query_set_size = (
        eval_cfg.query_set,
        eval_cfg.support_set,
        eval_cfg.query_set_size,
    )

    if is_icl and num_shot > 0:
        # evaluate icl
        support_set, query_set = prepare_icl_dataset(
            support_set_name,
            query_set_name,
            train_size=cfg.data.num_query_samples,
            seed=cfg.seed,
        )
        if query_set_size is not None:
            if query_set_size > len(query_set):
                raise ValueError(
                    f"query_set_size should be less than or equal to the size of query_set, but got {query_set_size} > {len(query_set)}"
                )
            query_set = query_set.select(range(min(len(query_set), query_set_size)))
            if num_shot * min(query_set_size, len(query_set)) > len(support_set):
                support_set_sampler = RandomSampler(
                    support_set,
                    replacement=True,
                    num_samples=num_shot * min(query_set_size, len(query_set)),
                )
        else:
            support_set_sampler = RandomSampler(
                support_set,
            )
        dataloader = prepare_dataloader(
            [support_set, query_set],
            batch_size=batch_size,
            num_per_dataset=[num_shot, 1],
            samplers=[
                support_set_sampler,
                SequentialSampler(query_set),
            ],
            drop_last=True,
        )
    else:
        query_set = prepare_dataset(
            query_set_name,
            split="validation",
            train_size=cfg.data.num_query_samples,
            seed=cfg.seed,
        )
        if query_set_size is not None:
            if query_set_size > len(query_set):
                raise ValueError(
                    f"query_set_size should be less than or equal to the size of query_set, but got {query_set_size} > {len(query_set)}"
                )
            query_set = query_set.select(range(min(len(query_set), query_set_size)))

        if is_icl:
            query_set = query_set.shuffle(seed=cfg.seed).select(
                range(min(len(query_set), query_set_size))
            )
        dataloader = prepare_dataloader(
            query_set, batch_size=batch_size, num_shots=num_shot, drop_last=True
        )

    return dataloader


@hydra.main(config_path="config", config_name="exp_settings.yaml", version_base=None)
def main(cfg: DictConfig):
    eval_cfg = cfg.eval
    is_icl = eval_cfg.ckpt_epochs is None
    hparams = {
        "is_icl": is_icl,
        "batch_size": eval_cfg.batch_size,
        "num_shot": eval_cfg.num_shot,
        "generate_args": dict(eval_cfg.generation_args),
    }
    if eval_cfg.query_set == "coco" or eval_cfg.query_set == "flickr":
        hparams["generate_args"]["max_new_tokens"] = 20
    elif eval_cfg.query_set == "hm":
        hparams["generate_args"]["max_new_tokens"] = 5

    dataloader = get_dataloader(
        cfg,
        batch_size=hparams["batch_size"],
        num_shot=hparams["num_shot"],
        is_icl=is_icl,
    )
    iterations = eval_cfg.iterations if eval_cfg.iterations != -1 else len(dataloader)

    if is_icl:
        runname = cfg.runname
        record_dir = os.path.join(config.result_dir, "record", runname)
        sub_dirs = [record_dir]
    else:
        runname = get_full_runname(cfg)
        record_dir = os.path.join(config.result_dir, "record", runname)
        model_dir = os.path.join(config.result_dir, "ckpt", runname)
        sub_dirs = {
            int(re.findall(r"\d+", epoch)[0]): os.path.join(model_dir, epoch)
            for epoch in os.listdir(model_dir)
            if os.path.isdir(os.path.join(model_dir, epoch))
        }
        try:
            if eval_cfg.ckpt_epochs == "all":
                sub_dirs = sub_dirs.values()
            elif isinstance(
                eval_cfg.ckpt_epochs, (list, omegaconf.listconfig.ListConfig)
            ):
                sub_dirs = [sub_dirs[e] for e in eval_cfg.ckpt_epochs]
            elif isinstance(eval_cfg.ckpt_epochs, int):
                sub_dirs = [sub_dirs[eval_cfg.ckpt_epochs]]
            else:
                if isinstance(eval_cfg.ckpt_epochs, str):
                    raise ValueError(
                        f"ckpt_epochs should be 'all' if it is str, but got {eval_cfg.ckpt_epochs}"
                    )
                else:
                    raise TypeError(
                        f"a int, list of int, str or None is required, but got {type(eval_cfg.ckpt_epochs)}"
                    )
        except KeyError as e:
            raise RuntimeError(
                f"The checkpoint {model_dir}/epoch-{str(e)} doesn't exist."
            )

    os.makedirs(record_dir, exist_ok=True)

    for ckpt_path_or_runname in sub_dirs:
        exist_results = [
            d.removesuffix(".json")
            for d in os.listdir(record_dir)
            if os.path.exists(os.path.join(record_dir, d))
        ]
        if cfg.resume_eval:
            prefix = (
                ckpt_path_or_runname if is_icl else ckpt_path_or_runname.split("/")[-1]
            )
            if f"{prefix}-on-{eval_cfg.query_set}" in exist_results:
                print(
                    f"Found exist record {os.path.join(record_dir, f'{prefix}-on-{eval_cfg.query_set}.json')}, skip..."
                )
                continue

        device = torch.device("cuda")
        lmm = build_model(cfg).to(device, eval(cfg.lmm.dtype))

        if not is_icl:
            encoder = hydra.utils.instantiate(cfg.encoder.cls, _partial_=True)(
                lmm=lmm
            ).to(device, eval(cfg.lmm.dtype))

            print(f"Loading from pretrained: {ckpt_path_or_runname}")
            load_from_pretrained(ckpt_path_or_runname, lmm, encoder)
            encoder.eval()
            hooks = encoder.register_shift_hooks()

        lmm.eval()

        save_path = os.path.join(
            record_dir,
            (
                f"{os.path.basename(ckpt_path_or_runname)}-on-{eval_cfg.query_set}.json"
                if not is_icl
                else f"{os.path.basename(ckpt_path_or_runname)}-on-{eval_cfg.query_set}-{cfg.data.num_shot}shot.json"
            ),
        )
        Path(save_path).touch()
        try:
            result, eval_result = DATASET_MAPPING[eval_cfg.query_set]["eval_fn"](
                cfg, lmm, iterations, dataloader, hparams, runname
            )
            print(f"Evaluation result for {ckpt_path_or_runname}: {eval_result}")

            evaluate.save(
                save_path,
                config=OmegaConf.to_container(cfg, resolve=True),
                hparams=hparams,
                eval_result=eval_result,
                records=result,
            )

            lmm.to(device=torch.device("cpu"))
            if not is_icl:
                encoder.to(device=torch.device("cpu"))
                del encoder
            del lmm
        except Exception:
            os.remove(save_path)
            raise


if __name__ == "__main__":
    main()
