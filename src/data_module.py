import pytorch_lightning as pl
import datasets
from torch.utils.data import (
    DistributedSampler,
    BatchSampler,
    SequentialSampler,
    RandomSampler,
)
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from testbed.data import prepare_dataloader, prepare_input
import config
from PIL import ImageFile
from mydatasets import *

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataModule(pl.LightningDataModule):

    def __init__(self, cfg, lmm) -> None:
        super().__init__()
        self.cfg = cfg
        self.lmm = lmm

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.dataset = prepare_dataset(
                self.cfg.data.name,
                "train",
                train_size=self.cfg.data.num_query_samples,
                seed=self.cfg.seed,
            )

    def convert_imgs_to_rgb(self, images):
        return [[img.convert("RGB") for img in img_list] for img_list in images]

    def collate_fn(self, batch):
        """
        Split batch into full context, in-context examples, query and answer, and process them into model inputs.
        """
        num_item_in_query = None  # the number of items in the query, see comment below

        if self.cfg.data.name in ["vqav2", "ok_vqa", "ocr_vqa"]:
            context, images = prepare_input(
                self.cfg.data.name, batch, instruction=self.cfg.data.vqa_instruction
            )

            # we use the first answer as grounding truth
            answers = [
                (
                    item[-1]["answers"][0]["answer"]
                    if "answers" in item[-1]  # ocr_vqa doesn't have answer list
                    else item[-1]["answer"]
                )
                for item in batch
            ]

            # the last 3 items:
            # [
            #   { "role" : "image",
            #     "content" :  ... },
            #   { "role" : "question",
            #     "content" : ... },
            #   { "role" : "answer" }
            # ]
            num_item_in_query = 3
        elif self.cfg.data.name == "coco" or self.cfg.data.name == "flickr":
            context, images = prepare_input(
                self.cfg.data.name, batch, instruction=self.cfg.data.caption_instruction
            )

            answers = [item[-1]["sentences_raw"][0] for item in batch]
            # the last 2 items:
            # [
            #   { "role" : "image"
            #     "content" :  ... },
            #   { "role" : "caption" }
            # ]
            num_item_in_query = 2
        elif self.cfg.data.name == "hm":
            context, images = prepare_input(
                "hateful_memes", batch, instruction=self.cfg.data.hm_instruction
            )
            answers = ["yes" if item[-1]["label"] == 1 else "no" for item in batch]
            # the last 2 items:
            # [
            #   { "role" : ""
            #     "content" :  ... },
            #   { "role" : "answer" }
            # ]
            num_item_in_query = 2
        elif self.cfg.data.name == "seed":
            context, images = prepare_input(
                "seed_bench", batch, instruction=self.cfg.data.vqa_instruction
            )
            answers = [item[-1]["answer"] for item in batch]
            # the last 3 items:
            # [
            #   { "role" : "image",
            #     "content" :  ... },
            #   { "role" : "question",
            #     "content" : ... },
            #   { "role" : "choices",
            #     "content" : ... },
            #   { "role" : "answer" }
            # ]
            num_item_in_query = 4
        elif self.cfg.data.name in ["mme"]:
            context, images = prepare_input(
                self.cfg.data.name,
                batch,
                instruction=self.cfg.data.mme_instruction,
            )
            answers = [item[-1]["answer"] for item in batch]
            # the last 3 items:
            # [
            #   { "role" : "image",
            #     "content" :  ... },
            #   { "role" : "question",
            #     "content" : ... },
            #   { "role" : "answer" }
            # ]
            num_item_in_query = 3
        elif self.cfg.data.name in ["mmmu-pro"]:
            context, images = prepare_input(
                self.cfg.data.name,
                batch,
                instruction=self.cfg.data.vqa_instruction,
            )
            answers = [item[-1]["answer"] for item in batch]
            # the last 2 items:
            # [
            #   { "role" : "question"
            #     "content" :  ... },
            #   { "role" : "choices",
            #     "content" : ... },
            #   { "role" : "answer" }
            # ]
            num_item_in_query = 3

        prefix_texts = (
            self.lmm.apply_prompt_template(
                [ctx[:-num_item_in_query] for ctx in context]
            )
            if self.cfg.data.num_shot > 0
            else None
        )
        
        query_texts = self.lmm.apply_prompt_template(
            [ctx[-num_item_in_query:] for ctx in context]
        )

        return {
            "prefix_texts": prefix_texts,
            "query_texts": query_texts,
            "answers": answers,
            "images": self.convert_imgs_to_rgb(images),
        }

    def train_dataloader(self):
        samplers = (
            [
                BatchSampler(
                    (
                        RandomSampler(
                            self.dataset,
                            replacement=True,
                            num_samples=self.cfg.data.num_shot
                            * self.cfg.data.num_query_samples,
                        )
                    ),
                    batch_size=self.cfg.data.num_shot,
                    drop_last=True,
                )
            ]
            if self.cfg.data.num_shot > 0
            else []
        )
        if self.trainer.world_size > 1:
            samplers.append(DistributedSampler(self.dataset, shuffle=False))
        else:
            samplers.append(SequentialSampler(self.dataset))

        # we use the same dataset for both query set and support set
        # because we assume we only have cfg.data.num_query_samples data
        return prepare_dataloader(
            [self.dataset, self.dataset] if len(samplers) > 1 else self.dataset,
            self.cfg.training.batch_size,
            num_per_dataset=[self.cfg.data.num_shot, 1] if len(samplers) > 1 else [1],
            collate_fn=self.collate_fn,
            samplers=samplers,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            shuffle=True,
        )
