import os
import sys
import json
import re
import hydra
from omegaconf import DictConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, "..")
import config
from utils import *
from termcolor import colored


def sort_runname_key(runname: str):
    if runname.endswith("shot"):
        # {runname}-{num_query_samples}-{num_shot}shot
        prefix, query_samples, shot = runname.rsplit("-", 2)
        return (prefix, int(query_samples), int(re.findall(r"\d+", shot)[0]))
    else:
        # legacy format: {runname}-{num_query_samples}
        prefix, query_samples = runname.rsplit("-", 1)
        return (prefix, int(query_samples))


@hydra.main(config_path="config", config_name="exp_settings.yaml", version_base=None)
def main(cfg: DictConfig):
    verbose = cfg.get("verbose", False)
    metric_key = cfg.get("metric_key", None)
    runname = get_full_runname(cfg)
    topk = cfg.get("topk", 1)

    record_base_dir = os.path.join(config.result_dir, "record")
    record_dirs = {
        name: os.path.join(record_base_dir, name)
        for name in os.listdir(record_base_dir)
        if name.startswith(runname)
        and os.path.isdir(os.path.join(record_base_dir, name))
    }
    ckpt_base_dir = os.path.join(config.result_dir, "ckpt")
    ckpt_dirs = {
        name: os.path.join(ckpt_base_dir, name)
        for name in sorted(os.listdir(ckpt_base_dir), key=sort_runname_key)
        if name.startswith(runname) and os.path.isdir(os.path.join(ckpt_base_dir, name))
    }

    summary = {}
    missing_records = {}

    all_runnames = set(record_dirs.keys()) | set(ckpt_dirs.keys())

    for full_runname in all_runnames:
        meta_info = {}
        record_dir = record_dirs.get(full_runname, None)
        ckpt_dir = ckpt_dirs.get(full_runname, None)

        if record_dir and os.path.isdir(record_dir):
            for record_file in os.listdir(record_dir):
                if record_file.endswith(f"-on-{cfg.eval.query_set}.json"):
                    # Extract epoch_ckpt from filename
                    epoch_ckpt = record_file.split("-on-")[0]
                    try:
                        with open(os.path.join(record_dir, record_file)) as f:
                            content = json.load(f)

                        # Get the metric based on the provided key, or fallback to 'overall' and 'CIDEr'
                        if metric_key:
                            metric_value = content["eval_result"].get(metric_key, None)
                        else:
                            metric_value = content["eval_result"].get("overall", None)
                            if metric_value is None:
                                metric_value = content["eval_result"].get("CIDEr", None)
                                if metric_value is None:
                                    metric_value = content["eval_result"].get("exact_match", None)
                                if metric_value is None:
                                    metric_value = content["eval_result"].get("accuracy", None)
                                if metric_value is not None:
                                    metric_value *= 100  # Multiply CIDEr by 100

                        if metric_value is None:
                            raise KeyError(
                                f"Metric '{metric_key}' or fallback metrics 'overall'/'CIDEr' not found in eval_result."
                            )

                        meta_info[epoch_ckpt] = metric_value

                    except (KeyError, json.JSONDecodeError) as e:
                        print(f"Error processing {record_file}: {e}", file=sys.stderr)
        else:
            if verbose:
                print(f"No record directory found for {full_runname}", file=sys.stderr)

        if ckpt_dir:
            if os.path.isdir(ckpt_dir):
                for epoch_ckpt in os.listdir(ckpt_dir):
                    if epoch_ckpt not in meta_info:
                        # There is a checkpoint without a corresponding record
                        missing_epoch = int(re.findall(r"\d+", epoch_ckpt)[0])
                        missing_records.setdefault(full_runname, []).append(
                            missing_epoch
                        )
            else:
                print(
                    colored(f"{full_runname:<40}", "light_blue"),
                    "| Cannot find any checkpoints",
                    file=sys.stderr,
                )
                continue

        if meta_info:
            sorted_meta_info = sorted(
                meta_info.items(), key=lambda item: item[1], reverse=True
            )

            topk_meta_info = sorted_meta_info[:topk]
            summary[full_runname] = topk_meta_info

            print(
                colored(f"{full_runname:<40}", "light_blue"),
                f"| Top {min(topk, len(topk_meta_info))} records:",
            )
            for i, (epoch, metric) in enumerate(topk_meta_info):
                metric = round(metric, 2) if metric is not None else None
                print(f"  {i+1}. Epoch: {epoch} (Metric: {metric})")
        else:
            print(
                colored(f"{full_runname:<40}", "light_blue"),
                f"| Cannot find any records",
                file=sys.stderr,
            )

    if verbose:
        if missing_records:
            print("Missing records detected:")
            for runname, epochs in missing_records.items():
                print(f"{runname} missing epochs: {sorted(epochs)}")
        elif not summary:
            print("No records or checkpoints are found.")
        else:
            print("All checkpoints are evaluated.")


if __name__ == "__main__":
    main()
