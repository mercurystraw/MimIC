import argparse
import os
import subprocess
import sys
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import utils
import shlex


def merge_args(base_args, new_args):
    base_dict = {arg.partition("=")[0]: arg for arg in base_args}

    if new_args:
        new_dict = {arg.partition("=")[0]: arg for arg in new_args}
        base_dict.update(new_dict)

    return list(base_dict.values())


def get_avail_devices(devices, requires_memory=None):
    if not hasattr(get_avail_devices, "cached_requires_memory"):
        if requires_memory is None:
            raise ValueError("requires_memory must be provided on the first call.")
        get_avail_devices.cached_requires_memory = requires_memory

    effective_requires_memory = (
        requires_memory
        if requires_memory is not None
        else get_avail_devices.cached_requires_memory
    )

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    free_memory = result.stdout.strip().split("\n")
    free_gpus = [
        str(idx)
        for idx, mem in enumerate(free_memory)
        if int(mem) > effective_requires_memory
    ]

    if devices:
        return ",".join(set(devices.split(",")) & set(free_gpus))
    else:
        return ",".join(free_gpus)


def run_train(
    run_name,
    dataset,
    num_query_sample,
    num_shot,
    model_name,
    train_args,
    devices,
):
    try:
        process = subprocess.Popen(
            [
                "python",
                "train.py",
            ]
            + merge_args(
                [
                    f"runname={run_name}",
                    f"lmm={model_name}",
                    f"data.num_query_samples={num_query_sample}",
                    f"data.name={dataset}",
                    f"data.num_shot={num_shot}",
                ],
                shlex.split(train_args),
            ),
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": get_avail_devices(devices),
            },
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
        )

        returncode = process.wait()

        if returncode != 0:
            if process.stderr and "torch.cuda.OutOfMemoryError" in process.stderr:
                return dataset, num_query_sample, num_shot
        else:
            return True
    except Exception as e:
        print(f"An error occurred during training: {e}", file=sys.stderr)

    return False


def run_eval(
    run_name, dataset, num_query_sample, num_shot, model_name, gpu_id, eval_args
):
    try:
        process = subprocess.Popen(
            ["python", "eval.py"]
            + merge_args(
                [
                    f"lmm={model_name}",
                    f"runname={run_name}",
                    f"data.name={dataset}",
                    f"data.num_shot={num_shot}",
                    f"data.num_query_samples={num_query_sample}",
                ],
                shlex.split(eval_args),
            ),
            env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu_id},
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
        )

        returncode = process.wait()

        if returncode != 0:
            if process.stderr and "torch.cuda.OutOfMemoryError" in process.stderr:
                return dataset, num_query_sample, num_shot
            print(
                f"Evaluation failed on GPU {gpu_id} for runname: {run_name}, dataset: {dataset}"
            )
        else:
            return True
    except Exception as e:
        print(f"An error occurred during evaluation: {e}", file=sys.stderr)
    return False


def run_analyze(
    run_name, dataset, num_query_sample, num_shot, model_name, analyze_args
):
    try:
        subprocess.run(
            ["python", "analyze.py"]
            + merge_args(
                [
                    f"lmm={model_name}",
                    f"runname={run_name}",
                    f"data.name={dataset}",
                    f"data.num_shot={num_shot}",
                    f"data.num_query_samples={num_query_sample}",
                ],
                shlex.split(analyze_args),
            ),
            env=os.environ,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(
            f"Analyze failed for runname: {run_name}, dataset: {dataset}",
            file=sys.stderr,
        )
        sys.exit(1)


def execute_eval(run_name, tasks, model_name, eval_args, devices):
    task_queue = tasks.copy()

    with ThreadPoolExecutor() as executor:
        futures = {}
        while task_queue or futures:
            available_gpus = get_avail_devices(devices).split(",")
            for gpu_id in available_gpus:
                if task_queue and gpu_id not in futures:
                    dataset, num_query_sample, num_shot = task_queue.pop(0)
                    print(
                        f"Assigning task to GPU {gpu_id}: {dataset}, {num_query_sample}, {num_shot}"
                    )
                    futures[gpu_id] = executor.submit(
                        run_eval,
                        run_name,
                        dataset,
                        num_query_sample,
                        num_shot,
                        model_name,
                        gpu_id,
                        eval_args,
                    )

            ret = next(as_completed(futures.values())).result()
            if isinstance(ret, tuple):
                task_queue.append(ret)
            elif not ret:
                sys.exit(1)

            futures = {gpu_id: f for gpu_id, f in futures.items() if not f.done()}


def main():
    parser = argparse.ArgumentParser(description="Run training and evaluation tasks.")
    parser.add_argument(
        "-r", "--run-name", required=True, help="Name for the current run."
    )
    parser.add_argument(
        "-d", "--datasets", required=True, help="Comma-separated list of datasets."
    )
    parser.add_argument(
        "-m", "--model-name", required=True, help="Name of the model to use."
    )
    parser.add_argument(
        "-q",
        "--num-query-samples",
        default="",
        help="Comma-separated list of query samples.",
    )
    parser.add_argument(
        "-s", "--num-shots", required=True, help="Comma-separated list of shots."
    )
    parser.add_argument("-t", "--train", action="store_true", help="Enable train mode.")
    parser.add_argument(
        "-e",
        "--eval",
        action="store_true",
        help="Enable eval mode. If set to true, analyze mode will be also enabled.",
    )
    parser.add_argument(
        "-a", "--analyze", action="store_true", help="Enable analyze mode."
    )
    parser.add_argument(
        "--train-args",
        required=False,
        help="Additional training arguments.",
    )
    parser.add_argument(
        "--eval-args",
        required=False,
        help="Additional evaluation arguments.",
    )
    parser.add_argument(
        "--analyze-args",
        required=False,
        help="Additional evaluation arguments.",
    )
    parser.add_argument(
        "--devices",
        required=False,
        help="Comma-separated list of integers that denotes devices used to train or evaluate.",
    )
    parser.add_argument(
        "--wait-devices-timeout",
        type=int,
        default=0,
        help="Maximum time in minutes to wait for free GPUs. If <= 0, exit immediately if no adequate GPUs are available.",
    )
    parser.add_argument(
        "--requires_memory",
        type=int,
        default=20000,
        help="The minimal cuda memory used to run train or eval, unit MB.",
    )
    parser.add_argument(
        "--wait-n-devices",
        type=int,
        default=1,
        help="Minimum devices are required to start.",
    )
    args = parser.parse_args()

    datasets = args.datasets.split(",")
    num_query_samples = args.num_query_samples.split(",")
    num_shots = args.num_shots.split(",")
    run_name = args.run_name
    model_name = args.model_name
    train_args = getattr(args, "train_args") or ""
    eval_args = getattr(args, "eval_args") or ""
    analyze_args = getattr(args, "analyze_args") or ""
    devices = args.devices
    timeout = args.wait_devices_timeout
    min_devices = max(args.wait_n_devices, 1)
    requires_memory = args.requires_memory

    if not (args.train | args.eval | args.analyze):
        args.train = args.eval = args.analyze = True

    if (
        not get_avail_devices(devices, requires_memory)  # no even one devices
        or len(get_avail_devices(devices, requires_memory).split(","))
        < min_devices  # less than minimal required devices
    ) and (args.eval or args.train):
        print(f"Cannot find at least {min_devices} devcie(s). Start waiting...")
        for i in range(0, timeout):
            if (
                get_avail_devices(devices, requires_memory)
                and len(get_avail_devices(devices, requires_memory).split(","))
                >= min_devices
            ):
                break
            time.sleep(60)
            print(f"Waited for {min_devices} device(s) for {(i+1)} mins")
        else:
            print(f"Cannot find at least {min_devices} devcie(s). Timeout, exit...")
            return

    if num_query_samples:
        tasks = list(product(datasets, num_query_samples, num_shots))
    else:
        if args.train:
            raise RuntimeError(
                "The option -q/--num_query_samples is required if train mode is enabled."
            )
        tasks = list(product(datasets, num_shots))

    if args.train:
        print("Starting training phase...")
        task_queue = tasks.copy()
        while task_queue:
            dataset, num_query_sample, num_shot = task_queue.pop(0)
            ret = run_train(
                run_name,
                dataset,
                num_query_sample,
                num_shot,
                model_name,
                train_args,
                devices,
            )

            if isinstance(ret, tuple):
                task_queue.append(ret)
            elif not ret:
                return

    if args.eval:
        print("Starting evaluation phase...")
        execute_eval(run_name, tasks, model_name, eval_args, devices)

    if args.analyze:
        print("Starting analysis phase...")
        for dataset, num_query_sample, num_shot in tasks:
            run_analyze(
                run_name, dataset, num_query_sample, num_shot, model_name, analyze_args
            )


if __name__ == "__main__":
    main()
