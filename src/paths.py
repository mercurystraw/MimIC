from pathlib import Path
import subprocess
import sys
sys.path.insert(0, "..")
machine_id = subprocess.run(
    ["cat", "/etc/machine-id"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
).stdout
# path
testbed_dir = str(Path(__file__).parent.parent / "testbed")
result_dir = str(Path(__file__).parent.parent / "results")

# 根据不同的机器码确定不同的机器，因此加载不同的/不同路径下数据集 以及 不同路径下的模型权重
# 4*A6000 主要从share/pyz/下加载数据集和模型权重

if "7fd422" in machine_id:
    # 8x3090
    coco_dir = "/data/share/datasets/mscoco2014"
    vqav2_dir = "/data/share/datasets/VQAv2"
    ok_vqa_dir = "/data/share/datasets/OK-VQA"
    karpathy_coco_caption_dir = "/data/share/datasets/karpathy-split"
    hateful_memes_dir = "/data1/share/dataset/hateful_memes"
    flickr30k_dir = karpathy_coco_caption_dir
    flickr30k_images_dir = "/data1/share/flickr30k"
    ocr_vqa_dir = "/data1/share/dataset/OCR-VQA"
    ocr_vqa_images_dir = "/data1/share/dataset/OCR-VQA/images"

    idefics_9b_path = "/data1/share/model_weight/idefics/idefics-9b"
    llava_interleave_7b_path = (
        "/data1/share/model_weight/llava/llava-interleave-qwen-7b-hf")
    idefics2_8b_path = "/data1/share/model_weight/idefics/idefics2-8b"  # you'd better not use idefics2-8b to run icl
    idefics2_8b_base_path = "/data1/share/model_weight/idefics/idefics2-8b-base"
elif "a5d380cc" in machine_id:
    # 4xa6000
    coco_dir = "/home/share/pyz/dataset/mscoco/mscoco2014"
    vqav2_dir = "/home/share/pyz/dataset/vqav2"
    ok_vqa_dir = "/home/share/pyz/dataset/okvqa"
    seed_dir = "/home/share/pyz/dataset/SEED"
    mme_dir = "/home/share/pyz/dataset/MME"
    karpathy_coco_caption_dir = "/home/share/karpathy-split"
    flickr30k_dir = karpathy_coco_caption_dir
    flickr30k_images_dir = "/home/share/flickr30k"
    ocr_vqa_dir = "/home/share/dataset/OCR-VQA"
    ocr_vqa_images_dir = "/home/share/dataset/OCR-VQA/images"

    idefics_9b_path = "/home/share/pyz/model_weight/idefics-9b"
    llava_interleave_7b_path = (
        "/home/share/pyz/model_weight/llava-interleave-qwen-7b-hf")
    idefics2_8b_path = "/home/share/pyz/model_weight/idefics2-8b"  # you'd better not use idefics2-8b to run icl
    idefics2_8b_base_path = "/home/share/pyz/model_weight/idefics2-8b-base"

elif "8faf59b" in machine_id:
    # 4x3090
    coco_dir = "/data/share/datasets/mscoco/mscoco2014"
    vqav2_dir = "/data/share/datasets/vqav2"
    seed_dir = "/data/share/datasets/SEED"
    ok_vqa_dir = "/data/share/datasets/okvqa"
    flickr30k_dir = "/data/share/datasets/flickr30k"
    karpathy_coco_caption_dir = "/data/share/datasets/karpathy-split"
    idefics_9b_path = "/data/share/model_weight/idefics/idefics-9b"
    llava_interleave_7b_path = "/data/share/model_weight/llava/llava-interleave-qwen-7b-hf"
    idefics2_8b_path = "/data/share/model_weight/idefics/idefics2-8b"  # you'd better not use idefics2-8b to run icl
    idefics2_8b_base_path = "/data/share/model_weight/idefics/idefics2-8b-base"
