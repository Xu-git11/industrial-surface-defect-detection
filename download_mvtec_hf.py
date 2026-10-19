"""
download_mvtec_hf.py

从 HuggingFace 镜像下载 MVTec AD 单类别，保存为项目需要的目录结构。

用途：
    MVTec 官网的按类别下载链接（myDrive.ch）经常 404，这是替代方案。

用法：
    python download_mvtec_hf.py --category bottle

需要：
    pip install datasets

注意：
    - 默认使用 HuggingFace 国内镜像 https://hf-mirror.com
    - 如果 HuggingFace 也访问失败，先用 prepare_demo_data.py 生成本地合成数据
"""

import os
import argparse
from pathlib import Path

# 默认走 HuggingFace 国内镜像，国内访问更稳定
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def main():
    ap = argparse.ArgumentParser(description="从 HuggingFace 下载 MVTec AD 单类别")
    ap.add_argument("--category", default="bottle", help="类别名，如 bottle、cable 等")
    ap.add_argument("--output", default="data/mvtec_anomaly_detection",
                    help="输出根目录")
    ap.add_argument("--repo", default="TheoM55/mvtec_all_objects_split",
                    help="HuggingFace 数据集 ID")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("[失败] 缺少 datasets 库")
        print("请运行：python -m pip install datasets")
        return

    out_root = Path(args.output) / args.category
    out_root.mkdir(parents=True, exist_ok=True)

    splits = [f"{args.category}.train", f"{args.category}.test"]

    for split_name in splits:
        print(f"\n[下载] {split_name} ...")
        try:
            ds = load_dataset(
                args.repo,
                split=split_name,
                trust_remote_code=True,
                download_mode="reuse_cache_if_exists",
            )
        except Exception as e:
            print(f"[失败] {type(e).__name__}: {e}")
            print("\n可能原因：")
            print("  1. HuggingFace（含 hf-mirror）访问失败")
            print("  2. 数据集 ID 或 split 名称已变更")
            print("  3. 网络被代理/防火墙拦截")
            print("\n建议：")
            print("  - 先运行 python prepare_demo_data.py 生成本地合成数据跑通流程")
            print("  - 或尝试 Kaggle 镜像：ipythonx/mvtec-ad")
            return

        if split_name.endswith(".train"):
            # train split 全部是正常样本
            good_dir = out_root / "train" / "good"
            good_dir.mkdir(parents=True, exist_ok=True)
            for i, sample in enumerate(ds):
                img = sample["image_path"]
                img.save(good_dir / f"{i:03d}.png")
            print(f"  保存 {len(ds)} 张训练图 -> {good_dir}")

        else:
            # test split 包含 good 和各种缺陷
            counts = {}
            for sample in ds:
                defect = sample.get("defect", "unknown")
                counts[defect] = counts.get(defect, 0) + 1
                idx = counts[defect]

                test_dir = out_root / "test" / defect
                test_dir.mkdir(parents=True, exist_ok=True)
                sample["image_path"].save(test_dir / f"{idx:03d}.png")

                if sample.get("label") == 1 and sample.get("mask_path") is not None:
                    mask_dir = out_root / "ground_truth" / defect
                    mask_dir.mkdir(parents=True, exist_ok=True)
                    sample["mask_path"].convert("L").save(mask_dir / f"{idx:03d}_mask.png")

            print(f"  保存测试图分布 -> {counts}")

    print(f"\n[完成] 数据集已保存到：{out_root.resolve()}")
    print("下一步：")
    print(f"  python 01_traditional_opencv.py --category {args.category}")


if __name__ == "__main__":
    main()
