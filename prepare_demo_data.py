"""
生成 MVTec AD 结构的合成缺陷数据集（v2）

v1 的问题：
    每张图的物体位置随机 ±15px、大小随机 ±10px，导致差分法完全失效——
    图像间的差异主要来自位置抖动，而不是缺陷。真实产线是相机固定、工件定位的。

v2 修正：
    - 物体位置固定，仅保留 ±2px 定位抖动（模拟真实上料误差）
    - 缺陷对比度和尺寸调到可检范围（模拟真实可见缺陷）
    - 保留噪声和纹理，让检测仍有挑战性（误检率不会是 0）

目录结构与 MVTec AD 一致：
    data/mvtec_ad_demo/
    └── bottle/
        ├── train/good/              正常样本（用于建模板）
        ├── test/good/               正常样本（用于测误检率）
        ├── test/broken_large/       大破损
        ├── test/broken_small/       小划痕
        ├── test/contamination/      污渍
        └── ground_truth/{缺陷类型}/*_mask.png

运行：
    python prepare_demo_data.py

生成后执行：
    python 01_traditional_opencv.py --data data/mvtec_ad_demo --category bottle
"""

import os
import cv2
import numpy as np
from pathlib import Path

# ------------------------------------------------------------------ 参数
SEED = 20261019          # 固定随机种子：保证结果可复现（面试时能说"可复现"）
IMG_SIZE = (256, 256)

# 物体几何：固定工位（模拟相机与夹具定位），仅保留微小抖动
OBJ_CENTER = (128, 128)
OBJ_RADIUS = (62, 82)
JITTER_PX = 2            # 上料定位误差 ±2px

# 灰度设定：背景暗、物体亮，模拟背光/同轴光打光后的对比
BG_GRAY = 45
OBJ_GRAY = 185

# 噪声与纹理：让检测不是白送分
NOISE_SIGMA = 2.5
TEXTURE_AMP = 6


def _rng():
    return np.random.default_rng(SEED)


def add_noise(img, sigma=NOISE_SIGMA, rng=None):
    """高斯噪声，模拟相机传感器噪点。"""
    rng = rng or _rng()
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def add_texture(img, amp=TEXTURE_AMP):
    """细密水平纹理，模拟金属表面加工纹路（这是误检的主要来源）。"""
    h, w = img.shape[:2]
    yy = np.arange(h).reshape(-1, 1).astype(np.float32)
    texture = (np.sin(yy / 3.0) * amp).astype(np.float32)
    texture = np.repeat(texture, w, axis=1)
    # 轻微模糊，避免纹理过于规则
    texture = cv2.GaussianBlur(texture, (0, 0), 0.8)
    return np.clip(img.astype(np.float32) + texture, 0, 255).astype(np.uint8)


def make_base_image(rng=None):
    """
    造一个"瓶子/金属件"在暗背景上的图。

    模拟真实产线：相机与光源固定，工件由夹具定位，因此物体位置基本固定，
    只有 ±2px 的上料抖动。这是差分检测法能成立的前提。

    返回：原图、物体 mask。
    """
    rng = rng or _rng()
    h, w = IMG_SIZE

    # 固定工位 + 微小定位抖动
    cx = OBJ_CENTER[0] + int(rng.integers(-JITTER_PX, JITTER_PX + 1))
    cy = OBJ_CENTER[1] + int(rng.integers(-JITTER_PX, JITTER_PX + 1))

    img = np.full((h, w), BG_GRAY, dtype=np.uint8)
    obj_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(obj_mask, (cx, cy), OBJ_RADIUS, 0, 0, 360, 255, -1)

    # 物体表面：径向渐变，模拟曲面受光不均
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    gradient = np.clip(1 - dist / max(OBJ_RADIUS), 0, 1)
    obj = (OBJ_GRAY - 20 + gradient * 35).astype(np.uint8)
    obj = np.clip(obj, 0, 255).astype(np.uint8)

    img[obj_mask > 0] = obj[obj_mask > 0]

    img = add_texture(img)
    img = add_noise(img, rng=rng)
    return img, obj_mask


def add_scratch(img, mask, obj_mask, kind="large", rng=None):
    """在物体区域内加划痕缺陷。返回修改后的图和 GT mask。"""
    rng = rng or _rng()
    ys, xs = np.where(obj_mask > 0)
    if len(xs) == 0:
        return img, mask

    # 在物体内随机选一点作为缺陷中心，向内收缩避免贴边
    h, w = img.shape[:2]
    keep = (xs > 18) & (xs < w - 18) & (ys > 18) & (ys < h - 18)
    xs, ys = xs[keep], ys[keep]
    if len(xs) == 0:
        return img, mask

    i = int(rng.integers(0, len(xs)))
    px, py = int(xs[i]), int(ys[i])

    if kind == "large":
        # 大破损：粗长划痕，对比度高（物体 185 → 缺陷 ~55）
        thickness = int(rng.integers(4, 7))
        length = int(rng.integers(35, 60))
        color = int(rng.integers(45, 70))
    else:
        # 小划痕：细短，对比度略低（更难检）
        thickness = int(rng.integers(2, 4))
        length = int(rng.integers(20, 35))
        color = int(rng.integers(75, 105))

    angle = float(rng.uniform(0, np.pi))
    dx = int(np.cos(angle) * length / 2)
    dy = int(np.sin(angle) * length / 2)
    p1 = (max(0, px - dx), max(0, py - dy))
    p2 = (min(w - 1, px + dx), min(h - 1, py + dy))

    cv2.line(img, p1, p2, int(color), thickness, lineType=cv2.LINE_AA)
    cv2.line(mask, p1, p2, 255, thickness, lineType=cv2.LINE_AA)
    return img, mask


def add_contamination(img, mask, obj_mask, rng=None):
    """在物体区域内加污渍（一处或多处）。"""
    rng = rng or _rng()
    h, w = img.shape[:2]
    ys, xs = np.where(obj_mask > 0)
    keep = (xs > 25) & (xs < w - 25) & (ys > 25) & (ys < h - 25)
    xs, ys = xs[keep], ys[keep]
    if len(xs) == 0:
        return img, mask

    n_spots = int(rng.integers(1, 3))
    for _ in range(n_spots):
        i = int(rng.integers(0, len(xs)))
        cx, cy = int(xs[i]), int(ys[i])
        rx = int(rng.integers(10, 20))
        ry = int(rng.integers(10, 20))
        color = int(rng.integers(55, 85))
        angle = float(rng.uniform(0, 180))
        cv2.ellipse(img, (cx, cy), (rx, ry), angle, 0, 360, int(color), -1)
        cv2.ellipse(mask, (cx, cy), (rx, ry), angle, 0, 360, 255, -1)
    return img, mask


def save(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


def generate_category(root, category, n_train_good=20, n_test_good=15, n_defect=15):
    rng = np.random.default_rng(SEED)
    cat_root = Path(root) / category

    splits = {
        "train/good": n_train_good,
        "test/good": n_test_good,
        "test/broken_large": n_defect,
        "test/broken_small": n_defect,
        "test/contamination": n_defect,
    }

    for split, count in splits.items():
        defect_type = split.split("/")[-1]
        for i in range(count):
            img, obj_mask = make_base_image(rng)
            gt = np.zeros(img.shape, dtype=np.uint8)

            if defect_type == "broken_large":
                img, gt = add_scratch(img, gt, obj_mask, "large", rng)
            elif defect_type == "broken_small":
                img, gt = add_scratch(img, gt, obj_mask, "small", rng)
            elif defect_type == "contamination":
                img, gt = add_contamination(img, gt, obj_mask, rng)

            save(img, str(cat_root / split / f"{i:03d}.png"))
            if defect_type != "good":
                save(gt, str(cat_root / "ground_truth" / defect_type / f"{i:03d}_mask.png"))

    total = sum(splits.values())
    print(f"[完成] {category}: {n_train_good} train-good + {n_test_good} test-good + "
          f"3 x {n_defect} defect = {total} 张  ->  {cat_root}")


def main():
    out_root = Path("data/mvtec_ad_demo")
    generate_category(out_root, "bottle")
    print("\n生成完成。下一步：")
    print("  python 01_traditional_opencv.py --data data/mvtec_ad_demo --category bottle --method diff")


if __name__ == "__main__":
    main()
