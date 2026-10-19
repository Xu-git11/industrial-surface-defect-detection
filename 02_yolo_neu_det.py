"""
02 - YOLOv8 训练 NEU-DET 钢材表面缺陷（6 类）

6 类缺陷：crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches
输出：mAP@0.5 / mAP@0.5:0.95，权重保存到 runs/detect/neu_det/

用法：
    python 02_yolo_neu_det.py --data data/NEU-DET --epochs 50 --imgsz 640
    python 02_yolo_neu_det.py --data data/NEU-DET --epochs 50 --model yolov8n.pt   # 更快（CPU 也能跑）
"""

import argparse
import json
import os

# NEU-DET 的 6 类缺陷（顺序决定 class id，务必与你的标注一致）
NEU_CLASSES = [
    "crazing",          # 裂纹
    "inclusion",        # 夹杂物
    "patches",          # 斑块
    "pitted_surface",   # 麻点
    "rolled-in_scale",  # 轧入氧化皮
    "scratches",        # 划痕
]


def write_data_yaml(data_root, out_path="data/neu_det.yaml"):
    """
    生成 Ultralytics 需要的 data.yaml。
    期望数据目录结构：
        NEU-DET/
          train/images/*.jpg
          train/labels/*.txt      (YOLO 格式：cls cx cy w h，已归一化)
          valid/images/*.jpg
          valid/labels/*.txt
    如果你的标注是 VOC(XML)，先跑 utils/voc2yolo.py 转换。
    """
    content = f"""# NEU-DET 钢材表面缺陷检测
path: {os.path.abspath(data_root)}
train: train/images
val: valid/images

nc: {len(NEU_CLASSES)}
names: {NEU_CLASSES}
"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK] 已生成 {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/NEU-DET", help="NEU-DET 根目录")
    ap.add_argument("--model", default="yolov8n.pt", help="yolov8n/s/m/l/x.pt，n 最快")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="", help="留空自动选择；CPU 填 cpu")
    args = ap.parse_args()

    # 校验目录
    train_img = os.path.join(args.data, "train", "images")
    if not os.path.isdir(train_img):
        print(f"[错误] 找不到 {train_img}")
        print("请确认目录结构：NEU-DET/train/images + NEU-DET/train/labels")
        print("若标注是 VOC(XML)，先跑：python utils/voc2yolo.py --src <xml目录> --dst <labels目录>")
        return

    yaml_path = write_data_yaml(args.data)

    from ultralytics import YOLO

    print(f"\n开始训练：model={args.model} epochs={args.epochs} imgsz={args.imgsz}")
    model = YOLO(args.model)
    results = model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device if args.device else None,
        project="runs/detect",
        name="neu_det",
        exist_ok=True,
        patience=20,
    )

    # 验证集评估
    print("\n在验证集上评估...")
    metrics = model.val()
    summary = dict(
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        map50=float(metrics.box.map50),
        map50_95=float(metrics.box.map),
        nc=len(NEU_CLASSES),
        names=NEU_CLASSES,
    )
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/02_yolo_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print(f"mAP@0.5      = {summary['map50']:.3f}")
    print(f"mAP@0.5:0.95 = {summary['map50_95']:.3f}")
    print("=" * 50)
    print(">>> 面试用数字：把 mAP@0.5 记下来，填进 README 话术。")
    print(f">>> 权重路径：runs/detect/neu_det/weights/best.pt")
    print(">>> 下一步：python 03_onnx_dnn_inference.py")


if __name__ == "__main__":
    main()
