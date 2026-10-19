"""
04 - 汇总所有实验结果，生成「面试速查数字卡」

把 01/02/03 跑出来的 json 汇总成一张表，直接打印 + 存 md。
面试前看这一张表就够了。

用法：python 04_interview_numbers.py
"""

import os
import json

FILES = {
    "01 传统方法（MVTec AD）": "outputs/01_traditional/metrics.json",
    "02 YOLOv8（NEU-DET）": "outputs/02_yolo_metrics.json",
    "03 OpenCV DNN 推理": "outputs/03_dnn_metrics.json",
}


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    lines = []
    lines.append("# 面试速查数字卡\n")
    lines.append("> 由 04_interview_numbers.py 自动生成。面试前扫一眼这张表即可。\n")

    # ---- 01 传统方法 ----
    d1 = load(FILES["01 传统方法（MVTec AD）"])
    lines.append("\n## 一、传统 OpenCV 流程（MVTec AD）\n")
    if d1:
        lines.append("| 类别 | 像素 IoU | Precision | Recall | 误检率 | 漏检率 |")
        lines.append("|---|---|---|---|---|---|")
        for r in (d1 if isinstance(d1, list) else [d1]):
            lines.append(
                f"| {r['category']} | {r['pixel_iou']:.3f} | {r['pixel_precision']:.3f} "
                f"| {r['pixel_recall']:.3f} | {r['false_positive_rate']:.1f}% "
                f"| {r['miss_rate']:.1f}% |"
            )
        avg_iou = sum(r["pixel_iou"] for r in d1) / len(d1) if isinstance(d1, list) else d1["pixel_iou"]
        avg_fp = sum(r["false_positive_rate"] for r in d1) / len(d1) if isinstance(d1, list) else d1["false_positive_rate"]
        avg_miss = sum(r["miss_rate"] for r in d1) / len(d1) if isinstance(d1, list) else d1["miss_rate"]
        lines.append(f"\n**面试口径**：传统方法像素级 IoU 约 **{avg_iou:.2f}**；"
                     f"图像级误检率约 **{avg_fp:.0f}%**、漏检率约 **{avg_miss:.0f}%**。")
        lines.append("误检主要来自光照不均与正常纹理被判为异常——这也是我理解到「打光比算法更重要」的原因。")
    else:
        lines.append("_未运行 01，先跑：python 01_traditional_opencv.py_")

    # ---- 02 YOLO ----
    d2 = load(FILES["02 YOLOv8（NEU-DET）"])
    lines.append("\n## 二、YOLOv8 深度学习（NEU-DET，6 类钢材缺陷）\n")
    if d2:
        lines.append("| 指标 | 数值 |")
        lines.append("|---|---|")
        lines.append(f"| 模型 | {d2['model']} |")
        lines.append(f"| mAP@0.5 | **{d2['map50']:.3f}** |")
        lines.append(f"| mAP@0.5:0.95 | {d2['map50_95']:.3f} |")
        lines.append(f"| 训练轮数 | {d2['epochs']} |")
        lines.append(f"| 输入尺寸 | {d2['imgsz']} |")
        lines.append(f"\n**面试口径**：NEU-DET 6 类缺陷 mAP@0.5 约 **{d2['map50']:.2f}**。")
    else:
        lines.append("_未运行 02，先跑：python 02_yolo_neu_det.py_")

    # ---- 03 推理延迟 ----
    d3 = load(FILES["03 OpenCV DNN 推理"])
    lines.append("\n## 三、部署性能（OpenCV DNN，CPU）\n")
    if d3:
        lines.append("| 指标 | 数值 |")
        lines.append("|---|---|")
        lines.append(f"| 平均延迟 | **{d3['mean_ms']:.1f} ms** |")
        lines.append(f"| P95 延迟 | {d3['p95_ms']:.1f} ms |")
        lines.append(f"| 吞吐 | {d3['fps']:.1f} FPS |")
        lines.append(f"| 设备 | {d3['device']} |")
        verdict = "满足一般产线节拍（通常 < 200ms）" if d3["mean_ms"] < 200 else "偏慢，需换更小模型或加 GPU"
        lines.append(f"\n**面试口径**：导出 ONNX 后用 OpenCV DNN 在 CPU 上推理，"
                     f"平均 **{d3['mean_ms']:.0f} ms/张**、约 **{d3['fps']:.0f} FPS**，{verdict}。")
    else:
        lines.append("_未运行 03，先跑：python 03_onnx_dnn_inference.py_")

    # ---- 金句 ----
    lines.append("\n## 四、面试金句（背下来）\n")
    lines.append("1. **为什么两套都做**：产线上缺陷类型固定、没标注数据时，传统方法加规则反而更稳、"
                 "可解释、客户能自己调；遇到多类复杂缺陷就得上深度学习。")
    lines.append("2. **最大收获**：理解了精度/速度/误检/漏检的工程权衡，以及产线节拍对算法选型的硬约束——"
                 "算法再准，超过节拍就上不了线。")
    lines.append("3. **打光比算法重要**：同样的算法，换个打光方式效果可能天差地别。"
                 "同轴光打反光金属，背光做尺寸测量，低角度光突出划痕，漫反射光做颜色检测。")

    out = "\n".join(lines)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/面试速查数字卡.md", "w", encoding="utf-8") as f:
        f.write(out)
    print(out)
    print("\n>>> 已保存到 outputs/面试速查数字卡.md")


if __name__ == "__main__":
    main()
